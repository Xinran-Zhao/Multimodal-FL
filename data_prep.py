"""
Indiana University CXR Dataset Preparation
Parses indiana_reports.csv and indiana_projections.csv to create a unified
prepared_data.csv with 8 multi-label disease annotations for FL experiments.
"""

import os
import csv
import json
import re
import argparse
import random
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# 8 target disease labels (top-frequency MeSH root categories + No_Finding)
# ---------------------------------------------------------------------------
LABEL_NAMES = [
    "No_Finding",
    "Cardiomegaly",
    "Atelectasis",
    "Pleural_Effusion",
    "Opacity",
    "Calcinosis",
    "Calcified_Granuloma",
    "Airspace_Disease",
]

# Mapping: MeSH root term  →  our label name
# Some MeSH entries use slightly different wordings; we normalise here.
MESH_TO_LABEL = {
    "cardiomegaly":         "Cardiomegaly",
    "pulmonary atelectasis": "Atelectasis",
    "atelectasis":          "Atelectasis",
    "pleural effusion":     "Pleural_Effusion",
    "opacity":              "Opacity",
    "calcinosis":           "Calcinosis",
    "calcified granuloma":  "Calcified_Granuloma",
    "airspace disease":     "Airspace_Disease",
}


def parse_mesh_labels(mesh_string: str) -> dict:
    """
    Convert a semicolon-separated MeSH string into an 8-dim binary dict.

    Example input:  "Cardiomegaly/borderline;Pulmonary Artery/enlarged"
    Output: {label: 0 or 1 for each of the 8 labels}
    """
    labels = {name: 0 for name in LABEL_NAMES}

    mesh_string = mesh_string.strip()
    if mesh_string.lower() == "normal" or mesh_string == "":
        labels["No_Finding"] = 1
        return labels

    entries = mesh_string.split(";")
    matched_any = False
    for entry in entries:
        # Take root term (before first '/') and lowercase
        root = entry.split("/")[0].strip().lower()
        if root in MESH_TO_LABEL:
            labels[MESH_TO_LABEL[root]] = 1
            matched_any = True

    # If none of our 8 target diseases were found but it's not "normal",
    # we still keep the sample — it just has all-zeros for our 8 labels
    # (the diseases present are outside our top-8 scope)
    return labels


def prepare_dataset(data_dir: str, output_dir: str, seed: int = 42):
    """
    Main preparation pipeline:
    1. Load projections CSV → filter to frontal views
    2. Load reports CSV → parse MeSH labels + extract text
    3. Join on uid
    4. Stratified train/val/test split (70/15/15)
    5. Save prepared_data.csv
    """
    random.seed(seed)
    np.random.seed(seed)

    reports_path = os.path.join(data_dir, "indiana_reports.csv")
    proj_path = os.path.join(data_dir, "indiana_projections.csv")
    images_dir = os.path.join(data_dir, "images", "images_normalized")

    # ------------------------------------------------------------------
    # 1. Load frontal projections:  uid -> [list of frontal image filenames]
    # ------------------------------------------------------------------
    uid_to_frontal_images = {}
    with open(proj_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["projection"].strip() == "Frontal":
                uid = row["uid"].strip()
                fname = row["filename"].strip()
                # Verify image exists
                if os.path.isfile(os.path.join(images_dir, fname)):
                    uid_to_frontal_images.setdefault(uid, []).append(fname)

    print(f"Frontal images found for {len(uid_to_frontal_images)} studies")

    # ------------------------------------------------------------------
    # 2. Load reports and parse labels + text
    # ------------------------------------------------------------------
    samples = []
    with open(reports_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"].strip()
            if uid not in uid_to_frontal_images:
                continue  # no frontal image for this study

            # Parse MeSH labels
            mesh_str = row.get("MeSH", "").strip()
            labels = parse_mesh_labels(mesh_str)

            # Extract text: prefer findings, fallback to impression
            findings = row.get("findings", "").strip()
            impression = row.get("impression", "").strip()

            if findings and findings.lower() != "none":
                text = findings
            elif impression and impression.lower() != "none":
                text = impression
            else:
                text = ""  # no text available

            # Clean text
            text = re.sub(r"X{2,}", "[DEIDENTIFIED]", text)
            text = re.sub(r"\s+", " ", text).strip()

            # Create one sample per frontal image (most studies have 1 frontal)
            for img_fname in uid_to_frontal_images[uid]:
                sample = {
                    "uid": uid,
                    "image_path": img_fname,
                    "findings_text": text,
                    "has_text": 1 if len(text) > 0 else 0,
                }
                sample.update(labels)
                samples.append(sample)

    print(f"Total samples (frontal, with reports): {len(samples)}")

    # ------------------------------------------------------------------
    # 3. Train / Val / Test split  (70 / 15 / 15)  stratified by uid
    #    (all images from same study go to same split)
    # ------------------------------------------------------------------
    uids = list(set(s["uid"] for s in samples))
    random.shuffle(uids)

    n = len(uids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    # rest goes to test

    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train : n_train + n_val])
    test_uids = set(uids[n_train + n_val :])

    for s in samples:
        if s["uid"] in train_uids:
            s["split"] = "train"
        elif s["uid"] in val_uids:
            s["split"] = "val"
        else:
            s["split"] = "test"

    # ------------------------------------------------------------------
    # 4. Save prepared_data.csv
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "prepared_data.csv")

    fieldnames = ["uid", "image_path", "findings_text", "has_text", "split"] + LABEL_NAMES
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(s)

    print(f"Saved {len(samples)} samples to {out_path}")

    # ------------------------------------------------------------------
    # 5. Print summary statistics
    # ------------------------------------------------------------------
    split_counts = Counter(s["split"] for s in samples)
    print(f"\nSplit distribution:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {split_counts[split]}")

    print(f"\nLabel distribution (all samples):")
    for label in LABEL_NAMES:
        pos = sum(1 for s in samples if s[label] == 1)
        print(f"  {label}: {pos} ({100*pos/len(samples):.1f}%)")

    text_count = sum(1 for s in samples if s["has_text"] == 1)
    print(f"\nSamples with text: {text_count} ({100*text_count/len(samples):.1f}%)")

    # Save stats as JSON too
    stats = {
        "total_samples": len(samples),
        "splits": dict(split_counts),
        "label_counts": {
            label: sum(1 for s in samples if s[label] == 1)
            for label in LABEL_NAMES
        },
        "samples_with_text": text_count,
        "num_labels": len(LABEL_NAMES),
        "label_names": LABEL_NAMES,
    }
    stats_path = os.path.join(output_dir, "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Indiana CXR dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr",
        help="Path to downloaded Indiana CXR dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared",
        help="Output directory for prepared_data.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.data_dir, args.output_dir, args.seed)
