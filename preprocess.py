"""
Preprocess MIMIC-CXR dataset.

Converts raw DICOM images + radiology reports into a ready-to-train format:
  - Images: DICOM -> CLAHE -> resize 224x224 -> saved as PNG
  - Text:   Reports -> extract FINDINGS -> clean -> saved in a manifest CSV

Output structure:
    <output_dir>/
    ├── images/
    │   ├── p10000032_s50414267_02aa804e.png
    │   ├── ...
    ├── manifest.csv     (columns: image_path, findings_text, patient_id, study_id)
    └── stats.json       (dataset statistics)

Usage:
    python preprocess.py --data_root <path_to_mimic_cxr_files> --output_dir <output>
    python preprocess.py  # uses default paths
"""

import os
import re
import sys
import csv
import json
import glob
import time
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image

# These are only needed at the worker level; imported inside worker to
# keep the main process lightweight.
_pydicom = None
_cv2 = None


def _init_worker():
    """Initialize heavy imports once per worker process."""
    global _pydicom, _cv2
    import pydicom
    import cv2
    _pydicom = pydicom
    _cv2 = cv2


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors dataset.py logic, but saves to disk)
# ---------------------------------------------------------------------------

def load_and_preprocess_dicom(dcm_path, image_size=224, use_clahe=True,
                              frontal_only=True):
    """
    Load a DICOM file and return a preprocessed PIL Image (grayscale, resized).

    Steps:
        1. Read DICOM pixel array
        2. Filter by view position (frontal only: AP/PA)
        3. Handle MONOCHROME1 inversion
        4. Normalize to 8-bit (0-255)
        5. Apply CLAHE contrast enhancement (optional)
        6. Resize to image_size x image_size
    """
    global _pydicom, _cv2
    try:
        dcm = _pydicom.dcmread(dcm_path)
    except Exception as e:
        return None, str(e)

    # Filter: keep only frontal views (AP or PA)
    if frontal_only:
        view = getattr(dcm, "ViewPosition", "").upper()
        if view not in ("AP", "PA"):
            return None, f"non-frontal view ({view})"

    try:
        img = dcm.pixel_array.astype(np.float32)
    except Exception as e:
        return None, str(e)

    # Handle inverted photometric interpretation
    if hasattr(dcm, "PhotometricInterpretation"):
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img

    # Normalize to 0-255
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-8:
        return None, "constant image (no contrast)"
    img = (img - img_min) / (img_max - img_min) * 255.0
    img = img.astype(np.uint8)

    # CLAHE
    if use_clahe:
        clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # Resize
    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)

    return pil_img, None


# ---------------------------------------------------------------------------
# Text preprocessing (mirrors dataset.py logic)
# ---------------------------------------------------------------------------

def extract_findings(report_text, missing_strategy="skip"):
    """
    Extract and clean the FINDINGS section from a radiology report.

    Args:
        report_text:      Raw report string
        missing_strategy: "skip" | "impression" | "full"

    Returns:
        Cleaned text string, or None if skipped.
    """
    # Try FINDINGS section
    match = re.search(
        r"FINDINGS\s*:(.*?)(?:IMPRESSION\s*:|$)", report_text, re.DOTALL
    )

    if match:
        findings = match.group(1).strip()
        cleaned = re.sub(r"_+", "", findings).strip()
        if len(cleaned) == 0:
            return _handle_missing(report_text, missing_strategy)
        return _clean_text(findings)
    else:
        return _handle_missing(report_text, missing_strategy)


def _handle_missing(report_text, strategy):
    if strategy == "skip":
        return None
    elif strategy == "impression":
        match = re.search(r"IMPRESSION\s*:(.*?)$", report_text, re.DOTALL)
        if match:
            return _clean_text(match.group(1).strip())
        return None
    elif strategy == "full":
        match = re.search(
            r"(?:INDICATION|HISTORY|REASON FOR EXAM)\s*:.*?\n(.*)",
            report_text, re.DOTALL,
        )
        if match:
            return _clean_text(match.group(1).strip())
        return _clean_text(report_text.strip())
    return None


def _clean_text(text):
    text = re.sub(r"_+", "[DEIDENTIFIED]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 0 else None


# ---------------------------------------------------------------------------
# Per-study processing (called by worker processes)
# ---------------------------------------------------------------------------

def process_study(args):
    """
    Process one study: preprocess all DICOMs + extract report text.
    Returns a list of (image_filename, findings_text, patient_id, study_id, word_count)
    tuples, or empty list if filtered out.
    """
    (study_dir, report_path, output_image_dir,
     image_size, use_clahe, missing_strategy, min_words, max_words,
     frontal_only) = args

    # Extract patient_id and study_id from paths
    patient_id = os.path.basename(os.path.dirname(study_dir))
    study_id = os.path.basename(study_dir)

    # --- Text ---
    try:
        with open(report_path, "r") as f:
            report_text = f.read()
    except Exception:
        return []

    findings = extract_findings(report_text, missing_strategy)
    if findings is None:
        return []

    # --- Word count filter ---
    word_count = len(findings.split())
    if word_count < min_words or word_count > max_words:
        return []

    # --- Images ---
    results = []
    dcm_files = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]

    for dcm_name in dcm_files:
        dcm_path = os.path.join(study_dir, dcm_name)
        pil_img, err = load_and_preprocess_dicom(dcm_path, image_size, use_clahe,
                                                frontal_only)

        if pil_img is None:
            continue

        # Save as PNG: {patient_id}_{study_id}_{dicom_uid_short}.png
        uid_short = dcm_name.replace(".dcm", "")[:8]
        out_name = f"{patient_id}_{study_id}_{uid_short}.png"
        out_path = os.path.join(output_image_dir, out_name)

        pil_img.save(out_path)
        results.append((out_name, findings, patient_id, study_id, word_count))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_studies(data_root, patient_groups=None, patient_ids=None):
    """
    Collect all (study_dir, report_path) pairs from the dataset.

    Args:
        data_root:       Base path to files/ directory
        patient_groups:  Set of group names like {"p10", "p11"}, or None for all
        patient_ids:     Set of patient IDs like {"p10000032"}, or None for all
    """
    studies = []

    # If specific patient IDs given, go directly to their folders (fast)
    if patient_ids:
        for pid in sorted(patient_ids):
            group = pid[:3]  # e.g. "p10" from "p10000032"
            patient_path = os.path.join(data_root, group, pid)
            if not os.path.isdir(patient_path):
                print(f"  Warning: patient dir not found: {patient_path}")
                continue
            studies.extend(_collect_from_patient(patient_path))
        return studies

    # Determine which group dirs to scan
    if patient_groups:
        patient_group_dirs = sorted(
            os.path.join(data_root, g)
            for g in patient_groups
            if os.path.isdir(os.path.join(data_root, g))
        )
    else:
        patient_group_dirs = sorted(glob.glob(os.path.join(data_root, "p1[0-9]")))

    for pg_dir in patient_group_dirs:
        print(f"  Scanning {os.path.basename(pg_dir)}...")
        for patient_dir in sorted(os.listdir(pg_dir)):
            patient_path = os.path.join(pg_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            studies.extend(_collect_from_patient(patient_path))

    return studies


def _collect_from_patient(patient_path):
    """Collect studies from a single patient directory."""
    studies = []
    for entry in sorted(os.listdir(patient_path)):
        study_path = os.path.join(patient_path, entry)
        if os.path.isdir(study_path) and entry.startswith("s"):
            report_path = os.path.join(patient_path, entry + ".txt")
            if os.path.isfile(report_path):
                studies.append((study_path, report_path))
    return studies


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-CXR dataset for multimodal learning"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/amciilab/mahfuz/mimic_cxr_dl_script/"
                "physionet.org/files/mimic-cxr/2.0.0/files",
        help="Path to MIMIC-CXR files directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/amciilab/xinran/mimic_cxr_preprocessed",
        help="Output directory for preprocessed data",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_clahe", action="store_true", default=True)
    parser.add_argument("--no_clahe", action="store_true", default=False)
    parser.add_argument(
        "--missing_findings",
        type=str,
        default="skip",
        choices=["skip", "impression", "full"],
        help="How to handle reports without FINDINGS section",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(8, cpu_count()),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--patient_groups",
        type=str,
        default=None,
        help="Comma-separated patient groups to process, e.g. 'p10,p11'. "
             "Default: all (p10-p19)",
    )
    parser.add_argument(
        "--patient_ids",
        type=str,
        default=None,
        help="Comma-separated specific patient IDs, e.g. 'p10000032,p10000764'. "
             "For quick testing on individual patients.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=10,
        help="Minimum word count for FINDINGS section (default: 10). "
             "Filters out placeholder-only or trivially short findings.",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=150,
        help="Maximum word count for FINDINGS section (default: 150). "
             "Filters out unusually long reports that may be outliers.",
    )
    parser.add_argument(
        "--frontal_only",
        action="store_true",
        default=True,
        help="Keep only frontal views (AP/PA), skip lateral views (default: True).",
    )
    parser.add_argument(
        "--all_views",
        action="store_true",
        default=False,
        help="Include all views (frontal + lateral). Overrides --frontal_only.",
    )
    args = parser.parse_args()

    if args.no_clahe:
        args.use_clahe = False
    if args.all_views:
        args.frontal_only = False

    # --- Setup output directory ---
    output_image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    print(f"Data root:          {args.data_root}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Image size:         {args.image_size}x{args.image_size}")
    print(f"CLAHE:              {args.use_clahe}")
    print(f"Missing findings:   {args.missing_findings}")
    print(f"Findings word range: [{args.min_words}, {args.max_words}]")
    print(f"Frontal only:       {args.frontal_only}")
    print(f"Workers:            {args.num_workers}")
    print()

    # --- Collect all studies ---
    print("Scanning for studies...")
    pg = set(args.patient_groups.split(",")) if args.patient_groups else None
    pids = set(args.patient_ids.split(",")) if args.patient_ids else None
    all_studies = collect_studies(args.data_root, patient_groups=pg, patient_ids=pids)

    print(f"Found {len(all_studies)} studies")

    # --- Build work items ---
    work_items = [
        (study_dir, report_path, output_image_dir,
         args.image_size, args.use_clahe, args.missing_findings,
         args.min_words, args.max_words, args.frontal_only)
        for study_dir, report_path in all_studies
    ]

    # --- Process in parallel ---
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    total_images = 0
    total_skipped_studies = 0
    word_counts = []

    t_start = time.time()

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image_path", "findings_text", "patient_id", "study_id", "word_count"
        ])

        with Pool(
            processes=args.num_workers, initializer=_init_worker
        ) as pool:
            for i, results in enumerate(
                pool.imap_unordered(process_study, work_items, chunksize=32)
            ):
                if not results:
                    total_skipped_studies += 1
                else:
                    for image_name, findings, patient_id, study_id, wc in results:
                        writer.writerow([
                            image_name, findings, patient_id, study_id, wc
                        ])
                        total_images += 1
                        word_counts.append(wc)

                # Progress report every 5000 studies (or every study for small runs)
                processed = i + 1
                if (len(work_items) <= 50
                        or processed % 5000 == 0
                        or processed == len(work_items)):
                    elapsed = time.time() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (len(work_items) - processed) / rate if rate > 0 else 0
                    print(
                        f"  [{processed}/{len(work_items)}] "
                        f"{total_images} images saved, "
                        f"{total_skipped_studies} studies skipped | "
                        f"elapsed: {elapsed:.1f}s"
                    )

    elapsed = time.time() - t_start

    # --- Save statistics ---
    stats = {
        "total_studies_scanned": len(all_studies),
        "total_images_saved": total_images,
        "studies_skipped": total_skipped_studies,
        "missing_findings_strategy": args.missing_findings,
        "findings_word_range": [args.min_words, args.max_words],
        "image_size": args.image_size,
        "clahe": args.use_clahe,
        "processing_time_seconds": round(elapsed, 1),
    }
    if word_counts:
        stats["findings_word_count"] = {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": round(sum(word_counts) / len(word_counts), 1),
            "median": round(sorted(word_counts)[len(word_counts) // 2], 1),
        }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f} seconds")
    print(f"  Images saved:      {total_images}")
    print(f"  Studies skipped:   {total_skipped_studies}")
    print(f"  Word count range:  [{args.min_words}, {args.max_words}]")
    print(f"  Manifest:          {manifest_path}")
    print(f"  Stats:             {stats_path}")
    print(f"  Images dir:        {output_image_dir}")
    if word_counts:
        print(f"  Findings words:    min={min(word_counts)}, "
              f"max={max(word_counts)}, "
              f"mean={sum(word_counts)/len(word_counts):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
