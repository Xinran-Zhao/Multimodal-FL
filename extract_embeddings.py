"""
Pre-extract and save embeddings for all preprocessed MIMIC-CXR samples.

Runs frozen image encoder (ResNet-50) and text encoder (BiomedBERT) on
all samples, saves embeddings as .pt files for use in FL training.

Output structure:
    <output_dir>/
    ├── image_embeddings.pt     # (N, embed_dim) tensor
    ├── text_embeddings.pt      # (N, embed_dim) tensor
    ├── fused_embeddings.pt     # (N, embed_dim) tensor
    └── metadata.csv            # patient_id, study_id, image_path per row

Usage:
    python extract_embeddings.py
    python extract_embeddings.py --preprocessed_dir <path> --output_dir <path>
    python extract_embeddings.py --embed_dim 512 --batch_size 16
"""

import os
import sys
import csv
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MIMICCXRPreprocessed
from model import ImageEncoder, TextEncoder, MultimodalFusionModel


def extract_and_save(
    preprocessed_dir,
    output_dir,
    image_backbone="resnet50",
    text_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    embed_dim=256,
    batch_size=8,
    num_workers=0,
    device="cpu",
):
    """Extract embeddings from all samples and save to disk."""

    # --- Load dataset ---
    dataset = MIMICCXRPreprocessed(
        preprocessed_dir=preprocessed_dir,
        tokenizer_name=text_model_name,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # keep order aligned with metadata
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # --- Build model ---
    model = MultimodalFusionModel(
        image_backbone=image_backbone,
        text_model_name=text_model_name,
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=2,
        freeze_encoders=True,
    )
    model.to(device)
    model.eval()

    # --- Extract ---
    all_img_emb = []
    all_txt_emb = []
    all_fused_emb = []
    all_metadata = []

    total_batches = (len(dataset) + batch_size - 1) // batch_size

    print(f"Extracting embeddings from {len(dataset)} samples "
          f"({total_batches} batches, embed_dim={embed_dim})...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            fused, img_emb, txt_emb = model(image, input_ids, attention_mask)

            all_img_emb.append(img_emb.cpu())
            all_txt_emb.append(txt_emb.cpu())
            all_fused_emb.append(fused.cpu())

            # Collect metadata
            for j in range(image.shape[0]):
                all_metadata.append({
                    "patient_id": batch["patient_id"][j],
                    "study_id": batch["study_id"][j],
                })

            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(f"  [{i+1}/{total_batches}] batches processed")

    # --- Concatenate ---
    img_embeddings = torch.cat(all_img_emb, dim=0)    # (N, embed_dim)
    txt_embeddings = torch.cat(all_txt_emb, dim=0)    # (N, embed_dim)
    fused_embeddings = torch.cat(all_fused_emb, dim=0) # (N, embed_dim)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, "image_embeddings.pt")
    txt_path = os.path.join(output_dir, "text_embeddings.pt")
    fused_path = os.path.join(output_dir, "fused_embeddings.pt")
    meta_path = os.path.join(output_dir, "metadata.csv")

    torch.save(img_embeddings, img_path)
    torch.save(txt_embeddings, txt_path)
    torch.save(fused_embeddings, fused_path)

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "study_id"])
        writer.writeheader()
        writer.writerows(all_metadata)

    # --- Summary ---
    print(f"\nSaved embeddings to {output_dir}/")
    print(f"  image_embeddings.pt:  {img_embeddings.shape}")
    print(f"  text_embeddings.pt:   {txt_embeddings.shape}")
    print(f"  fused_embeddings.pt:  {fused_embeddings.shape}")
    print(f"  metadata.csv:         {len(all_metadata)} rows")

    # Verify alignment
    print(f"\nVerification:")
    print(f"  Shapes match metadata: "
          f"{img_embeddings.shape[0] == txt_embeddings.shape[0] == len(all_metadata)}")
    cos_sim = torch.nn.functional.cosine_similarity(img_embeddings, txt_embeddings)
    print(f"  Mean cos(image, text): {cos_sim.mean():.4f}")

    return img_embeddings, txt_embeddings, fused_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save embeddings for FL training"
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="/data/amciilab/xinran/mimic_cxr_preprocessed_test",
        help="Path to preprocessed data (from preprocess.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save embeddings. Default: <preprocessed_dir>/embeddings/",
    )
    parser.add_argument("--image_backbone", type=str, default="resnet50",
                        choices=["resnet50", "dinov2"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.preprocessed_dir, "embeddings")

    extract_and_save(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        image_backbone=args.image_backbone,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
