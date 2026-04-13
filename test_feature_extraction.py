"""
Test feature extraction for both image and text modalities.

Loads preprocessed MIMIC-CXR data, runs it through:
  - Image encoder (ResNet-50 pretrained on ImageNet)
  - Text encoder (BiomedBERT)
  - Fusion transformer

Prints embedding shapes and cosine similarities to verify everything works.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MIMICCXRPreprocessed
from model import ImageEncoder, TextEncoder, MultimodalFusionModel


def test_individual_encoders(dataset):
    """Test image and text encoders separately."""
    print("=" * 60)
    print("1. Testing Individual Encoders")
    print("=" * 60)

    sample = dataset[0]
    image = sample["image"].unsqueeze(0)           # (1, 3, 224, 224)
    input_ids = sample["input_ids"].unsqueeze(0)    # (1, 512)
    attention_mask = sample["attention_mask"].unsqueeze(0)  # (1, 512)

    print(f"\nInput shapes:")
    print(f"  Image:          {image.shape}")
    print(f"  Input IDs:      {input_ids.shape}")
    print(f"  Attention mask:  {attention_mask.shape}")

    # --- Image Encoder ---
    print("\n--- Image Encoder (ResNet-50) ---")
    img_encoder = ImageEncoder(backbone="resnet50", embed_dim=256, freeze=True)
    img_encoder.eval()
    with torch.no_grad():
        img_embed = img_encoder(image)
    print(f"  Image embedding: {img_embed.shape}")  # (1, 256)
    print(f"  Embedding stats: mean={img_embed.mean():.4f}, "
          f"std={img_embed.std():.4f}, "
          f"min={img_embed.min():.4f}, max={img_embed.max():.4f}")

    # --- Text Encoder ---
    print("\n--- Text Encoder (BiomedBERT) ---")
    txt_encoder = TextEncoder(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim=256,
        freeze=True,
    )
    txt_encoder.eval()
    with torch.no_grad():
        txt_embed = txt_encoder(input_ids, attention_mask)
    print(f"  Text embedding:  {txt_embed.shape}")  # (1, 256)
    print(f"  Embedding stats: mean={txt_embed.mean():.4f}, "
          f"std={txt_embed.std():.4f}, "
          f"min={txt_embed.min():.4f}, max={txt_embed.max():.4f}")

    return img_encoder, txt_encoder


def test_fusion_model(dataset):
    """Test the full multimodal fusion pipeline."""
    print("\n" + "=" * 60)
    print("2. Testing Multimodal Fusion Model")
    print("=" * 60)

    model = MultimodalFusionModel(
        image_backbone="resnet50",
        text_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        freeze_encoders=True,
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    # Process a mini-batch
    loader = DataLoader(dataset, batch_size=min(3, len(dataset)), shuffle=False)
    batch = next(iter(loader))

    print(f"\n  Batch size: {batch['image'].shape[0]}")

    with torch.no_grad():
        fused, img_emb, txt_emb = model(
            batch["image"], batch["input_ids"], batch["attention_mask"]
        )

    print(f"\n  Output shapes:")
    print(f"    Image embeddings:  {img_emb.shape}")
    print(f"    Text embeddings:   {txt_emb.shape}")
    print(f"    Fused embeddings:  {fused.shape}")

    # Cosine similarity between modalities
    cos_sim = torch.nn.functional.cosine_similarity(img_emb, txt_emb, dim=1)
    print(f"\n  Cosine similarity (image vs text) per sample:")
    for i, sim in enumerate(cos_sim):
        pid = batch.get("patient_id", ["?"] * len(cos_sim))[i]
        sid = batch.get("study_id", ["?"] * len(cos_sim))[i]
        print(f"    {pid}/{sid}: {sim.item():.4f}")

    # Cosine similarity between fused embeddings
    if fused.shape[0] >= 2:
        fused_sim = torch.nn.functional.cosine_similarity(
            fused[0].unsqueeze(0), fused[1].unsqueeze(0)
        )
        print(f"\n  Cosine similarity (fused[0] vs fused[1]): {fused_sim.item():.4f}")

    return model


def test_all_samples(dataset):
    """Extract features for all samples and print summary."""
    print("\n" + "=" * 60)
    print("3. Feature Extraction on All Samples")
    print("=" * 60)

    model = MultimodalFusionModel(
        image_backbone="resnet50",
        text_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        freeze_encoders=True,
    )
    model.eval()

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        fused, img_emb, txt_emb = model(
            batch["image"], batch["input_ids"], batch["attention_mask"]
        )

    print(f"\n  Processed {len(dataset)} samples")
    print(f"\n  {'Sample':<45} {'ImgNorm':>8} {'TxtNorm':>8} {'FusedNorm':>10} {'Cos(I,T)':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    cos_sims = torch.nn.functional.cosine_similarity(img_emb, txt_emb, dim=1)
    for i in range(len(dataset)):
        pid = batch.get("patient_id", ["?"] * len(dataset))[i]
        sid = batch.get("study_id", ["?"] * len(dataset))[i]
        label = f"{pid}/{sid}"
        print(f"  {label:<45} "
              f"{img_emb[i].norm():.4f}   "
              f"{txt_emb[i].norm():.4f}   "
              f"{fused[i].norm():.4f}     "
              f"{cos_sims[i].item():.4f}")

    print(f"\n  Mean cosine similarity (image vs text): {cos_sims.mean():.4f}")


if __name__ == "__main__":
    preprocessed_dir = (
        sys.argv[1] if len(sys.argv) > 1
        else "/data/amciilab/xinran/mimic_cxr_preprocessed_test"
    )

    print(f"Loading data from: {preprocessed_dir}")
    dataset = MIMICCXRPreprocessed(
        preprocessed_dir=preprocessed_dir,
        augment=False,
    )

    test_individual_encoders(dataset)
    test_fusion_model(dataset)
    test_all_samples(dataset)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
