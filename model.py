"""
Multimodal Encoder Architecture
Image encoder (DINOv2 / ResNet-50) + Text encoder (BiomedBERT) + Transformer fusion.
Designed for federated learning research on MIMIC-CXR.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ImageEncoder(nn.Module):
    """
    Visual feature extractor using DINOv2 or ResNet-50.

    Args:
        backbone: "dinov2" or "resnet50"
        embed_dim: Output projection dimension
        freeze: Whether to freeze backbone weights
    """

    def __init__(self, backbone="dinov2", embed_dim=256, freeze=False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "dinov2":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            feature_dim = 768  # ViT-B/14 output
        elif backbone == "resnet50":
            import torchvision.models as models
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove the final FC layer, keep up to avgpool
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) image tensor
        Returns:
            (B, embed_dim) image embedding
        """
        if self.backbone_name == "dinov2":
            features = self.backbone(x)  # (B, 768) CLS token
        else:
            features = self.backbone(x).squeeze(-1).squeeze(-1)  # (B, 2048)

        return self.projector(features)


class TextEncoder(nn.Module):
    """
    Text feature extractor using a pretrained biomedical language model.

    Args:
        model_name: HuggingFace model name
        embed_dim: Output projection dimension
        freeze: Whether to freeze backbone weights
    """

    def __init__(
        self,
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim=256,
        freeze=False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        feature_dim = self.backbone.config.hidden_size  # typically 768

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            (B, embed_dim) text embedding (CLS token projected)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768) CLS token
        return self.projector(cls_embedding)


class MultimodalFusionModel(nn.Module):
    """
    Combines image and text encoders with a transformer fusion layer.

    Architecture:
        Image  -> ImageEncoder  -> img_embed  (B, embed_dim)
        Text   -> TextEncoder   -> txt_embed  (B, embed_dim)
        [img_embed, txt_embed]  -> TransformerEncoder -> fused (B, embed_dim)

    Args:
        image_backbone:  "dinov2" or "resnet50"
        text_model_name: HuggingFace model name for text encoder
        embed_dim:       Shared embedding dimension
        num_heads:       Attention heads in fusion transformer
        num_layers:      Number of fusion transformer layers
        freeze_encoders: Freeze both encoder backbones
    """

    def __init__(
        self,
        image_backbone="dinov2",
        text_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        freeze_encoders=False,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            backbone=image_backbone, embed_dim=embed_dim, freeze=freeze_encoders
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name, embed_dim=embed_dim, freeze=freeze_encoders
        )

        # Modality type embeddings (learned)
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.text_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head (customize for your downstream task)
        self.output_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, image, input_ids, attention_mask):
        """
        Args:
            image:          (B, 3, 224, 224)
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            fused_embedding: (B, embed_dim) joint representation
            img_embed:       (B, embed_dim) image-only embedding
            txt_embed:       (B, embed_dim) text-only embedding
        """
        img_embed = self.image_encoder(image)    # (B, embed_dim)
        txt_embed = self.text_encoder(input_ids, attention_mask)  # (B, embed_dim)

        # Add modality type embeddings and stack as sequence of 2 tokens
        img_token = img_embed.unsqueeze(1) + self.image_type_embed  # (B, 1, embed_dim)
        txt_token = txt_embed.unsqueeze(1) + self.text_type_embed   # (B, 1, embed_dim)

        # Concatenate: (B, 2, embed_dim)
        multimodal_seq = torch.cat([img_token, txt_token], dim=1)

        # Transformer fusion
        fused = self.fusion_transformer(multimodal_seq)  # (B, 2, embed_dim)

        # Pool: mean over the 2 tokens
        fused_embedding = fused.mean(dim=1)  # (B, embed_dim)
        fused_embedding = self.output_head(fused_embedding)

        return fused_embedding, img_embed, txt_embed


if __name__ == "__main__":
    # Quick shape verification with random data
    model = MultimodalFusionModel(
        image_backbone="resnet50",  # use resnet50 for quick test (no download)
        embed_dim=256,
    )

    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_ids = torch.randint(0, 1000, (batch_size, 512))
    dummy_mask = torch.ones(batch_size, 512, dtype=torch.long)

    fused, img_emb, txt_emb = model(dummy_image, dummy_ids, dummy_mask)
    print(f"Fused embedding: {fused.shape}")    # (4, 256)
    print(f"Image embedding: {img_emb.shape}")  # (4, 256)
    print(f"Text embedding:  {txt_emb.shape}")  # (4, 256)
