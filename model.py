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


# =========================================================================
# Classification models for FL experiments
# =========================================================================


class ImageClassifier(nn.Module):
    """
    Unimodal image classifier: Frozen ResNet-50 → projection → classifier.

    For FL, the trainable parameters are:
      - image_encoder.projector (Linear + LayerNorm)
      - classifier (Linear)

    These are the "shared parameters" that get aggregated across all clients.
    """

    def __init__(self, num_labels=8, embed_dim=256, backbone="resnet50"):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone=backbone, embed_dim=embed_dim, freeze=True
        )
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, image, **kwargs):
        embed = self.image_encoder(image)      # (B, embed_dim)
        logits = self.classifier(embed)        # (B, num_labels)
        return logits, embed

    def get_shared_params(self):
        """Return state_dict of parameters shared across all FL clients."""
        shared = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                shared[name] = param
        return shared

    def get_shared_state_dict(self):
        """Return state_dict of trainable (shared) parameters."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_shared_state_dict(self, state_dict):
        """Load shared parameters from aggregated state_dict."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)


class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier: Frozen ResNet-50 + Frozen BiomedBERT → fusion → classifier.

    For FL, only the image-side parameters are shared:
      - image_encoder.projector
      - classifier

    Text encoder, fusion transformer, and type embeddings are LOCAL-ONLY
    (not aggregated), since unimodal clients don't have them.
    """

    def __init__(
        self,
        num_labels=8,
        embed_dim=256,
        image_backbone="resnet50",
        text_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        num_heads=8,
        num_fusion_layers=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Encoders (backbones frozen)
        self.image_encoder = ImageEncoder(
            backbone=image_backbone, embed_dim=embed_dim, freeze=True
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name, embed_dim=embed_dim, freeze=True
        )

        # Modality type embeddings (local-only)
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.text_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Fusion transformer (local-only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_fusion_layers
        )

        # Classifier (shared across all clients)
        self.classifier = nn.Linear(embed_dim, num_labels)

        # Define which parameter name prefixes are shared vs local
        self._shared_prefixes = ("image_encoder.projector.", "classifier.")

    def forward(self, image, input_ids=None, attention_mask=None, **kwargs):
        img_embed = self.image_encoder(image)  # (B, embed_dim)

        if input_ids is not None and attention_mask is not None:
            txt_embed = self.text_encoder(input_ids, attention_mask)  # (B, embed_dim)

            img_token = img_embed.unsqueeze(1) + self.image_type_embed
            txt_token = txt_embed.unsqueeze(1) + self.text_type_embed
            multimodal_seq = torch.cat([img_token, txt_token], dim=1)

            fused = self.fusion_transformer(multimodal_seq)  # (B, 2, embed_dim)
            embed = fused.mean(dim=1)  # (B, embed_dim)
        else:
            # Fallback to image-only if no text provided
            embed = img_embed

        logits = self.classifier(embed)
        return logits, embed

    def get_shared_state_dict(self):
        """Return state_dict of shared parameters (image projector + classifier)."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad and name.startswith(self._shared_prefixes)
        }

    def load_shared_state_dict(self, state_dict):
        """Load shared parameters from aggregated state_dict."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)


class UnifiedMultimodalClassifier(nn.Module):
    """
    Unified model used by ALL clients (both unimodal and multimodal).

    Architecture:
      Image → Frozen ResNet-50 → img_projector → img_embed (256-d)
      Text  → Frozen BiomedBERT → txt_projector → txt_embed (256-d)
             (or zeros if text unavailable)
      [img_embed + txt_embed] → FusionTransformer → fused → Classifier → labels

    Key design:
      - ALL clients use the SAME architecture and parameter space.
      - Unimodal clients pass zeros for the text embedding.
      - ALL trainable parameters are shared and aggregated via FedAvg.
      - This ensures the classifier always receives fused embeddings
        from the same fusion pathway, regardless of client modality.

    Args:
        num_labels:       Number of disease labels
        embed_dim:        Shared embedding dimension
        image_backbone:   "resnet50" or "dinov2"
        text_model_name:  HuggingFace model for text encoder
        num_heads:        Attention heads in fusion transformer
        num_fusion_layers: Number of fusion transformer layers
    """

    def __init__(
        self,
        num_labels=8,
        embed_dim=256,
        image_backbone="resnet50",
        text_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        num_heads=8,
        num_fusion_layers=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Encoders (backbones frozen, projectors trainable)
        self.image_encoder = ImageEncoder(
            backbone=image_backbone, embed_dim=embed_dim, freeze=True
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name, embed_dim=embed_dim, freeze=True
        )

        # Modality type embeddings (trainable, shared)
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.text_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Fusion transformer (trainable, shared)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_fusion_layers
        )

        # Classifier (trainable, shared)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, image, input_ids=None, attention_mask=None, **kwargs):
        """
        Args:
            image:          (B, 3, 224, 224)
            input_ids:      (B, seq_len) or None for unimodal clients
            attention_mask: (B, seq_len) or None for unimodal clients
        Returns:
            logits: (B, num_labels)
            embed:  (B, embed_dim) fused embedding
        """
        # Image path (always runs)
        img_embed = self.image_encoder(image)  # (B, embed_dim)

        # Text path: real features or zeros
        if input_ids is not None and attention_mask is not None:
            txt_embed = self.text_encoder(input_ids, attention_mask)  # (B, embed_dim)
        else:
            # Zero-padding for missing text modality
            txt_embed = torch.zeros_like(img_embed)  # (B, embed_dim)

        # Add type embeddings and stack as 2-token sequence
        img_token = img_embed.unsqueeze(1) + self.image_type_embed  # (B, 1, embed_dim)
        txt_token = txt_embed.unsqueeze(1) + self.text_type_embed   # (B, 1, embed_dim)
        multimodal_seq = torch.cat([img_token, txt_token], dim=1)   # (B, 2, embed_dim)

        # Fusion (always runs, same pathway for all clients)
        fused = self.fusion_transformer(multimodal_seq)  # (B, 2, embed_dim)
        embed = fused.mean(dim=1)  # (B, embed_dim)

        # Classification
        logits = self.classifier(embed)  # (B, num_labels)
        return logits, embed

    def get_shared_state_dict(self):
        """Return ALL trainable parameters (all are shared in unified model)."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_shared_state_dict(self, state_dict):
        """Load shared parameters from aggregated state_dict."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)


if __name__ == "__main__":
    # Quick shape verification
    print("=== UnifiedMultimodalClassifier ===")
    model = UnifiedMultimodalClassifier(num_labels=8, embed_dim=256)
    dummy_image = torch.randn(4, 3, 224, 224)
    dummy_ids = torch.randint(0, 1000, (4, 128))
    dummy_mask = torch.ones(4, 128, dtype=torch.long)

    # Test with text (multimodal client)
    logits, embed = model(dummy_image, dummy_ids, dummy_mask)
    print(f"[Multimodal] Logits: {logits.shape}, Embed: {embed.shape}")

    # Test without text (unimodal client — zero padding)
    logits, embed = model(dummy_image)
    print(f"[Unimodal]   Logits: {logits.shape}, Embed: {embed.shape}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    shared = model.get_shared_state_dict()
    print(f"Shared param keys ({len(shared)}): {list(shared.keys())}")
