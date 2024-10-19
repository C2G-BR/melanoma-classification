import torch
import torch.nn as nn

from melanoma_classification.model.patch_embedding import PatchEmbeddingCNN
from melanoma_classification.model.encoder import TransformerEncoderLayer

from collections import OrderedDict


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_classes: int = 1000,
        dropout: float = 0.1,
    ):
        """Vision Transformer (ViT) model for image classification.

        This class implements the Vision Transformer architecture as introduced
        in "An Image is Worth 16x16 Words: Transformers for Image Recognition at
        Scale" by Dosovitskiy et al. It utilizes patch embeddings, Multi-Head
        Self-Attention (MHSA), and positional encodings to process image data.

        Args:
            img_size: Size of the input image (height and width).
            patch_size: Size of each square patch (height and width).
            in_channels: Number of input channels (e.g., 3 for RGB images).
            embed_dim: Dimensionality of the embedding space.
            depth: Number of layers in the transformer encoder.
            num_heads: Number of attention heads in each transformer layer.
            num_classes: Number of output classes for classification.
            dropout: The dropout probability.
        """
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbeddingCNN(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Random initialize cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.patch_embedding.num_patches, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Define the classification heads (initially set for num_classes)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input image as a tensor. (B, in_channels, height, width)

        Returns:
            Output tensor. (B, num_classes)
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Append [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Normalize the [CLS] token
        cls_output = self.norm(x[:, 0])  # Extract the [CLS] token output

        # Pass through classifier
        return self.classifier(cls_output)

    def set_classifier(self, num_classes: int) -> None:
        """Dynamically replace the classifier

        Args:
            num_classes: The number of classes for the classifier
        """

        self.classifier = nn.Linear(self.classifier.in_features, num_classes)

    def save_model(self, file_path: str) -> None:
        """Save the entire model (core + classifier)

        Args:
            file_path: The file path to save the model
        """

        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the entire model (core + classifier)

        Args:
            file_path: The file path to save the model
        """

        self.load_state_dict(torch.load(file_path))

    def save_backbone(self, file_path) -> None:
        """Save only the transformer backbone

        Args:
            file_path: The file path to save the backbone
        """
        backbone_state_dict = OrderedDict(
            {k: v for k, v in self.state_dict().items() if "classifier" not in k}
        )
        torch.save(backbone_state_dict, file_path)

    def load_backbone(self, file_path: str) -> None:
        """Load only the transformer backbone

        Args:
            file_path: The file path to load the backbone
        """

        state_dict = torch.load(file_path)
        self.load_state_dict(state_dict, strict=False)

    def save_classifier(self, file_path: str) -> None:
        """Save only the final classifier head

        Args:
            file_path: The file path to save the classifier

        """
        torch.save(self.classifier.state_dict(), file_path)

    def load_classifier(self, file_path: str) -> None:
        """Load only the final classifier head

        Args:
            file_path: The file path to load the classifier
        """
        self.classifier.load_state_dict(torch.load(file_path))

    def load_pretrained_weights(
        self, model_name: str = "vit_base_patch16_224", load_classifier: bool = False
    ) -> None:
        """Load pre-trained weights from PyTorch's available models, optionally
        loading the classifier

        TODO: Fix import issues with vit_base_patch16_224
        Model that work: deit_base_patch16_224

        Args:
            model_name: The name of the pre-trained model to load
            load_classifier: Whether to load the classifier weights or not
        """
        # Load the pre-trained model from PyTorch's vision transformer models
        pretrained_model = torch.hub.load(
            "facebookresearch/deit:main", model_name, pretrained=True
        )

        # Load the core transformer weights
        state_dict = pretrained_model.state_dict()

        # Remove keys related to classifier if not loading classifier
        if not load_classifier:
            state_dict = OrderedDict(
                {k: v for k, v in state_dict.items() if "head" not in k}
            )

        # Load state dict into the current model
        self.load_state_dict(state_dict, strict=False)