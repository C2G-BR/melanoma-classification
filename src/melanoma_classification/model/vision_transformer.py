import torch
import torch.nn as nn

from melanoma_classification.model.encoder import TransformerEncoderLayer
from melanoma_classification.model.patch_embedding import (
    PatchEmbeddingCNN,
    PatchEmbeddingLinear,
)

from collections import OrderedDict


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) Model for Image Classification

    This class implements the Vision Transformer architecture as introduced in
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    by Dosovitskiy et al. It utilizes patch embeddings, Multi-Head Self-
    Attention (MHSA), and positional encodings to process image data.

    Attributes:
        class_map: A mapping from indices to class names.
    """

    def __init__(
        self,
        classifier: nn.Module,
        class_map: dict[int, str],
        embedding: str,
        in_channels: int,
        patch_size: int,
        img_size: int,
        dropout: float,
        mlp_ratio: float,
        num_heads: int,
        norm: str,
        depth: int,
    ):
        """Constructor

        N - Number of Classes

        Args:
            classifier: Classifier to use. Must take in a tensor of dimension
                (B, E) and output a tensor of dimension (B, N).
            class_map: A mapping from indices to class names.
            embedding: Embedding to use. Either "cnn" or "linear".
            in_channels: Number of input channels (C), e.g., 3 for RGB images.
            patch_size: Size of a single square patch (S).
            img_size: Size of the input image. The image is expected to be of
                dimensions (C, H, W). H == W is assumed.
            dropout: The dropout probability.
            mlp_ratio: The ratio of the hidden dimension of the MLP to the input
                embedding.
            num_heads: Number of attention heads in each transformer layer (A).
            norm: The way normalization in the encoder is applied. Either "pre"
                or "post".
            depth: Number of layers in the transformer encoder.
        """
        super(VisionTransformer, self).__init__()

        self._classifier = classifier
        self.class_map = class_map

        embedding = embedding.lower()
        if embedding == "cnn":
            self._patch_embedding = PatchEmbeddingCNN(
                in_channels=in_channels, patch_size=patch_size
            )
        elif embedding == "linear":
            self._patch_embedding = PatchEmbeddingLinear(
                in_channels=in_channels, patch_size=patch_size
            )
        else:
            raise ValueError(f"Unknown embedding technique '{embedding}'.")

        embed_dim = in_channels * patch_size * patch_size
        num_patches = (img_size // patch_size) ** 2

        self._cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self._pos_embed = nn.Parameter(
            torch.randn(1, 1 + num_patches, embed_dim)
        )
        self._dropout = nn.Dropout(dropout)
        self._transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(depth)
            ]
        )
        self._norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass

        P - Number of Patches

        Args:
            x: Input images as a tensor (B, C, H, W).

        Returns:
            [0]: Output tensor (B, N).
            [1]: Attention list (B, A, P+1, P+1).
        """
        B = x.size(0)

        x = self._patch_embedding(x)  # (B, P, E)

        cls_tokens = self._cls_token.expand(B, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, P+1, E)

        x = x + self._pos_embed  # (B, P+1, E)
        x = self._dropout(x)  # (B, P+1, E)

        attention_maps = []
        for layer in self._transformer_layers:
            x, attention = layer(x)  # (B, P+1, E), (B, A, P+1, P+1)
            attention_maps.append(attention)

        cls_token_output = self._norm(x[:, 0])  # (B, E)

        return self._classifier(cls_token_output), attention_maps

    def load_pretrained_weights(
        self,
        model_name: str = "deit_base_patch16_224",
        load_classifier: bool = False,
    ) -> None:
        """Load pre-trained weights from PyTorch's available models, optionally
        loading the classifier

        Args:
            model_name: The name of the pre-trained model to load.
            load_classifier: Whether to load the classifier weights.
        """
        pretrained_model = torch.hub.load(
            "facebookresearch/deit:main", model_name, pretrained=True
        )
        state_dict = pretrained_model.state_dict()

        # Remove keys related to classifier if not loading classifier.
        if not load_classifier:
            state_dict = OrderedDict(
                {k: v for k, v in state_dict.items() if "head" not in k}
            )

        self.load_state_dict(state_dict, strict=False)

    def freeze_backbone(self) -> None:
        """Freeze the backbone of the model.

        This will prevent any gradients from being calculated for the backbone,
        effectively freezing it during training.
        """
        for param in self._patch_embedding.parameters():
            param.requires_grad = False

        for layer in self._transformer_layers:
            for param in layer.parameters():
                param.requires_grad = False

        self._cls_token.requires_grad = False
        self._pos_embed.requires_grad = False
        for param in self._norm.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze the backbone of the model.

        This will allow gradients to be calculated for the backbone, making it
        trainable again.
        """
        for param in self._patch_embedding.parameters():
            param.requires_grad = True

        for layer in self._transformer_layers:
            for param in layer.parameters():
                param.requires_grad = True

        self._cls_token.requires_grad = True
        self._pos_embed.requires_grad = True
        for param in self._norm.parameters():
            param.requires_grad = True

    def unfreeze_sequentially(self) -> None:
        """Unfreeze the backbone of the model sequentially.

        Each call to this method will unfreeze the next layer if one is frozen.
        """
        unfroze_layer = False

        for param in self._norm.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfroze_layer = True

        if unfroze_layer:
            print("Unfroze norm.")
            return

        for layer in reversed(self._transformer_layers):
            for param in layer.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfroze_layer = True

            if unfroze_layer:
                print("Unfroze layer of transformers.")
                return

        if not self._pos_embed.requires_grad:
            self._pos_embed.requires_grad = True
            print("Unfroze positional embedding.")
            return

        if not self._cls_token.requires_grad:
            self._cls_token.requires_grad = True
            print("Unfroze cls_token.")
            return

        for param in self._patch_embedding.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfroze_layer = True

        if unfroze_layer:
            print("Unfroze patch_embedding.")


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test ViT
    embed_dim = 768
    vit = VisionTransformer(
        classifier=nn.Linear(in_features=768, out_features=2),
        class_map={0: "true_class", 1: "false_class"},
        embedding="cnn",
        in_channels=3,
        patch_size=16,
        img_size=224,
        dropout=0.1,
        mlp_ratio=4.0,
        num_heads=12,
        norm="post",
        depth=12,
    )
    summary(vit, input_size=(16, 3, 224, 224))
    vit.train()
    input_tensor = torch.randn(16, 3, 224, 224).to(device)
    output = vit(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
