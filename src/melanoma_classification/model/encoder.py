import torch
import torch.nn as nn

from melanoma_classification.model.multi_head_self_attention import (
    MultiHeadSelfAttention,
)

from typing import Tuple, Optional

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer

    The Transformer Encoder Layer consists of a Multi-Head Self-Attention (MHSA)
    module followed by a Multi-Layer Perceptron (MLP) module. The encoder in a
    Vision Transformer is responsible for processing the input image and
    extracting relevant features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        norm: str,
    ):
        """Constructor

        Args:
            embed_dim: The size of the input embedding (E).
            num_heads: The number of attention heads (A).
            mlp_ratio: The ratio of the hidden dimension of the MLP to the input
                embedding.
            dropout: The dropout probability.
            norm: The way normalization is applied. Either "pre" or "post".
        """
        super(TransformerEncoderLayer, self).__init__()

        norm = norm.lower()
        if norm == "pre":
            self._norm = self._pre_norm
        elif norm == "post":
            self._norm = self._post_norm
        else:
            raise ValueError(f"Unknown normalization technique '{norm}'.")

        self._mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        self._norm1 = nn.LayerNorm(embed_dim)
        self._norm2 = nn.LayerNorm(embed_dim)
        self._mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def _pre_norm(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x2, attention = self._mhsa(self._norm1(x))  # (B, P, E), (B, A, P, P)
        x = x + x2  # (B, P, E)
        x = x + self._mlp(self._norm2(x))  # (B, P, E)

        if self.training:
            return x, None
        return x, attention
    
    def _post_norm(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x2, attention = self._mhsa(x)  # (B, P, E), (B, A, P, P)
        x = x + x2  # (B, P, E)
        x = x + self._mlp(self._norm1(x))  # (B, P, E)
        x = self._norm2(x)  # (B, P, E)

        if self.training:
            return x, None
        return x, attention

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        P - Number of Patches

        Args:
            x: Input tensor (B, P, E).

        Returns:
            [0]: Output tensor (B, P, E).
            [1]: Attention (B, A, P, P).
        """
        return self._norm(x)


if __name__ == "__main__":
    from melanoma_classification.utils import get_device
    from torchinfo import summary

    device = get_device()

    # Test Transformer Encoder Layer
    encoder = TransformerEncoderLayer(
        embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.5, norm="post"
    ).to(device)
    summary(encoder, input_size=(16, 196, 768), device=device)
    encoder.train()
    input_tensor = torch.randn(16, 196, 768).to(device)
    output, _ = encoder(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
