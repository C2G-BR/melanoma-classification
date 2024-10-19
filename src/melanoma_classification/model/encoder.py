import torch
import torch.nn as nn

from melanoma_classification.model.multi_head_self_attention import (
    MultiHeadSelfAttention,
)

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer

    The Transformer Encoder Layer consists of a Multi-Head Self-Attention (MHSA)
    module followed by a Multi-Layer Perceptron (MLP) module. The encoder in a
    Vision Transformer is responsible for processing the input image and
    extracting relevant features.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Constructor

        Args:
            embed_dim: The size of the input embedding.
            num_heads: The number of attention heads.
            mlp_ratio: The ratio of the hidden dimension of the MLP to the input
                embedding.
            dropout: The dropout probability.
        """
        super(TransformerEncoderLayer, self).__init__()

        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor. (B, Num Patches, embed_dim)

        Returns:
            Output tensor. (B, Num Patches, embed_dim)
        """

        # Currently, we use post-norm:
        x = x + self.mhsa(x)
        x = self.norm1(x)

        x = x + self.mlp(x)
        x = self.norm2(x)

        # Alternatively, one could use pre-norm
        # x = x + self.mhsa(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))

        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = TransformerEncoderLayer().to(device)
    summary(encoder, input_size=(16, 196, 768))

    # Test forwad & backward pass for transformer encoder layer.
    encoder.train()
    input_tensor = torch.randn(16, 196, 768).to(device)
    output = encoder(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
