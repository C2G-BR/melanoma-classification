import math
import torch
import torch.nn as nn

from typing import Tuple, Optional

class MultiHeadSelfAttention(nn.Module):
    """Multi-head Self-Attention (MHSA)

    The input tensor is split into num_heads, and the attention scores are
    computed for each head. Please see
    https://paperswithcode.com/method/multi-head-attention for more information.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """Constructor

        Args:
            embed_dim: The size of the input embedding (E).
            num_heads: The number of attention heads (A). A is assumed to be
                exactly divisible by E.
        """
        super(MultiHeadSelfAttention, self).__init__()

        self._embed_dim = embed_dim
        self._num_heads = num_heads

        self._head_dim = self._embed_dim // self._num_heads
        self._scale = math.sqrt(self._head_dim)

        self._query = nn.Linear(self._embed_dim, self._embed_dim)
        self._key = nn.Linear(self._embed_dim, self._embed_dim)
        self._value = nn.Linear(self._embed_dim, self._embed_dim)
        self._fc_out = nn.Linear(self._embed_dim, self._embed_dim)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass

        P - Number of Patches
        D - Head Dimension: E // A
        Q - Query: Represents the embedding of the current token/patch that
            attention is calculated for.
        K - Key: Represents all tokens/patches against which attention is
            calculated.
        V - Value: Information we ultimately want to aggregate based on
            attention scores.

        Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V

        Args:
            x: Input tensor (B, P, E).

        Returns:
            [0]: Output tensor (B, P, E).
            [1]: Attention (B, A, P, P) during eval mode.
        """

        B, P, _ = x.size()

        # Output Dimension: (B, P, A, D)
        query = self._query(x).view(B, P, self._num_heads, self._head_dim)
        key = self._key(x).view(B, P, self._num_heads, self._head_dim)
        value = self._value(x).view(B, P, self._num_heads, self._head_dim)

        # Output Dimension: (B, A, P, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / self._scale, dim=-1
        )  # (B, A, P, P)

        # Compute the weighted sum of values
        out = torch.matmul(attention, value)  # (B, A, P, D)
        out = out.transpose(1, 2).reshape(B, P, self._embed_dim)  # (B, P, E)
        out = self._fc_out(out)  # (B, P, E)

        if self.training:
            return out, None
        return out, attention


if __name__ == "__main__":
    from melanoma_classification.utils import get_device
    from torchinfo import summary

    device = get_device()

    # Test MHSA
    mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=8).to(device)
    summary(mhsa, input_size=(16, 196, 768), device=device)
    mhsa.train()
    input_tensor = torch.randn(16, 196, 768).to(device)
    output, _ = mhsa(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()