import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head Self-Attention (MHSA)"""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        """Constructor

        The input tensor is split into num_heads, and the attention scores are
        computed for each head.

        Args:
            embed_dim: The size of the input embedding
            num_heads: The number of attention heads. For more information, see
                https://paperswithcode.com/method/multi-head-attention.
        """
        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Check if embed_dim is divisible by num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # TODO: can we optimize q,k,v computation by using a single NN
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        - Query (Q) represents the current token/patch's embedding that we
        want to calculate attention for
        - Key (K) represents all tokens/patches against which attention is
        calculated
        - Value (V) is the information we ultimately want to aggregate based on
        attention scores

        Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V

        Args:
            x: Input tensor (B, Number Patches, embed_dim)

        Returns:
            Output tensor (B, Number Patches, embed_dim)
        """

        B, N, _ = x.size()

        # Split the embed_dim into num_heads
        # TODO: Maybe rather reshape

        query = self.query(x).view(B, N, self.num_heads, self.head_dim)
        key = self.key(x).view(B, N, self.num_heads, self.head_dim)
        value = self.value(x).view(B, N, self.num_heads, self.head_dim)

        # Transpose to get dimensions (B, num_heads, N, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)  # TODO: Why do we transpose here? B, num_heads, N, head_dim
        value = value.transpose(1, 2)

        # Compute the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)

        # Compute the weighted sum of values
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)

        # Apply the final linear layer
        out = self.fc_out(out)

        return out, attention


if __name__ == "__main__":
    from torchinfo import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mhsa = MultiHeadSelfAttention().to(device)
    summary(mhsa, input_size=(16, 196, 768))

    # Test forwad & backward pass for mhsa
    mhsa.train()
    input_tensor = torch.randn(16, 196, 768).to(device)
    output = mhsa(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()