import torch
import torch.nn as nn

from typing import Protocol


class PatchEmbeddingProtocol(nn.Module, Protocol):
    """Patch Embedding"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            A sequence of flattened 2D patches (B, P, E).
        """
        ...


class PatchEmbeddingCNN(nn.Module):
    """Patch Embedding with Convolutional Projection"""

    def __init__(self, in_channels: int, patch_size: int):
        """Constructor.

        The image is expected to be of dimensions (C, H, W). H == W is assumed.

        Args:
            in_channels: Number of input channels (C).
            patch_size: Size of a single patch (P). A patch will have the
                dimensions (C, P, P).
        """
        super(PatchEmbeddingCNN, self).__init__()

        embed_dim = in_channels * patch_size * patch_size

        # The projection layer projects the input image to a sequence of
        # flattened 2D patches
        self._projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        S - Patch Size
        G - Grid Size: (H or W) // S
        P - Number of Patches: G**2
        E - Embedding Dimension: C * S**2

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            A sequence of flattened 2D patches (B, P, E).
        """

        x = self._projection(x)  # (B, E, G, G)
        x = x.flatten(2)  # (B, E, G**2)
        x = x.transpose(1, 2)  # (B, G**2, E)

        return x


class PatchEmbeddingLinear(nn.Module):
    """Patch Embedding with Linear Projection"""

    def __init__(self, in_channels: int, patch_size: int):
        """Constructor

        The image is expected to be of dimensions (C, H, W). H == W is assumed.

        Args:
            in_channels: Number of input channels (C).
            patch_size: Size of a single patch (P). A patch will have the
                dimensions (C, P, P).
        """
        super(PatchEmbeddingLinear, self).__init__()

        self._patch_size = patch_size

        embed_dim = in_channels * self._patch_size * self._patch_size

        self._projection = nn.Linear(
            in_features=embed_dim, out_features=embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        S - Patch Size
        G - Grid Size: (H or W) // S
        P - Number of Patches: G**2
        E - Embedding Dimension: C * S**2

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            A sequence of flattened 2D patches (B, P, E).
        """
        B, C, H, _ = x.shape
        grid_size = H // self._patch_size

        x = x.view(
            B, C, grid_size, self._patch_size, grid_size, self._patch_size
        )  # (B, C, G, S, G, S)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, G, G, C, S, S)
        x = x.view(B, -1, self._patch_size * self._patch_size * C)  # (B, P, E)
        x = self._projection(x)  # (B, P, E)

        return x


if __name__ == "__main__":
    from melanoma_classification.utils import get_device
    from torchinfo import summary

    device = get_device()

    # Test CNN Embedding
    cnn = PatchEmbeddingCNN(in_channels=3, patch_size=16).to(device)
    summary(cnn, input_size=(16, 3, 224, 224))
    cnn.train()
    input_tensor = torch.randn(16, 3, 224, 224).to(device)
    output = cnn(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()

    # Test Linear Embedding
    linear = PatchEmbeddingLinear(in_channels=3, patch_size=16)
    summary(linear, input_size=(16, 3, 224, 224))
    linear.train()
    input_tensor = torch.randn(16, 3, 224, 224).to(device)
    output = linear(input_tensor)
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
