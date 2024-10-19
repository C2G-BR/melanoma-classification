import torch
import torch.nn as nn


class PatchEmbeddingCNN(nn.Module):
    """Patch Embedding with Convolutional Projection."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,  # TODO: remove if not required anymore
    ):
        """Constructor

        The image is expected to be of dimensions (in_channels, img_size,
        img_size).

        Args:
            img_size: Size of the input image.
            in_channels: Number of input channels.
            patch_size: Size of a single patch. A patch will have the dimensions
                (in_channels, patch_size, patch_size).
            embed_dim: Size of the output embedding vector. Generally, the rule
                is in_channels * patch_size * patch_size.
        """

        super(PatchEmbeddingCNN, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # TODO: What happens if this is not exactly divisible?
        # Define the projection layer
        self.grid_size = img_size // patch_size  # Size of the grid
        self.num_patches = self.grid_size**2  # Number of patches

        # TODO: Enable use of linear encoding.
        # The projection layer projects the input image
        # to a sequence of flattened 2D patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor. (B, in_channels, img_size, img_size)

        Returns:
            A sequence of flattened 2D patches. (B, num_patches, embed_dim)
        """

        x = self.projection(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)

        return x


class PatchEmbeddingLinear(nn.Module):
    """Patch Embedding with Linear Projection."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """Constructor

        The image is expected to be of dimensions (in_channels, img_size,
        img_size).

        Args:
            img_size: Size of the input image.
            in_channels: Number of input channels.
            patch_size: Size of a single patch. A patch will have the dimensions
                (in_channels, patch_size, patch_size).
            embed_dim: Size of the output embedding vector. Generally, the rule
                is in_channels * patch_size * patch_size.
        """
        super(PatchEmbeddingLinear, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Linear(
            patch_size * patch_size * in_channels,
            embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor. (B, in_channels, img_size, img_size)

        Returns:
            A sequence of flattened 2D patches. (B, num_patches, embed_dim)
        """
        # TODO: Understand
        batch_size, channels, height, width = x.shape

        # Step 1: Reshape the input into patches
        # Rearrange the input into patches of size (batch_size, num_patches, patch_size * patch_size * in_channels)
        x = x.view(
            batch_size,
            channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        x = x.permute(
            0, 2, 4, 1, 3, 5
        ).contiguous()  # Bring channels to the right place
        x = x.view(
            batch_size, -1, self.patch_size * self.patch_size * channels
        )  # Flatten patches

        # Step 2: Linear projection of each patch
        x = self.projection(x)  # Shape: [batch_size, num_patches, embed_dim]

        return x


if __name__ == "__main__":
    from torchinfo import summary

    cnn = PatchEmbeddingCNN()
    print(summary(cnn, input_size=(16, 3, 224, 224)))
    linear = PatchEmbeddingLinear()
    print(summary(linear, input_size=(16, 3, 224, 224)))