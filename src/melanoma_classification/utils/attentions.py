import torch
import math
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


def _compute_image_attentions(image: Image.Image, attentions: torch.Tensor):
    """Helper function for Visualization Preprocessing.

    S - Patch Size

    Args:
        image: The image to generate the attention map for.
        attentions: Tensor containing the attention values with shape (S, S).

    Returns:
        Processed attentions.
    """
    num_patches = int(math.sqrt(attentions.shape[0]))
    attentions = attentions.mean(0)
    attentions = attentions.reshape((num_patches, num_patches))
    attentions = attentions / attentions.max()
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0).unsqueeze(0),
        size=(image.size[1], image.size[0]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    return attentions.cpu().numpy()


def visualize_single_attention(
    attention_maps: list[torch.Tensor],
    image: Image.Image,
    save_path: Path | None = None,
) -> None:
    """Visualize the attention of a single head in the last layer of the model.

    Args:
        attention_maps: The attention maps of all heads of all layers.
        image: The image to generate the attention map for.
        save_path: The path to save the visualization.
    """
    attentions = torch.cat(attention_maps)[:, :, 1:, 1:]
    attentions = attentions.mean(dim=(0, 1))

    att = _compute_image_attentions(image=image, attentions=attentions)

    plt.imshow(image)
    plt.imshow(att, cmap="jet", alpha=0.5)
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def visualize_multihead_as_single_attention(
    attention_maps: list[torch.Tensor],
    image: Image.Image,
    layer: int = -1,
    save_path: Path | None = None,
) -> None:
    """Visualize the attention of all heads in a single layer of the model as a
    single, aggregated attention map.

    Args:
        attention_maps: The attention maps of all heads of all layers.
        image: The image to generate the attention map for.
        layer: The layer to visualize.
        save_path: The path to save the visualization.
    """
    attentions = torch.cat(attention_maps)[layer, :, 1:, 1:]
    attentions = attentions.mean(dim=0)

    att = _compute_image_attentions(image=image, attentions=attentions)

    plt.imshow(image)
    plt.imshow(att, cmap="jet", alpha=0.5)
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def visualize_multihead_attention(
    attention_maps: list[torch.Tensor],
    image: Image.Image,
    layer: int = -1,
    save_path: Path | None = None,
) -> None:
    """Visualize the attention of all heads in a single layer of the model.

    Args:
        attention_maps: The attention maps of all heads of all layers.
        image: The image to generate the attention map for.
        layer: The layer to visualize.
        save_path: The path to save the visualization.
    """
    attentions = torch.cat(attention_maps)[layer, :, 1:, 1:]

    n_imgs = attentions.size(0)
    cols = math.ceil(math.sqrt(n_imgs))
    rows = math.ceil(n_imgs / cols)
    _, axes = plt.subplots(rows, cols, figsize=(10, 7))

    for head, ax in enumerate(axes.flat):
        att = _compute_image_attentions(
            image=image, attentions=attentions[head]
        )
        ax.imshow(image)
        ax.imshow(att, cmap="jet", alpha=0.5)
        ax.axis("off")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
