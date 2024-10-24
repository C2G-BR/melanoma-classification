import matplotlib.pyplot as plt
import numpy as np
import typer
import torch

from pathlib import Path
from PIL import Image
from melanoma_classification.utils import get_device, production_transform
from melanoma_classification.model import get_dermmel_classifier_v1


app = typer.Typer(name="Melanoma Classification", pretty_exceptions_enable=False)


def visualize_single_attention(image, attention_map):
    """
    Visualizes the attention map on top of the input image.

    Args:
        image: Input image (H, W, 3) or (C, H, W) in numpy format
        attention_map: Attention weights in shape (num_layers, num_heads, num_patches**2, num_patches**2)
    """
    if image.dim() == 4:
        image = image.squeeze(0)

    attentions = torch.cat(attention_map)[:, :, 1:, 1:]
    print(attentions.shape)
    attentions = attentions.mean(dim=(0, 1))
    # attentions = attentions.reshape(())
    num_patches = int(np.sqrt(attentions.shape[0]))
    attentions = attentions.mean(0)
    attentions = attentions.reshape((num_patches, num_patches))
    attentions = attentions / attentions.max()

    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0).unsqueeze(0),
        size=(image.shape[1], image.shape[2]),  # Upsample to match image size
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    plt.imshow(
        image.permute(1, 2, 0).cpu().numpy()
    )  # Assuming image is a torch tensor (C, H, W)
    plt.imshow(
        attentions.cpu().numpy(), cmap="jet", alpha=0.5
    )  # Overlay the attention map
    plt.axis("off")
    plt.savefig("single_attention.png")
    plt.show()


def visualize_multihead_as_single_attention(image, attention_map, layer=-1):
    """
    Visualizes attention of a specified Multi-Head in a single map on top of the
    input image.

    Args:
        image: Input image (H, W, 3) or (C, H, W) in numpy format
        attention_map: Attention weights in shape (encoder_layers, num_heads, num_patches, num_patches)
    """
    if image.dim() == 4:
        image = image.squeeze(0)

    attentions = torch.cat(attention_map)[layer, :, 1:, 1:]
    attentions = attentions.mean(dim=0)

    num_patches = int(np.sqrt(attentions.shape[0]))
    attentions = attentions.mean(0)
    attentions = attentions.reshape((num_patches, num_patches))
    attentions = attentions / attentions.max()

    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0).unsqueeze(0),
        size=(image.shape[1], image.shape[2]),  # Upsample to match image size
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    plt.imshow(
        image.permute(1, 2, 0).cpu().numpy()
    )  # Assuming image is a torch tensor (C, H, W)
    plt.imshow(
        attentions.cpu().numpy(), cmap="jet", alpha=0.5
    )  # Overlay the attention map
    plt.axis("off")
    plt.savefig(f"single_multihead_attention_{layer}.png")
    plt.show()


def visualize_multihead_attention(raw_image, attention_map, layer=-1):
    """
    Visualizes attention of a specified Multi-Hea for each of its heads on top
    of the input image.

    Args:
        raw_image: Input image (H, W, 3)
        attention_map: Attention weights in shape (encoder_layers, num_heads, num_patches, num_patches)
    """
    # print(raw_image.size)
    attentions = torch.cat(attention_map)[layer, :, 1:, 1:]

    # TODO init figure with subplots (square row/col layout)
    _, axes = plt.subplots(3, 4, figsize=(10, 7))

    for head, ax in enumerate(axes.flat):
        attention = attentions[head]

        num_patches = int(np.sqrt(attention.shape[0]))
        attention = attention.mean(0)
        attention = attention.reshape((num_patches, num_patches))
        attention = attention / attention.max()

        attention = torch.nn.functional.interpolate(
            attention.unsqueeze(0).unsqueeze(0),
            # Weird dimensions because of PIL
            size=(raw_image.size[1], raw_image.size[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        print(raw_image.size, attention.shape)
        ax.imshow(raw_image)
        ax.imshow(attention.cpu().numpy(), cmap="jet", alpha=0.5)
        ax.axis("off")
    plt.savefig(f"multihead_attention_{layer}.png")
    plt.show()


@app.command()
def predict(model_path: Path, img_path: Path):
    device = get_device()
    raw_image = Image.open(img_path).convert("RGB")
    image = production_transform()(raw_image).to(device).unsqueeze(0)
    checkpoint = torch.load(model_path, weights_only=True)
    model = get_dermmel_classifier_v1()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=get_device())
    model.eval()
    with torch.no_grad():
        model_outputs = model(image)
        logits = model_outputs["outputs"]
        attention = model_outputs["attentions"]
        logits = torch.nn.functional.softmax(logits, 1)
        confidence, prediction = torch.max(logits, dim=1)
        confidence, prediction = confidence.item(), prediction.item()

    detected = model.classes[prediction]
    print(f"Found a {detected} sample with confidence {confidence*100:.2f}%.")
    visualize_single_attention(image, attention)
    visualize_multihead_as_single_attention(image, attention)
    visualize_multihead_attention(raw_image, attention)


if __name__ == "__main__":
    app()
