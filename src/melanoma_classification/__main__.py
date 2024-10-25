import numpy as np
import typer
import torch
from importlib.resources import files

from pathlib import Path
from PIL import Image
from melanoma_classification.utils import (
    get_device,
    production_transform,
    visualize_single_attention,
    visualize_multihead_as_single_attention,
    visualize_multihead_attention,
)
from melanoma_classification.model import get_dermmel_classifier_v1


app = typer.Typer(
    name="Melanoma Classification", pretty_exceptions_enable=False
)


@app.command()
def predict(img_path: Path, model_path: Path | None = None):
    if model_path is None:
        model_path = files("melanoma_classification").joinpath(
            "weights/vit.pth"
        )
    device = get_device()
    raw_image = Image.open(img_path).convert("RGB")
    image = (
        production_transform()(image=np.array(raw_image))["image"]
        .to(device)
        .unsqueeze(0)
    )
    checkpoint = torch.load(model_path, weights_only=True)
    model = get_dermmel_classifier_v1()
    model.load_state_dict(checkpoint)
    model.to(device=get_device())
    model.eval()
    with torch.no_grad():
        model_outputs = model(image)
        logits = model_outputs["outputs"]
        attention = model_outputs["attentions"]
        logits = torch.nn.functional.softmax(logits, 1)
        confidence, prediction = torch.max(logits, dim=1)
        confidence, prediction = confidence.item(), prediction.item()

    detected = model.class_map[prediction]
    print(f"Found a {detected} sample with confidence {confidence*100:.2f}%.")
    visualize_single_attention(raw_image, attention)
    visualize_multihead_as_single_attention(raw_image, attention)
    visualize_multihead_attention(raw_image, attention)


if __name__ == "__main__":
    app()
