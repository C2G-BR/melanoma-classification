import typer
import torch
from pathlib import Path
from PIL import Image
from melanoma_classification.utils import get_device, production_transform
from melanoma_classification.model import get_dermmel_classifier_v1
app = typer.Typer(name="Melanoma Classification")


@app.command()
def predict(model_path: Path, img_path: Path):
    device = get_device()
    image = Image.open(img_path).convert('RGB')
    image = production_transform()(image).to(device).unsqueeze(0)
    checkpoint = torch.load(model_path, weights_only=True)
    model = get_dermmel_classifier_v1()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=get_device())
    model.eval()
    with torch.no_grad():
        logits = model(image)
        logits = torch.nn.functional.softmax(logits, 1)
        confidence, prediction = torch.max(logits, dim=1)
        confidence, prediction = confidence.item(), prediction.item()
    
    detected = model.classes[prediction]
    print(f"Found a {detected} sample with confidence {confidence*100:.2f}%.")


if __name__ == "__main__":
    app()
