import numpy as np
import typer
import torch
from importlib.resources import files

from pathlib import Path
from PIL import Image
from melanoma_classification.utils import (
    get_device,
    production_transform,
    train_transform,
    visualize_single_attention,
    visualize_multihead_as_single_attention,
    visualize_multihead_attention,
    DermMel,
    train as _train,
)
from melanoma_classification.model import get_dermmel_classifier_v1


app = typer.Typer(
    name="Melanoma Classification", pretty_exceptions_enable=False
)


@app.command()
def train(
    data_path: str,
    checkpoint_path: str,
    checkpoint_model_file: str | None = None,
):
    device = get_device()
    vit = get_dermmel_classifier_v1()
    vit.load_pretrained_weights("deit_base_patch16_224")
    train_dataset = DermMel(
        data_path, split="train_sep", transform=train_transform()
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2
    )

    valid_dataset = DermMel(
        data_path, split="valid", transform=production_transform()
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, shuffle=True, num_workers=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": vit.cls_token, "lr": 1e-7},
            {"params": vit.pos_embed, "lr": 1e-7},
            {"params": vit.patch_embedding.parameters(), "lr": 1e-6},
            {"params": vit.transformer_layers.parameters(), "lr": 1e-5},
            {"params": vit.norm.parameters(), "lr": 1e-6},
            {"params": vit.classifier.parameters(), "lr": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )
    _train(
        model=vit,
        train_loader=train_dataloader,
        val_loader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        freezed_epochs=5,
        device=device,
        checkpoint_model_file=checkpoint_model_file,
        checkpoint_path=Path(checkpoint_path),
        save_every_n_epochs=1,
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
