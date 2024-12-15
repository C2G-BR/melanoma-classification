from importlib.resources import files
from logging import getLogger

import mlflow
import numpy as np
import torch
from PIL import Image

from melanoma_classification.inference.attention_viz import (
    visualize_multihead_as_single_attention,
    visualize_multihead_attention,
    visualize_single_attention,
)
from melanoma_classification.model import get_dermmel_classifier_v1
from melanoma_classification.paths import MODEL_STATE_DICT, STATE_FILES
from melanoma_classification.utils import (
    get_device,
    load_state_dict,
    production_transform,
)

logger = getLogger(__name__)


def inference(
    run: mlflow.ActiveRun | None, img_path: str, epoch: int | None
) -> None:
    device = get_device()
    model = get_dermmel_classifier_v1()
    if epoch is None:
        model_path = files("melanoma_classification").joinpath(
            "weights/vit.pth"
        )
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        load_state_dict(
            run=run,
            container=model,
            artifact_path=STATE_FILES.format(epoch=epoch)
            + "/"
            + MODEL_STATE_DICT,
        )
    model.to(device)
    model.eval()

    raw_image = Image.open(img_path).convert("RGB")
    image = (
        production_transform()(image=np.array(raw_image))["image"]
        .to(device)
        .unsqueeze(0)
    )

    with torch.no_grad():
        model_outputs = model(image)
        logits = model_outputs["outputs"]
        attention = model_outputs["attentions"]
        logits = torch.nn.functional.softmax(logits, 1)
        confidence, prediction = torch.max(logits, dim=1)
        confidence, prediction = confidence.item(), prediction.item()

    detected = model.class_map[prediction]
    logger.info(
        "Found a %s sample with confidence %.2f%%.", detected, confidence * 100
    )
    visualize_single_attention(image=raw_image, attention_maps=attention)
    visualize_multihead_as_single_attention(
        image=raw_image, attention_maps=attention
    )
    visualize_multihead_attention(image=raw_image, attention_maps=attention)
