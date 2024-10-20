import torch.nn as nn
from melanoma_classification.model.vision_transformer import VisionTransformer


def get_dermmel_classifier_v1() -> nn.Module:
    vit = VisionTransformer()
    classifier = nn.Sequential(
        nn.Linear(vit.get_embedding_dimension(), 64), nn.ReLU(), nn.Linear(64, 2)
    )
    vit.classifier = classifier
    vit.classes = {0: "Melanoma", 1: "Benign"}
    return vit
