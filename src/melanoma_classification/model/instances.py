import torch.nn as nn
from melanoma_classification.model.vision_transformer import VisionTransformer


def get_dermmel_classifier_v1() -> nn.Module:
    patch_size = 16
    in_channels = 3

    classifier = nn.Sequential(
        nn.Linear(patch_size**2 * in_channels, 64), nn.ReLU(), nn.Linear(64, 2)
    )
    vit = VisionTransformer(
        classifier=classifier,
        class_map={0: "Melanoma", 1: "Benign"},
        embedding="cnn",
        in_channels=in_channels,
        patch_size=patch_size,
        img_size=224,
        dropout=0.1,
        mlp_ratio=4.0,
        num_heads=12,
        norm="post",
        depth=12,
    )

    return vit
