import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
from typing import Callable
from PIL.Image import Image
from torch import Tensor

# Define ImageNet normalization parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def train_transform_v1() -> Callable[[Image], Tensor]:
    """Returns a transformation pipeline for training images using torchvision.

    Legacy function.
    Can be removed if training works flawlessly with albumentations.

    Returns:
        A function that takes an image and returns a transformed tensor.
    """

    return v2.Compose(
        [
            v2.Resize((224, 224)),  # Resize to 224x224
            v2.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            v2.RandomRotation(15),  # Random rotation between -15 and 15 degrees
            v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Apply color jitter
            v2.ToTensor(),  # Convert image to PyTorch tensor
            v2.Normalize(
                mean=imagenet_mean, std=imagenet_std
            ),  # Normalize based on ImageNet
        ]
    )


def production_transform_v1() -> Callable[[Image], Tensor]:
    """Returns a transformation pipeline for production images using
    torchvision.

    Legacy function.
    Can be removed if training works flawlessly with albumentations.

    Returns:
        A function that takes an image and returns a transformed tensor.
    """
    return v2.Compose(
        [
            v2.Resize((224, 224)),  # Resize to 224x224
            v2.ToTensor(),  # Convert image to PyTorch tensor
            v2.Normalize(
                mean=imagenet_mean, std=imagenet_std
            ),  # Normalize based on ImageNet
        ]
    )


def train_transform(prob=0.5) -> Callable[[Image], Tensor]:
    """Returns a transformation pipeline for training images using
    albumentations.

    The pipeline includes:
    - Resizing to 224x224 pixels
    - Random horizontal flip with a given probability
    - Random rotation within 15 degrees with a given probability
    - Random grid distortion with a given probability
    - Random color jittering for brightness, contrast, saturation, and hue with
      a given probability
    - Normalization using ImageNet mean and standard deviation
    - Conversion to tensor

    Args:
        prob: Probability for applying certain transformations.

    Returns:
        A function that takes an image and returns a transformed tensor.
    """
    return A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=prob),
            A.Rotate(limit=15, p=prob),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=prob
            ),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )


def production_transform() -> Callable[[Image], Tensor]:
    """Returns a transformation pipeline for production images using
    albumentations.

    The pipeline includes:
    - Resizing to 224x224 pixels
    - Normalization using ImageNet mean and standard deviation
    - Conversion to tensor

    Returns:
        A function that takes an image and returns a transformed tensor.
    """
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
