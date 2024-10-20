from torchvision.transforms import v2
from typing import Callable
from PIL.Image import Image
from torch import Tensor

# Define ImageNet normalization parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Training Transformations
def train_transform() -> Callable[[Image], Tensor]:
    return v2.Compose([
        v2.Resize((224, 224)),              # Resize to 224x224
        v2.RandomHorizontalFlip(),          # Randomly flip the image horizontally
        v2.RandomRotation(15),              # Random rotation between -15 and 15 degrees
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply color jitter
        v2.ToTensor(),                      # Convert image to PyTorch tensor
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize based on ImageNet
    ])

# Production Transformations
def production_transform() -> Callable[[Image], Tensor]:
    return v2.Compose([
        v2.Resize((224, 224)),              # Resize to 224x224
        v2.ToTensor(),                      # Convert image to PyTorch tensor
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize based on ImageNet
    ])