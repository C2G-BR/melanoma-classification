import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DermMel(Dataset):
    def __init__(
        self, root_dir: str, split: str = "train_sep", transform: any = None
    ) -> None:
        """Constructor for the DermMel dataset.

        Args:
            root_dir: The root directory of the dataset.
            split: The split to load (train_sep, valid, test).
            transform: The transformations to apply to the images.
        """

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.classes = ["Melanoma", "NotMelanoma"]
        self.image_paths: list[Path] = []
        self.labels = []

        base_path = "DermMel"

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(
                self.root_dir, base_path, self.split, class_name
            )
            for img_file in os.listdir(class_dir):
                if img_file.endswith((".jpg", ".jpeg")):
                    self.image_paths.append(Path(class_dir, img_file))
                    self.labels.append(label)

    def __len__(self) -> int:
        """Get the number of images in the dataset.

        Returns:
            The number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Get an image and its label.

        Args:
            idx: The index of the image to retrieve.

        Returns:
            The image, its label, and its id.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label, img_path.stem

    def visualize_image(self, idx: int) -> None:
        """Visualize an image and its corresponding label from the DermMel
        dataset.

        Args:
            idx: The index of the image to visualize.
        """
        # Get the image and label
        image, label = self[idx]

        # If the image is a normalized tensor, undo the normalization
        if isinstance(image, torch.Tensor):
            # If normalization was applied, we should reverse it
            if hasattr(self.transform, "transforms"):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        # Undo normalization: image * std + mean
                        mean = torch.tensor(t.mean).view(-1, 1, 1)
                        std = torch.tensor(t.std).view(-1, 1, 1)
                        image = image * std + mean

            # Convert the image tensor to NumPy and transpose to HWC
            image = image.permute(1, 2, 0).numpy()
            # Clip to the valid range [0, 1] for displaying
            image = np.clip(image, 0, 1)

        # Get the class name from the label
        label_name = self.classes[label]

        # Plot the image
        plt.imshow(image)
        plt.title(f"Label: {label_name}")
        plt.axis("off")  # Hide the axes
        plt.show()
