import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DermMel(Dataset):

    def __init__(self, root_dir:str, split:str='train_sep', transform:any=None) -> None:
        """Initialize the dataset

        Parameters
        ----------
        root_dir : str
            The root directory of the dataset
        split : str, optional
            The split to load (train_sep, valid, test), by default 'train_sep'
        transform : any, optional
            The transformations to apply to the images, by default None
        """
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform()
        
        # Define the subdirectories for each class (Melanoma, NotMelanoma)
        self.classes = ['Melanoma', 'NotMelanoma']
        self.image_paths = []
        self.labels = []

        base_path = 'DermMel'

        # Load image paths and labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, base_path, self.split, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx:int) -> tuple[torch.Tensor, int]:
        """Get an image and its label

        Parameters
        ----------
        idx : int
            The index of the image to retrieve
        Returns
        -------
        tuple[torch.Tensor, int]
            The image and its label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get the label
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def visualize_image(self, idx: int) -> None:
        """
        Visualize an image and its corresponding label from the DermMel dataset.

        Parameters
        ----------
        idx : int
            The index of the image to visualize
        """
        # Get the image and label
        image, label = self[idx]

        # If the image is a normalized tensor, undo the normalization
        if isinstance(image, torch.Tensor):
            # If normalization was applied, we should reverse it
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        # Undo normalization: image * std + mean
                        mean = torch.tensor(t.mean).view(-1, 1, 1)
                        std = torch.tensor(t.std).view(-1, 1, 1)
                        image = image * std + mean
            
            # Convert the image tensor to NumPy and transpose to HWC (Height, Width, Channels)
            image = image.permute(1, 2, 0).numpy()
            # Clip to the valid range [0, 1] for displaying
            image = np.clip(image, 0, 1)

        # Get the class name from the label
        label_name = self.classes[label]

        # Plot the image
        plt.imshow(image)
        plt.title(f"Label: {label_name}")
        plt.axis('off')  # Hide the axes
        plt.show()