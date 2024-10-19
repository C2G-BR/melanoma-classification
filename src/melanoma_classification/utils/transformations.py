from torchvision import transforms

# Define ImageNet normalization parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Training Transformations
def train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),              # Resize to 224x224
        transforms.RandomHorizontalFlip(),          # Randomly flip the image horizontally
        transforms.RandomRotation(15),              # Random rotation between -15 and 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply color jitter
        transforms.ToTensor(),                      # Convert image to PyTorch tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize based on ImageNet
    ])

# Production Transformations
def production_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),              # Resize to 224x224
        transforms.ToTensor(),                      # Convert image to PyTorch tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize based on ImageNet
    ])