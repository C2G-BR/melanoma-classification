import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)
import pandas as pd
import os


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    checkpoint_path: str,
    metrics_df: pd.DataFrame
) -> None:
    """Saves the model, optimizer, and scheduler state.

    Args:
        model: 
        optimizer: 
        scheduler: 
        epoch:
        checkpoint_path:
        metrics_df:
    """

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
 
    checkpoint_file = os.path.join(
        checkpoint_path,
        f'checkpoint_epoch_{epoch+1}.pth'
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_file)

    # Save metrics
    metrics_df.to_csv(
        os.path.join(checkpoint_path, 'metrics.csv'),
        index=False
    )

    print(f"Checkpoint saved: {checkpoint_file}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    checkpoint_path: str,
    save_every_n_epochs: int = 5,
    resume_training: bool = False,
    checkpoint_model_file: callable = None,
    checkpoint_metrics_file: str = 'metrics.csv'
) -> None:
    

    # Initialize metrics
    start_epoch = 0

    # Load checkpoint if resuming training
    if resume_training:
        # Checkpoint contains model, optimizer, epoch, and scheduler
        checkpoint = torch.load(
            os.path.join(checkpoint_path, checkpoint_model_file)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        metrics_df = pd.read_csv(
            os.path.join(checkpoint_path, checkpoint_metrics_file)
        )
    
    # Move model to device
    model.to(device)

    # Start/continue training
    for epoch in range(start_epoch, num_epochs):
        
        model.train()
        running_loss = 0.0
        true_positives = 0
        total_predictions = 0
        y_true_train = []       # True labels
        y_pred_train = []       # Predicted labels

        with tqdm(train_loader, unit='batch') as tepoch:

            tepoch.set_description(f'Epoch {epoch+1}/{num_epochs} [Training]')

            for images, labels in tepoch:

                # Move images and labels to device
                images, labels = images.to(device), labels.to(device)

                # Reset gradients after each batch
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Predicitons & Metrics
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                true_positives += (predicted == labels).sum().item()

                # Store true labels and predictions for F1 calculation
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())

                # Update progress bar
                tepoch.set_postfix(
                    loss=running_loss/total_predictions,
                    accuracy=100. * true_positives / total_predictions
                )

        # Calculate metrics for training set
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * true_positives / total_predictions

        # Calculate precision, recall, F1-score
        train_f1 = f1_score(y_true_train, y_pred_train, average='binary')
        train_precision = precision_score(
            y_true_train,
            y_pred_train,
            average='binary'
        )

        train_recall = recall_score(
            y_true_train,
            y_pred_train,
            average='binary'
        )

        # Prepare validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true_val = []
        y_pred_val = []

        # Evaluation
        with torch.no_grad():
            with tqdm(val_loader, unit='batch') as vepoch:
                vepoch.set_description(
                    f"Epoch {epoch+1}/{num_epochs} [Validation]"
                )
                
                for images, labels in vepoch:

                    # Move images and labels to device
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images)

                    # Compute loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Predictions & Metrics
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Store true labels and predictions for F1 calculation
                    y_true_val.extend(labels.cpu().numpy())
                    y_pred_val.extend(predicted.cpu().numpy())

                    # Update progress bar
                    vepoch.set_postfix(
                        val_loss=val_loss/total,
                        val_accuracy=100. * correct / total
                    )

        # Calculate metrics for validation set
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / total

        # Calculate precision, recall, F1-score for validation
        val_f1 = f1_score(y_true_val, y_pred_val, average='binary')
        val_precision = precision_score(
            y_true_val, y_pred_val, average='binary'
        )
        val_recall = recall_score(y_true_val, y_pred_val, average='binary')

        # Learning rate scheduling
        scheduler.step()

        # Append metrics to DataFrame
        metrics_df = metrics_df.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall
        }, ignore_index=True)

        # Save the DataFrame every epoch
        metrics_df.to_csv(
            os.path.join(checkpoint_path, 'metrics.csv'),
            index=False
        )

        # Save checkpoint every X epochs
        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                checkpoint_path,
                metrics_df
            )