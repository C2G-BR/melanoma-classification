import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os
from pathlib import Path
from melanoma_classification.model.vision_transformer import VisionTransformer


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
    device: torch.device,
    checkpoint_path: Path,
    freezed_epochs: int = 0,
    save_every_n_epochs: int = 5,
    checkpoint_model_file: str = None,
    checkpoint_metrics_file: str = "metrics.csv",
) -> None:
    """Trains the model.

    Args:
        model: The model to train.
        train_loader: The DataLoader for the training set.
        val_loader: The DataLoader for the validation set.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        num_epochs: The number of epochs to train.
        device: The device to train on.
        checkpoint_path: The path to save the checkpoints.
        freezed_epochs: The number of epochs to freeze the backbone.
        save_every_n_epochs: Save a checkpoint every n epochs.
        checkpoint_model_file: The model checkpoint file to resume training.
        checkpoint_metrics_file: The metrics checkpoint file to resume training.
    """
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_model_file:
        # Resuming Training
        checkpoint = torch.load(
            checkpoint_path / checkpoint_model_file, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)

            # Handle nested states (e.g., momentum buffers)
            elif isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

        start_epoch = checkpoint["epoch"] + 1
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        metrics_df = pd.read_csv(checkpoint_path / checkpoint_metrics_file)
    else:
        # Beginning new Training
        start_epoch = 0
        metrics_df = pd.DataFrame()

    if freezed_epochs > start_epoch:
        model.freeze_backbone()

    model.to(device)

    # TRAINING
    model.train()
    for epoch in range(start_epoch, num_epochs):
        if epoch >= freezed_epochs and freezed_epochs != 0:
            model.unfreeze_sequentially()
            model.unfreeze_sequentially()

        running_loss = 0.0
        true_positives = 0
        total_predictions = 0
        y_true_train = []
        y_pred_train = []

        for images, labels in (
            tbar := tqdm(
                train_loader,
                unit="batch",
                desc=f"Epoch {epoch+1}/{num_epochs} [Training]",
            )
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)["outputs"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.argmax(outputs, 1)
            total_predictions += labels.size(0)
            true_positives += (predicted == labels).sum().item()

            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

            tbar.set_postfix(
                loss=running_loss / total_predictions,
                accuracy=100.0 * true_positives / total_predictions,
            )

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * true_positives / total_predictions
        train_f1 = f1_score(y_true_train, y_pred_train)
        train_precision = precision_score(y_true_train, y_pred_train)
        train_recall = recall_score(y_true_train, y_pred_train)

        # VALIDATION
        val_loss = 0.0
        correct = 0
        total = 0
        y_true_val = []
        y_pred_val = []

        model.eval()
        with torch.no_grad():
            for images, labels in (
                vbar := tqdm(
                    val_loader,
                    unit="batch",
                    desc=f"Epoch {epoch+1}/{num_epochs} [Validation]",
                )
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)["outputs"]
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

                vbar.set_postfix(
                    val_loss=val_loss / total,
                    val_accuracy=100.0 * correct / total,
                )

        val_loss /= len(val_loader.dataset)
        val_acc = 100.0 * correct / total
        val_f1 = f1_score(y_true_val, y_pred_val)
        val_precision = precision_score(y_true_val, y_pred_val)
        val_recall = recall_score(y_true_val, y_pred_val)

        scheduler.step(val_loss)

        new_row = pd.DataFrame(
            [
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_f1": train_f1,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
            ]
        )
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        if (
            (epoch + 1) % save_every_n_epochs == 0
            or (epoch + 1) == num_epochs
            and not os.path.exists(checkpoint_path)
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path / f"checkpoint_epoch_{epoch+1}.pth",
            )
            metrics_df.to_csv(checkpoint_path / "metrics.csv", index=False)
            print(f"Checkpoint for epoch {epoch+1} saved.")
