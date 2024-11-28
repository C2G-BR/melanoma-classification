import mlflow.artifacts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
from melanoma_classification.model.vision_transformer import VisionTransformer
import mlflow

CHECKPOINT_FILE = "checkpoints/epoch_{epoch}.json"


def _train(dataloader, model, criterion, optimizer, epoch, device):
    running_loss = 0.0
    y_true = torch.empty(0, device=device, dtype=torch.float)
    y_pred = torch.empty(0, device=device, dtype=torch.float)

    model.train()
    for images, labels in (
        tbar := tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch+1} [Training]",
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

        y_true = torch.cat((y_true, labels))
        y_pred = torch.cat((y_pred, predicted))
        total_predictions = y_pred.shape[0]

        tbar.set_postfix(
            loss=running_loss / total_predictions,
            accuracy=100.0
            * (y_pred == y_true).sum().item()
            / total_predictions,
        )

    accuracy = (y_pred == y_true).sum().item() / total_predictions
    y_true, y_pred = y_true.cpu(), y_pred.cpu()

    mlflow.log_metrics(
        metrics={
            "train_f1": f1_score(y_true, y_pred),
            "train_precision": precision_score(y_true, y_pred),
            "train_recall": recall_score(y_true, y_pred),
            "train_accuracy": accuracy,
            "train_loss": running_loss / total_predictions,
        },
        step=epoch,
    )


def _eval(dataloader, model, criterion, scheduler, epoch, device):
    running_loss = 0.0
    y_true = torch.empty(0, device=device, dtype=torch.float)
    y_pred = torch.empty(0, device=device, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        for images, labels in (
            vbar := tqdm(
                dataloader,
                unit="batch",
                desc=f"Epoch {epoch+1} [Validation]",
            )
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)["outputs"]
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = torch.argmax(outputs, 1)

            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predicted))
            total_predictions = y_pred.shape[0]

            vbar.set_postfix(
                val_loss=running_loss / total_predictions,
                val_accuracy=100.0
                * (y_pred == y_true).sum().item()
                / total_predictions,
            )

    accuracy = (y_pred == y_true).sum().item() / total_predictions
    y_true, y_pred = y_true.cpu(), y_pred.cpu()
    loss = running_loss / total_predictions

    mlflow.log_metrics(
        metrics={
            "validation_f1": f1_score(y_true, y_pred),
            "validation_precision": precision_score(y_true, y_pred),
            "validation_recall": recall_score(y_true, y_pred),
            "validation_accuracy": accuracy,
            "validation_loss": loss,
        },
        step=epoch,
    )
    scheduler.step(loss)


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
    device: torch.device,
    freezed_epochs: int = 0,
    num_unfreeze_layers: int | None = None,
    save_every_n_epochs: int = 5,
    init_epoch: int | None = None,
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
        freezed_epochs: The number of epochs to freeze the backbone.
        num_unfreeze_layers: The number of layers to unfreeze sequentially. If
            None, unfreezes all layers.
        save_every_n_epochs: Save a checkpoint every n epochs.
        init_epoch: Epoch to start from again. This indicates, that the run
            already exists.
    """
    if init_epoch:
        # Resuming Training
        checkpoint = mlflow.artifacts.load_dict(
            CHECKPOINT_FILE.format(epoch=init_epoch)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                # Handle nested states (e.g., momentum buffers)
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

        start_epoch = checkpoint["epoch"] + 1
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        # Beginning new Training
        start_epoch = 0

    if freezed_epochs > start_epoch:
        model.freeze_backbone()

    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        if (
            epoch >= freezed_epochs
            and freezed_epochs != 0
            and (
                num_unfreeze_layers is None
                or epoch <= freezed_epochs + num_unfreeze_layers
            )
        ):
            model.unfreeze_sequentially()

        _train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch+1,
            device=device,
        )
        _eval(
            dataloader=val_loader,
            model=model,
            criterion=criterion,
            scheduler=scheduler,
            epoch=epoch+1,
            device=device,
        )

        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == num_epochs:
            data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            mlflow.log_dict(
                data, artifact_file=CHECKPOINT_FILE.format(epoch=epoch + 1)
            )
            print(f"Checkpoint for epoch {epoch+1} saved.")
