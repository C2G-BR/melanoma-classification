import time
from logging import getLogger

import mlflow
import mlflow.artifacts
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from melanoma_classification.model import get_dermmel_classifier_v1
from melanoma_classification.paths import (
    MODEL_STATE_DICT,
    OPTIMIZER_STATE_DICT,
    SCHEDULER_STATE_DICT,
    STATE_FILES,
)
from melanoma_classification.utils.dermmel import (
    DermMel,
)
from melanoma_classification.utils.devices import get_device
from melanoma_classification.utils.mlflow import (
    load_state_dict,
    log_state_dict,
)
from melanoma_classification.utils.transformations import (
    production_transform,
    train_transform,
)

logger = getLogger(__name__)


def _train(dataloader, model, criterion, optimizer, epoch, device):
    running_loss = 0.0
    y_true = torch.empty(0, device=device, dtype=torch.float)
    y_pred = torch.empty(0, device=device, dtype=torch.float)

    model.train()
    for images, labels, _ in (
        tbar := tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} [Training]",
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


def _validate(dataloader, model, criterion, scheduler, epoch, device):
    running_loss = 0.0
    y_true = torch.empty(0, device=device, dtype=torch.float)
    y_pred = torch.empty(0, device=device, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        for images, labels, _ in (
            vbar := tqdm(
                dataloader,
                unit="batch",
                desc=f"Epoch {epoch} [Validation]",
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


def training(
    run: mlflow.ActiveRun,
    data_path: str,
    num_epochs: int,
    freezed_epochs: int = 0,
    num_unfreeze_layers: int | None = None,
    save_every_n_epochs: int = 5,
    init_epoch: int | None = None,
) -> None:
    """Trains the model.

    Args:
        run: The active MLflow run.
        data_path: The path to the data.
        num_epochs: The number of epochs to train.
        freezed_epochs: The number of epochs to freeze the backbone.
        num_unfreeze_layers: The number of layers to unfreeze sequentially. If
            None, unfreezes all layers.
        save_every_n_epochs: Save a checkpoint every n epochs.
        init_epoch: Epoch to start from again. This indicates, that the run
            already exists.
    """
    device = get_device()
    model = get_dermmel_classifier_v1()
    model.to(device)
    train_dataloader = torch.utils.data.DataLoader(
        DermMel(data_path, split="train_sep", transform=train_transform()),
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )
    val_dataloader = torch.utils.data.DataLoader(
        DermMel(data_path, split="valid", transform=production_transform()),
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.cls_token, "lr": 1e-7},
            {"params": model.pos_embed, "lr": 1e-7},
            {"params": model.patch_embedding.parameters(), "lr": 1e-6},
            {"params": model.transformer_layers.parameters(), "lr": 1e-5},
            {"params": model.norm.parameters(), "lr": 1e-6},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )

    if init_epoch is None:
        logger.info("Initializing new training run.")
        start_epoch = 0
        model.load_pretrained_weights("deit_base_patch16_224")
        path = STATE_FILES.format(epoch=start_epoch)
        log_state_dict(
            container=model,
            artifact_path=path,
            file_name=MODEL_STATE_DICT,
        )
        log_state_dict(
            container=optimizer,
            artifact_path=path,
            file_name=OPTIMIZER_STATE_DICT,
        )
        log_state_dict(
            container=scheduler,
            artifact_path=path,
            file_name=SCHEDULER_STATE_DICT,
        )
    else:
        logger.info("Resuming training from epoch %d.", init_epoch)
        start_epoch = init_epoch
        base_path = STATE_FILES.format(epoch=start_epoch) + "/"
        load_state_dict(
            run=run,
            container=model,
            artifact_path=base_path + MODEL_STATE_DICT,
        )
        load_state_dict(
            run=run,
            container=optimizer,
            artifact_path=base_path + OPTIMIZER_STATE_DICT,
        )
        load_state_dict(
            run=run,
            container=scheduler,
            artifact_path=base_path + SCHEDULER_STATE_DICT,
        )

        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                # Handle nested states (e.g., momentum buffers)
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
    model.to(device)

    if freezed_epochs > start_epoch:
        model.freeze_backbone()

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

        start = time.time()
        _train(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch + 1,
            device=device,
        )
        train = time.time()
        _validate(
            dataloader=val_dataloader,
            model=model,
            criterion=criterion,
            scheduler=scheduler,
            epoch=epoch + 1,
            device=device,
        )
        validation = time.time()
        mlflow.log_metric(key="training_time", value=train - start, step=epoch)
        mlflow.log_metric(
            key="validation_time", value=validation - train, step=epoch
        )

        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == num_epochs:
            checkpoint = STATE_FILES.format(epoch=epoch + 1)
            log_state_dict(
                container=model,
                artifact_path=checkpoint,
                file_name=MODEL_STATE_DICT,
            )
            log_state_dict(
                container=optimizer,
                artifact_path=checkpoint,
                file_name=OPTIMIZER_STATE_DICT,
            )
            log_state_dict(
                container=scheduler,
                artifact_path=checkpoint,
                file_name=SCHEDULER_STATE_DICT,
            )
            logger.info("Checkpoint for epoch %d saved.", epoch + 1)
