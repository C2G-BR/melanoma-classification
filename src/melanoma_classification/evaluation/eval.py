from logging import getLogger

import mlflow
import torch

from melanoma_classification.evaluation.report import (
    create_evaluation_report,
    visualize_confusion_matrix,
    visualize_model_confidence,
)
from melanoma_classification.model import get_dermmel_classifier_v1
from melanoma_classification.paths import (
    EVALUATION_FILES,
    IMAGE_FILES,
    MODEL_STATE_DICT,
    STATE_FILES,
)
from melanoma_classification.utils import (
    DermMel,
    get_device,
    load_state_dict,
    production_transform,
)

logger = getLogger(__name__)


def evaluation(run: mlflow.ActiveRun, data_path: str, epoch: int) -> None:
    device = get_device()
    model = get_dermmel_classifier_v1()

    load_state_dict(
        run=run,
        container=model,
        artifact_path=STATE_FILES.format(epoch=epoch) + "/" + MODEL_STATE_DICT,
    )
    model.to(device)
    dataset = DermMel(data_path, split="test", transform=production_transform())
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    report = create_evaluation_report(
        model=model, dataloader=dataloader, device=device
    )
    mlflow.log_table(
        report, EVALUATION_FILES.format(epoch=epoch) + "/evaluation_report.parquet"
    )

    fig = visualize_confusion_matrix(
        evaluation_report=report, class_labels=dataset.classes
    )
    mlflow.log_figure(
        fig, IMAGE_FILES.format(epoch=epoch) + "/confusion_matrix.png"
    )

    fig = visualize_model_confidence(evaluation_report=report)
    mlflow.log_figure(fig, IMAGE_FILES.format(epoch=epoch) + "/confidence.png")
