import mlflow
import torch
from melanoma_classification.model import get_dermmel_classifier_v1
from melanoma_classification.utils import (
    get_device,
    DermMel,
    production_transform,
)
from melanoma_classification.paths import (
    MODEL_STATE_DICT,
    EVALUATION_FILES,
    IMAGE_FILES,
)
from logging import getLogger
from melanoma_classification.evaluation.report import (
    create_evaluation_report,
    visualize_confusion_matrix,
    visualize_model_confidence,
)

logger = getLogger(__name__)


def evaluation(data_path: str, epoch: int) -> None:
    device = get_device()
    model = get_dermmel_classifier_v1()
    model.load_state_dict(
        mlflow.artifacts.load_dict(MODEL_STATE_DICT.format(epoch=epoch))
    )
    model.to(device)
    dataset = DermMel(data_path, split="test", transform=production_transform())
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )

    report = create_evaluation_report(
        model=model, test_dataloader=dataloader, device=device
    )
    mlflow.log_table(
        report, EVALUATION_FILES.format(epoch=epoch) / "evaluation_report.csv"
    )

    fig = visualize_confusion_matrix(
        evaluation_report=report, class_labels=dataset.class_labels
    )
    mlflow.log_figure(
        fig, IMAGE_FILES.format(epoch=epoch) / "confusion_matrix.png"
    )

    fig = visualize_model_confidence(evaluation_report=report)
    mlflow.log_figure(fig, IMAGE_FILES.format(epoch=epoch) / "confidence.png")

    model._save_to_state_dict()
