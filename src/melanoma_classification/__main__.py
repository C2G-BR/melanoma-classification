import numpy as np
import typer
import torch
from importlib.resources import files
import mlflow
from pathlib import Path
from PIL import Image
from melanoma_classification.utils import (
    get_device,
    production_transform,
    visualize_single_attention,
    visualize_multihead_as_single_attention,
    visualize_multihead_attention,
    training,
    get_git_commit_hash,
    git_changes_detected,
)
from melanoma_classification.model import get_dermmel_classifier_v1
from logging import getLogger, basicConfig, INFO

basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=INFO
)
logger = getLogger(__name__)


app = typer.Typer(
    name="Melanoma Classification", pretty_exceptions_enable=False
)


@app.command()
def train(data_path: str, experiment_name: str):
    if git_changes_detected():
        logger.warning(
            "Changes detected in the git repository. This is not recommended as"
            " results will not be reproducible. Please stop the run and commit "
            "your changes. The commit hash is used for model versioning.",
        )

    commit_hash = get_git_commit_hash()
    if commit_hash is None:
        logger.warning("Could not retrieve the git commit hash.")
        return

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_param("commit_hash", commit_hash)
        logger.info("Commit Hash: %s", commit_hash)
        logger.info("Experiment Name: %s", experiment_name)
        logger.info("Experiment ID: %s", run.info.experiment_id)
        logger.info("Run Name: %s", run.info.run_name)
        logger.info("Run ID: %s", run.info.run_id)
        training(
            data_path=data_path,
            num_epochs=50,
            freezed_epochs=5,
            num_unfreeze_layers=None,
            save_every_n_epochs=1,
            init_epoch=None,
        )


# @app.command()
# def evaluate(data_path: str, experiment_name: str, run_name: str, epoch: int):
#     mlflow.set_experiment(experiment_name=experiment_name)
#     with mlflow.start_run(run_name=run_name) as run:
#         logger.info("Experiment Name: %s", experiment_name)
#         logger.info("Experiment ID: %s", run.info.experiment_id)
#         logger.info("Run Name: %s", run.info.run_name)
#         logger.info("Run ID: %s", run.info.run_id)
#         _eval(data_path=data_path, epoch=epoch)


# @app.command()
# def predict(img_path: Path, model_path: Path | None = None):
#     if model_path is None:
#         model_path = files("melanoma_classification").joinpath(
#             "weights/vit.pth"
#         )
#     device = get_device()
#     raw_image = Image.open(img_path).convert("RGB")
#     image = (
#         production_transform()(image=np.array(raw_image))["image"]
#         .to(device)
#         .unsqueeze(0)
#     )
#     checkpoint = torch.load(model_path, weights_only=True)
#     model = get_dermmel_classifier_v1()
#     model.load_state_dict(checkpoint)
#     model.to(device=get_device())
#     model.eval()
#     with torch.no_grad():
#         model_outputs = model(image)
#         logits = model_outputs["outputs"]
#         attention = model_outputs["attentions"]
#         logits = torch.nn.functional.softmax(logits, 1)
#         confidence, prediction = torch.max(logits, dim=1)
#         confidence, prediction = confidence.item(), prediction.item()

#     detected = model.class_map[prediction]
#     print(f"Found a {detected} sample with confidence {confidence*100:.2f}%.")
#     visualize_single_attention(raw_image, attention)
#     visualize_multihead_as_single_attention(raw_image, attention)
#     visualize_multihead_attention(raw_image, attention)


if __name__ == "__main__":
    app()
