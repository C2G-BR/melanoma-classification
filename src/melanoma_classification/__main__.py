import typer
import mlflow
from pathlib import Path
from melanoma_classification.utils import (
    training,
    get_git_commit_hash,
    git_changes_detected,
)
from melanoma_classification.evaluation import evaluation
from melanoma_classification.inference import inference
from logging import getLogger, basicConfig, INFO

basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=INFO
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


@app.command()
def evaluate(data_path: str, experiment_name: str, run_name: str, epoch: int):
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Experiment Name: %s", experiment_name)
        logger.info("Experiment ID: %s", run.info.experiment_id)
        logger.info("Run Name: %s", run.info.run_name)
        logger.info("Run ID: %s", run.info.run_id)
        evaluation(data_path=data_path, epoch=epoch)


@app.command()
def infer(
    img_path: Path,
    experiment_name: str | None = None,
    run_name: str | None = None,
    epoch: int | None = None,
):
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            logger.info("Experiment Name: %s", experiment_name)
            logger.info("Experiment ID: %s", run.info.experiment_id)
            logger.info("Run Name: %s", run.info.run_name)
            logger.info("Run ID: %s", run.info.run_id)
            inference(img_path=img_path, epoch=epoch)
    else:
        inference(img_path=img_path, epoch=None)


if __name__ == "__main__":
    app()
