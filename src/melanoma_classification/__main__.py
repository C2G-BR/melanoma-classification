from logging import INFO, basicConfig, getLogger
from pathlib import Path

import mlflow
import typer

from melanoma_classification.evaluation import evaluation
from melanoma_classification.inference import inference
from melanoma_classification.utils import (
    get_git_commit_hash,
    git_changes_detected,
    training,
)

basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=INFO
)
logger = getLogger(__name__)


app = typer.Typer(
    name="Melanoma Classification", pretty_exceptions_enable=False
)


@app.command()
def train(
    data_path: str,
    experiment_name: str,
    run_id: str | None = None,
    epoch: int | None = None,
    num_epochs: int | None = 50,
):
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

    # Variables required to continue a run.
    cont_run_vars = [run_id, epoch]
    continue_run = all(var is not None for var in cont_run_vars)
    new_run = all(var is None for var in cont_run_vars)

    if not new_run and not continue_run:
        logger.warning(
            "To continue a run, all necessary variables must be provided.",
        )
        return

    if continue_run:
        previous_commit_hash = mlflow.get_run(run_id).data.params.get(
            "commit_hash"
        )
        if previous_commit_hash != commit_hash:
            logger.warning(
                "The commit hash of the previous run does not match the current"
                " commit hash. This is not recommended as results will not be "
                "reproducible. Please stop the run and commit your changes. "
                "The commit hash is used for model versioning."
            )

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_id=run_id) as run:
        if new_run:
            mlflow.log_param("commit_hash", commit_hash)
        logger.info("Commit Hash: %s", commit_hash)
        logger.info("Experiment Name: %s", experiment_name)
        logger.info("Experiment ID: %s", run.info.experiment_id)
        logger.info("Run Name: %s", run.info.run_name)
        logger.info("Run ID: %s", run.info.run_id)
        training(
            run=run,
            data_path=data_path,
            num_epochs=num_epochs,
            freezed_epochs=5,
            num_unfreeze_layers=None,
            save_every_n_epochs=1,
            init_epoch=epoch,
        )


@app.command()
def evaluate(data_path: str, experiment_name: str, run_id: str, epoch: int):
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_id=run_id) as run:
        logger.info("Experiment Name: %s", experiment_name)
        logger.info("Experiment ID: %s", run.info.experiment_id)
        logger.info("Run Name: %s", run.info.run_name)
        logger.info("Run ID: %s", run.info.run_id)
        evaluation(run, data_path=data_path, epoch=epoch)


@app.command()
def infer(
    img_path: Path,
    experiment_name: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
):
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_id=run_id) as run:
            logger.info("Experiment Name: %s", experiment_name)
            logger.info("Experiment ID: %s", run.info.experiment_id)
            logger.info("Run Name: %s", run.info.run_name)
            logger.info("Run ID: %s", run.info.run_id)
            inference(run=run, img_path=img_path, epoch=epoch)
    else:
        inference(run=None, img_path=img_path, epoch=None)


if __name__ == "__main__":
    app()
