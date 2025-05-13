from tempfile import TemporaryDirectory
from typing import Protocol

import mlflow
import torch


class StateDictContainer(Protocol):
    def state_dict(self): ...
    def load_state_dict(self, state_dict): ...


def log_state_dict(
    container: StateDictContainer, artifact_path: str, file_name: str
) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = tmpdir + "/" + file_name
        torch.save(container.state_dict(), tmp)
        mlflow.log_artifact(tmp, artifact_path)


def load_state_dict(
    run: mlflow.ActiveRun, container: StateDictContainer, artifact_path: str
):
    with TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, dst_path=tmpdir, artifact_path=artifact_path
        )
        container.load_state_dict(torch.load(local_path, weights_only=True))
