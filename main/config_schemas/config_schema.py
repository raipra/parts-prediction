import os

from dataclasses import field
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from main.config_schemas.trainer.trainer_schema import TrainerConfig
from main.utils.mixin import LoggableParamsMixin
from main.config_schemas.infrastructure import infrastructure_schema


@dataclass
class LightningModuleConfig(LoggableParamsMixin):
    _target_: str = MISSING


@dataclass
class TransformerModuleConfig(LightningModuleConfig):
    _target_: str = ""
    
    save_last_checkpoint_every_n_train_steps: int = 500
    seed: int = 1234
    registered_model_name: Optional[str] = None
    docker_image: Optional[str] = None

    pretrained_model: str = "roberta-base"
    num_classes: int = 2
    lr: float = 2e-4
    max_length: int = 128
    batch_size: int = 256
    num_workers: int = os.cpu_count()
    max_epochs: int = 10
    debug_mode_sample: int | None = None
    max_time: dict[str, float] = field(default_factory=lambda: {"hours": 3})
    model_checkpoint_dir: str = os.path.join(
        Path(__file__).parents[2],
        "model-checkpoints",
    )
    min_delta: float = 0.005
    patience: int = 4

    # MLflow
    mlflow_experiment_name: str = "Lex-GLUE Terms of Service"
    mlflow_run_name: str = "onnx-gpu-run12"
    mlflow_description: str = f"PEFT tune roberta-base to classify {mlflow_experiment_name}."


@dataclass
class Config:
    trainer: TrainerConfig = TrainerConfig()
    lightningmodel: TransformerModuleConfig = TransformerModuleConfig()
    infrastructure: infrastructure_schema.InfrastructureConfig =  infrastructure_schema.InfrastructureConfig()
    docker_image: Optional[str] = None


def setup_config() -> None:
    infrastructure_schema.setup_config()
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
