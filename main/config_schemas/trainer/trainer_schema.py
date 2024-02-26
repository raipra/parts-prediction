import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore

from main.config_schemas.trainer import callback_schema, logger_schema
from main.utils.mixin import LoggableParamsMixin


@dataclass
class TrainerConfig(LoggableParamsMixin):
    _target_: str = "lightning.pytorch.trainer.trainer.Trainer"
    accelerator: str = "auto"
    strategy: str = "ddp_find_unused_parameters_true"
    devices: str = "auto"
    num_nodes: int = 1  # SI("${}")
    precision: str = "16-mixed"
    fast_dev_run: bool = False
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[str] = None
    limit_train_batches: Optional[float] = 1
    limit_val_batches: Optional[float] = 1
    limit_test_batches: Optional[float] = 1
    limit_predict_batches: Optional[float] = 1
    overfit_batches: float = 0.0
    val_check_interval: Optional[float] = 1
    check_val_every_n_epoch: Optional[int] = 1
    num_sanity_val_steps: int = 2
    log_every_n_steps: int = 20
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = 5
    gradient_clip_algorithm: Optional[str] = "value"
    deterministic: Optional[bool] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    detect_anomaly: bool = False
    barebones: bool = False
    sync_batchnorm: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[str] = "./data/pytorch-lightning"

    pretrained_model: str = "roberta-base"
    num_classes: int = 2
    lr: float = 2e-4
    max_length: int = 128
    batch_size: int = 256
    debug_mode_sample: int | None = None
    model_checkpoint_dir: str = os.path.join(
        Path(__file__).parents[2],
        "model-checkpoints",
    )
    min_delta: float = 0.005
    patience: int = 4

    # MLflow
    mlflow_experiment_name: str = "FirstExperimemt"
    mlflow_run_name: str = "onnx-gpu-run12"
    mlflow_description: str = f"PEFT tune roberta-base to classify {mlflow_experiment_name}."

    def loggable_params(self) -> list[str]:
        return ["max_epochs", "max_steps", "strategy", "precision"]


@dataclass
class GPUDev(TrainerConfig):
    max_epochs: int = 3
    accelerator: str = "gpu"
    log_every_n_steps: int = 1
    limit_train_batches: float = 0.01
    limit_val_batches: float = 0.01
    limit_test_batches: float = 0.01
    logger: Optional[list[logger_schema.LoggerConfig]] = field(
        default_factory=lambda: [logger_schema.MLFlowLoggerConfig()]
    )  # type: ignore
    callbacks: Optional[list[callback_schema.CallbackConfig]] = field(
        default_factory=lambda: [
            callback_schema.ValidationF1ScoreBestModelCheckpointConfig(),
            callback_schema.LastModelCheckpointConfig(),
            callback_schema.LearningRateMonitorConfig(),
        ]
    )


@dataclass
class GPUProd(TrainerConfig):
    max_epochs: int = 20
    accelerator: str = "gpu"
    log_every_n_steps: int = 20
    logger: Optional[list[logger_schema.LoggerConfig]] = field(
        default_factory=lambda: [logger_schema.MLFlowLoggerConfig()]
    )  # type: ignore
    callbacks: Optional[list[callback_schema.CallbackConfig]] = field(
        default_factory=lambda: [
            callback_schema.ValidationF1ScoreBestModelCheckpointConfig(),
            callback_schema.LastModelCheckpointConfig(),
            callback_schema.LearningRateMonitorConfig(),
        ]
    )


def setup_config() -> None:
    logger_schema.setup_config()
    callback_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="trainer_schema", group="trainer", node=TrainerConfig)
