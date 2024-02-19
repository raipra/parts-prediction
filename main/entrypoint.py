import gc
import os

from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

import mlflow
import torch
import torch.onnx

from azure.storage.blob import BlobServiceClient
from main.data.lex_glue import LexGlueDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from main.config_schemas.config_schema import Config

from main.lightning_modules.transformer import TransformerModule
from main.utils.config_utils import get_config
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from main.utils.mlflow_utils import log_artifacts_for_reproducibility
from main.utils.torch_utils import get_local_rank
from main.utils.utils import get_logger
from main.utils.mlflow_utils import activate_mlflow
from main.config_schemas.config_schema import TrainerConfig

@get_config(config_path="../configs", config_name="config")  # type: ignore
def run(config: "TrainerConfig") -> TransformerModule:
    experiment_name = config.infrastructure.mlflow.experiment_name
    run_id = config.infrastructure.mlflow.run_id
    run_name = config.infrastructure.mlflow.run_name

    with activate_mlflow(experiment_name=experiment_name, run_id=run_id, run_name=run_name) as _:
        """Train and checkpoint the model with highest F1; log that model to MLflow and
        return it."""
        model = TransformerModule(
            pretrained_model=config.trainer.pretrained_model,
            num_classes=config.trainer.num_classes,
            lr=config.trainer.lr,
        )
        datamodule = LexGlueDataModule(
            pretrained_model=config.trainer.pretrained_model,
            max_length=config.trainer.max_length,
            batch_size=config.trainer.batch_size,
            num_workers=config.trainer.num_workers,
            debug_mode_sample=config.trainer.debug_mode_sample,
        )

        # Wire up MLflow context manager to Azure ML.
        mlflow.set_experiment(config.trainer.mlflow_experiment_name)

        # Connect Lightning's MLFlowLogger plugin to azureml-mlflow as defined in the
        # context manager. TODO: MLflow metrics should show epochs rather than steps on
        #  the x-axis
        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=True,
        )
        mlf_logger._run_id = run.info.run_id
        mlflow.log_params({k: v for k, v in asdict(config).items() if not k.startswith("mlflow_")})

        # Keep the model with the highest F1 score.
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{Val_F1_Score:.2f}",
            monitor="Val_F1_Score",
            mode="max",
            verbose=True,
            save_top_k=1,
        )

        # Run the training loop.
        trainer = Trainer(
            callbacks=[
                EarlyStopping(
                    monitor="Val_F1_Score",
                    min_delta=config.trainer.min_delta,
                    patience=config.trainer.patience,
                    verbose=True,
                    mode="max",
                ),
                checkpoint_callback,
            ],
            default_root_dir=config.trainer.model_checkpoint_dir,
            fast_dev_run=bool(config.trainer.debug_mode_sample),
            max_epochs=config.trainer.max_epochs,
            max_time=config.trainer.max_time,
            log_every_n_steps = 20,
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
            logger=mlf_logger,
        )
        trainer.fit(model=model, datamodule=datamodule)
        best_model_path = checkpoint_callback.best_model_path

        # Evaluate the last and the best models on the test sample.
        trainer.test(model=model, datamodule=datamodule)
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_model_path,
        )

        return model, datamodule

    

def convert_to_onnx(
    model: torch.nn.Module,
    save_path: PathLike | str,
    sequence_length: int,
    vocab_size: int,
) -> None:
    model.eval()

    dummy_input_ids = torch.randint(
        0,
        vocab_size,
        (1, sequence_length),
        dtype=torch.long,
    )
    dummy_attention_mask = torch.ones(
        (1, sequence_length),
        dtype=torch.long,
    )
    dummy_label = torch.zeros(
        1,
        dtype=torch.long,
    )

    torch.onnx.export(
        model=model,
        args=(dummy_input_ids, dummy_attention_mask, dummy_label),
        f=save_path,
        input_names=["input_ids", "attention_mask", "label"],
    )


def copy_dir_to_abs(localdir: str, abs_container: str):
    """Copy the contents of a local directory to an Azure Blob Storage container.

    Args:
        localdir (str): The path to the local directory to copy.
        abs_container (str): The name of the Azure Blob Storage container.
    """
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))
    container = blob_service_client.get_container_client(abs_container)

    # Iterate through the local directory and upload each file while keeping the dir
    # structure.
    localdir = Path(localdir)
    for filepath in localdir.rglob("*"):
        if filepath.is_file():
            blobpath = filepath.relative_to(localdir)
            with filepath.open("rb") as file_data:
                container.upload_blob(name=str(blobpath), data=file_data)

    print("Successfully copied directory to Azure Blob Storage.")

@get_config(
    config_path="../configs/automatically_generated", config_name="config", to_object=False, return_dict_config=True
)
def run_tasks(config: Config) -> None:
    logger = get_logger(__file__)
    assert config.infrastructure.mlflow.run_id is not None, "Run id has to be set for running tasks"

    backend = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend)

    seed_everything(seed=config.lightningmodel.seed, workers=True)

    for task_name, task_config in config.tasks.items():
        logger.info(f"Running task: {task_name}")
        task = instantiate(task_config)
        task.run(config=config, task_config=task_config)
if __name__ == "__main__":
    # Free up gpu vRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()
    model, data_module = run()

"""     # Save model to ONNX format to the `onnx_path`.
    onnx_path = os.path.join(
        train_config.model_checkpoint_dir,
        train_config.mlflow_run_name + "_model.onnx.pb",
    )
    convert_to_onnx(
        model=model,
        save_path=onnx_path,
        sequence_length=train_config.max_length,
        vocab_size=data_module.tokenizer.vocab_size,
    )

    copy_dir_to_abs(
        localdir=train_config.model_checkpoint_dir,
        abs_container="model-artifacts",
    )
 """
