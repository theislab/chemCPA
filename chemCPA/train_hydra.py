from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning_module import ChemCPA
from omegaconf import OmegaConf

from chemCPA.data import PerturbationDataModule, load_dataset_splits
from chemCPA.paths import WB_DIR


@hydra.main(version_base=None, config_path="../config/", config_name="main")
def main(args):
    OmegaConf.set_struct(args, False)
    data_params = args["dataset"]
    datasets, dataset = load_dataset_splits(**data_params, return_dataset=True)

    dataset_config = {
        "num_genes": datasets["training"].num_genes,
        "num_drugs": datasets["training"].num_drugs,
        "num_covariates": datasets["training"].num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }
    dm = PerturbationDataModule(datasplits=datasets, train_bs=args["model"]["hparams"]["batch_size"])

    model = ChemCPA(args, dataset_config)

    wandb_logger = WandbLogger(**args["wandb"], save_dir=WB_DIR)
    run_id = wandb_logger.experiment.id

    checkpoint_callback = ModelCheckpoint(dirpath=Path(args["training"]["save_dir"]) / run_id, save_last=True, verbose=True)
    print(Path(args["training"]["save_dir"]) / run_id)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        logger=wandb_logger,
        max_epochs=args["training"]["num_epochs"],
        max_time=args["training"]["max_minutes"],
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=args["training"]["checkpoint_freq"],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
