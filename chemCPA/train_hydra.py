from pathlib import Path
import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import numpy as np

from chemCPA.data.data import PerturbationDataModule, load_dataset_splits
from chemCPA.paths import WB_DIR
from lightning_module import ChemCPA  # your LightningModule containing ComPert usage


@hydra.main(version_base=None, config_path="../config/", config_name="main")
def main(args):
    OmegaConf.set_struct(args, False) 

    data_params = args["dataset"]
    datasets, dataset = load_dataset_splits(**data_params, return_dataset=True)
    dm = PerturbationDataModule(datasplits=datasets, train_bs=args["model"]["hparams"]["batch_size"])
    dataset_config = {
        "num_genes": datasets["training"].num_genes,
        "num_drugs": datasets["training"].num_drugs,
        "num_covariates": datasets["training"].num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }
    dataset.debug_print()

    # Initialize model
    model = ChemCPA(args, dataset_config)

    # 1) Check drug indices are in range
    drugs_idx = datasets["training"].drugs_idx
    print(f"drugs_idx range: {drugs_idx.min().item()} to {drugs_idx.max().item()}, total num_drugs={dataset_config['num_drugs']}")
    assert drugs_idx.min() >= 0, "Negative drug index found!"
    assert drugs_idx.max() < dataset_config["num_drugs"], "Drug index out of range!"

    # 2) After your ChemCPA model is instantiated:
    embedding_w = model.drug_embeddings.weight.data  # shape [num_drugs, embedding_dim]
    if torch.isnan(embedding_w).any():
        bad_rows = torch.where(torch.isnan(embedding_w).any(dim=1))[0]
        print(f"NaNs in embedding row(s): {bad_rows.tolist()}")
        raise ValueError("drug_embeddings contains NaNs!")

    # Load pretrained weights if specified
    if args["model"]["load_pretrained"]:
        print("Debug - full model config:", args["model"])
        pretrained_path = Path(args["model"]["pretrained_model_path"])
        model_hash = args["model"]["pretrained_model_hashes"].get("model")
        print("Debug - model_hash:", model_hash)
        checkpoint_path = pretrained_path / model_hash / "last.ckpt"
        print(f"Using pretrained model weights from {checkpoint_path}")
        
        if model_hash and checkpoint_path.exists():
            print(f"Loading pretrained model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['state_dict']
            
            # Remove components we want to train from scratch
            keys = list(state_dict.keys())
            for key in keys:
                if (key.startswith('model.adversary_drugs') or
                    key.startswith('model.drug_embeddings') or
                    key.startswith('drug_embeddings') or
                    key.startswith('adversary_drugs')):
                    state_dict.pop(key)

            # For vanilla CPA, also remove doser and drug embedding encoder
            if args["model"]["embedding"]["model"] == "vanilla":
                keys = list(state_dict.keys())
                for key in keys:
                    if (key.startswith('model.dosers') or
                        key.startswith('model.drug_embedding_encoder') or
                        key.startswith('dosers') or
                        key.startswith('drug_embedding_encoder')):
                        state_dict.pop(key)

            # Handle covariate embeddings - map between old and new indices
            if 'model.covariates_embeddings.0.weight' in state_dict:
                old_embeddings = state_dict['model.covariates_embeddings.0.weight']  # e.g. shape [82, 32]
                embedding_dim = old_embeddings.shape[1]
                
                new_cell_types = np.unique(datasets['training'].covariate_names['cell_type'])
                old_cell_types = np.unique(dataset.covariate_names['cell_type'])
                
                print("\nDEBUG Unique cell types:")
                print(f"New dataset: {new_cell_types}")
                print(f"Old dataset: {old_cell_types}")
                
                # Create new embeddings tensor
                num_covariates = datasets["training"].num_covariates[0]
                new_embeddings = torch.zeros((num_covariates, embedding_dim), device=old_embeddings.device)
                
                for new_idx, cell_type in enumerate(new_cell_types):
                    if cell_type in old_cell_types:
                        old_idx = np.where(old_cell_types == cell_type)[0][0]
                        new_embeddings[new_idx] = old_embeddings[old_idx]
                    else:
                        print(f"Warning: Cell type {cell_type} not found in pretrained model")
                
                state_dict['model.covariates_embeddings.0.weight'] = new_embeddings

            # Handle adversary covariates similarly
            if 'model.adversary_covariates.0.network.9.weight' in state_dict:
                old_layer_weight = state_dict['model.adversary_covariates.0.network.9.weight']
                old_layer_bias = state_dict['model.adversary_covariates.0.network.9.bias']
                hidden_dim = old_layer_weight.shape[1]
                
                new_cell_types = np.unique(datasets['training'].covariate_names['cell_type'])
                old_cell_types = np.unique(dataset.covariate_names['cell_type'])
                num_covariates = datasets["training"].num_covariates[0]
                
                new_layer_weight = torch.zeros((num_covariates, hidden_dim), device=old_layer_weight.device)
                new_layer_bias = torch.zeros(num_covariates, device=old_layer_bias.device)
                
                for new_idx, cell_type in enumerate(new_cell_types):
                    if cell_type in old_cell_types:
                        old_idx = np.where(old_cell_types == cell_type)[0][0]
                        new_layer_weight[new_idx] = old_layer_weight[old_idx]
                        new_layer_bias[new_idx] = old_layer_bias[old_idx]
                    else:
                        print(f"Warning: Cell type {cell_type} not found in pretrained model")
                
                state_dict['model.adversary_covariates.0.network.9.weight'] = new_layer_weight
                state_dict['model.adversary_covariates.0.network.9.bias'] = new_layer_bias

            model_sd = model.state_dict()
            keys = list(state_dict.keys())
            for k in keys:
                if k in model_sd:
                    if state_dict[k].shape != model_sd[k].shape:
                        print(
                            f"Skipping param '{k}' due to shape mismatch: "
                            f"checkpoint {state_dict[k].shape} vs model {model_sd[k].shape}"
                        )
                        state_dict.pop(k)

            # Now load what remains
            incomp_keys = model.load_state_dict(state_dict, strict=False)
            print("Missing keys:", incomp_keys.missing_keys)
            print("Unexpected keys:", incomp_keys.unexpected_keys)
        else:
            print(f"Warning: Pretrained model file not found at {checkpoint_path}")
    else:
        print(f"Warning: No pretrained hash found for model")

    wandb_logger = WandbLogger(**args["wandb"], save_dir=WB_DIR)
    run_id = wandb_logger.experiment.id

    checkpoint_callback = ModelCheckpoint(dirpath=Path(args["training"]["save_dir"]) / run_id, save_last=True)
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

