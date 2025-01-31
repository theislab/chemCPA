from sympy import N
import hydra
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only

# Example import paths (adjust to your project structure)
from chemCPA.data.data import load_dataset_splits
from chemCPA.embedding import get_chemical_representation
from chemCPA.model import ComPert
from chemCPA.train import evaluate_disentanglement, evaluate_r2


class ChemCPA(L.LightningModule):
    def __init__(self, config, dataset_config):
        super().__init__()
        self.automatic_optimization = False  # We'll manually call .step()
        self.config = config

        # Retrieve hyperparams and additional model kwargs from config
        additional_params = self.config["model"]["additional_params"]
        hparams = self.config["model"]["hparams"]

        # Build / load any required embeddings
        self.drug_embeddings = get_chemical_representation(
            smiles=dataset_config["canon_smiles_unique_sorted"],
            embedding_model=self.config["model"]["embedding"],
            data_path=self.config["model"]["embedding"]["datapath"],
        )

        append_ae_layer = self.config["model"]["append_ae_layer"]
        append_layer_width = dataset_config["num_genes"] if append_ae_layer else None
        # Create the ComPert model
        self.model = ComPert(
            num_genes=dataset_config["num_genes"],
            num_drugs=dataset_config["num_drugs"],
            num_covariates=dataset_config["num_covariates"],
            device="cuda",
            hparams=hparams,
            drug_embeddings=self.drug_embeddings,
            use_drugs_idx=dataset_config["use_drugs_idx"],
            append_layer_width=append_layer_width,
            **additional_params
        )

        # Log hyperparameters in checkpoint
        self.save_hyperparameters()

    def forward(self, batch, return_latent_basal=False):
        """
        Wraps self.model.predict(...) in a consistent interface:
           - If return_latent_basal=True, returns ((mean, var), (latent_basal, drug_emb, latent_treated)).
           - If return_latent_basal=False, returns (mean, var).
        """
        if self.model.use_drugs_idx:
            # batch structure: (genes, drugs_idx, dosages, degs, *covariates)
            genes = batch[0]
            drugs_idx = batch[1]
            dosages = batch[2]
            covariates = batch[4:]  # after degs
            if return_latent_basal:
                gene_recon, cell_drug_emb, latents = self.model.predict(
                    genes=genes,
                    drugs_idx=drugs_idx,
                    dosages=dosages,
                    covariates=covariates,
                    return_latent_basal=True
                )
            else:
                gene_recon, cell_drug_emb = self.model.predict(
                    genes=genes,
                    drugs_idx=drugs_idx,
                    dosages=dosages,
                    covariates=covariates,
                    return_latent_basal=False
                )
                latents = None
        else:
            # batch structure: (genes, drugs, degs, *covariates)
            genes = batch[0]
            drugs = batch[1]
            covariates = batch[3:]  # after degs
            if return_latent_basal:
                gene_recon, cell_drug_emb, latents = self.model.predict(
                    genes=genes,
                    drugs=drugs,
                    covariates=covariates,
                    return_latent_basal=True
                )
            else:
                gene_recon, cell_drug_emb = self.model.predict(
                    genes=genes,
                    drugs=drugs,
                    covariates=covariates,
                    return_latent_basal=False
                )
                latents = None

        # gene_recon has shape [batch_size, 2 * num_genes]
        dim = gene_recon.shape[1] // 2
        mean = gene_recon[:, :dim]
        var = gene_recon[:, dim:]

        if return_latent_basal:
            # latents is (latent_basal, drug_embedding, latent_treated)
            return (mean, var), latents
        return mean, var

    def training_step(self, batch, batch_idx):
        # Retrieve our 3 optimizers (example code)
        optimizers = self.optimizers()
        optimizer_autoencoder = optimizers[0]
        optimizer_adversaries = optimizers[1]
        optimizer_dosers = optimizers[2]

        # 1. Forward pass (get mean, var + latents)
        (mean, var), latents = self.forward(batch, return_latent_basal=True)
        latent_basal = None
        if latents is not None:
            # latents = (latent_basal, drug_embedding, latent_treated)
            latent_basal = latents[0]

        # ---- Debug prints to detect NaNs in decoder outputs ----
        if torch.isnan(mean).any() or torch.isnan(var).any():
            print(
                "\n***** WARNING: NaN detected in model outputs! *****\n"
                f"  mean range: [{mean.min().item()}, {mean.max().item()}]\n"
                f"  var range:  [{var.min().item()}, {var.max().item()}]\n"
                f"  Sample mean[:2]: {mean[:2]}\n"
                f"  Sample var[:2]:  {var[:2]}"
            )

        # 2. Reconstruction loss
        genes = batch[0]  # the first item is the gene expression
        reconstruction_loss = self.model.loss_autoencoder(mean, genes, var)

        # Debug if reconstruction_loss is NaN
        if torch.isnan(reconstruction_loss):
            print("***** WARNING: reconstruction_loss is NaN! *****")

        # 3. Adversary losses
        adversary_drugs_loss = torch.tensor(0.0, device=self.device)
        adversary_covariates_loss = torch.tensor(0.0, device=self.device)

        if self.model.num_drugs > 0 and latent_basal is not None:
            # compute drug predictions
            adversary_drugs_predictions = self.model.adversary_drugs(latent_basal)
            if self.model.use_drugs_idx:
                # batch: (genes, drugs_idx, dosages, degs, *covariates)
                drugs_idx = batch[1]
                if drugs_idx.ndim == 1:
                    # single-drug => single-hot
                    batch_size = drugs_idx.size(0)
                    multi_hot_targets = torch.zeros(batch_size, self.model.num_drugs, device=drugs_idx.device)
                    multi_hot_targets[torch.arange(batch_size), drugs_idx] = 1.0
                    adversary_drugs_loss = self.model.loss_adversary_drugs(
                        adversary_drugs_predictions, multi_hot_targets
                    )
                else:
                    # multi-drug => build multi-hot
                    batch_size, combo_size = drugs_idx.shape
                    multi_hot_targets = torch.zeros(batch_size, self.model.num_drugs, device=drugs_idx.device)
                    for i in range(combo_size):
                        multi_hot_targets[torch.arange(batch_size), drugs_idx[:, i]] = 1.0
                    adversary_drugs_loss = self.model.loss_adversary_drugs(
                        adversary_drugs_predictions, multi_hot_targets
                    )
            else:
                # We have one-hot or numeric drug usage in batch[1]
                # Use BCE for presence/absence
                drugs = batch[1]
                adversary_drugs_loss = self.model.loss_adversary_drugs(
                    adversary_drugs_predictions, (drugs > 0).float()
                )

        # Covariates adversary
        if self.model.num_covariates[0] > 0 and latent_basal is not None:
            covariates = batch[4:] if self.model.use_drugs_idx else batch[3:]
            for i, adv in enumerate(self.model.adversary_covariates):
                pred_cov = adv(latent_basal)
                # each covariate is one-hot => .argmax(dim=1)
                adversary_covariates_loss += self.model.loss_adversary_covariates[i](
                    pred_cov, covariates[i].argmax(dim=1)
                )

        # 4. Decide whether to optimize adversaries or autoencoder/doser
        if (self.global_step % self.model.hparams["adversary_steps"]) == 0:
            # ---- Adversary update ----
            optimizer_adversaries.zero_grad()

            # (Optional) gradient penalty
            adv_drugs_grad_penalty = torch.tensor(0.0, device=self.device)
            adv_covs_grad_penalty = torch.tensor(0.0, device=self.device)

            if latent_basal is not None:
                def compute_gradient_penalty(out, x):
                    grads = torch.autograd.grad(out, x, create_graph=True)[0]
                    return (grads ** 2).mean()

                if self.model.num_drugs > 0:
                    adv_drugs_grad_penalty = compute_gradient_penalty(
                        adversary_drugs_predictions.sum(), latent_basal
                    )

                if self.model.num_covariates[0] > 0:
                    for i, adv in enumerate(self.model.adversary_covariates):
                        out = adv(latent_basal).sum()
                        adv_covs_grad_penalty += compute_gradient_penalty(out, latent_basal)

            penalty_scale = self.model.hparams["penalty_adversary"]
            loss_adversary_total = (
                adversary_drugs_loss
                + adversary_covariates_loss
                + penalty_scale * adv_drugs_grad_penalty
                + penalty_scale * adv_covs_grad_penalty
            )

            self.manual_backward(loss_adversary_total)
            optimizer_adversaries.step()

            # Debug prints
            #print("adversary_drugs_loss:", adversary_drugs_loss.item())
            #print("adversary_covariates_loss:", adversary_covariates_loss.item())

        else:
            # ---- Autoencoder & doser update ----
            optimizer_autoencoder.zero_grad()
            optimizer_dosers.zero_grad()

            # Weighted combination of reconstruction and adversary terms
            loss_ae = (
                reconstruction_loss
                - self.model.hparams["reg_adversary"] * adversary_drugs_loss
                - self.model.hparams.get("reg_adversary_cov", 1.0) * adversary_covariates_loss
            )

            self.manual_backward(loss_ae)
            self.clip_gradients(optimizer_autoencoder, gradient_clip_val=1, gradient_clip_algorithm="norm")
            self.clip_gradients(optimizer_dosers, gradient_clip_val=1, gradient_clip_algorithm="norm")
            optimizer_autoencoder.step()
            optimizer_dosers.step()

            # Debug prints
            #print("reconstruction_loss:", reconstruction_loss.item#())
            #print("adversary_drugs_loss:", adversary_drugs_loss.item())
            #print("adversary_covariates_loss:", adversary_covariates_loss.item())

        # Step LR schedulers at epoch end if you want
        if self.trainer.is_last_batch:
            for lr_s in self.lr_schedulers():
                lr_s.step()
        # Logging
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("adversary_drugs_loss", adversary_drugs_loss)
        self.log("adversary_covariates_loss", adversary_covariates_loss)

        return None



    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Minimal placeholder, you can fill in with your validation logic."""
        pass

    def on_validation_epoch_end(self):
        """Example R2 evaluation after each validation epoch."""
        # Make sure we don't accidentally train in these evaluations
        self.eval()
        result = evaluate_r2(
            self.model,
            self.trainer.datamodule.test_treated_dataset,
            self.trainer.datamodule.test_control_dataset.genes,
        )
        # E.g. result = [R2_mean, R2_mean_de, R2_var, R2_var_de]
        metric_names = ["R2_mean", "R2_mean_de", "R2_var", "R2_var_de"]
        self.log_dict(dict(zip(metric_names, result)), on_step=False, on_epoch=True)
        self.train()

    def on_train_end(self):
        """Example final evaluation once training completes."""
        self.eval()
        if self.config["training"].get("run_eval_r2", False):
            evaluation_stats = {}
            # Evaluate on train
            _train_eval = evaluate_r2(
                self.model,
                self.trainer.datamodule.train_treated_dataset,
                self.trainer.datamodule.train_control_dataset.genes,
            )
            # Evaluate on test
            _test_eval = evaluate_r2(
                self.model,
                self.trainer.datamodule.test_treated_dataset,
                self.trainer.datamodule.test_control_dataset.genes,
            )
            # Evaluate on OOD
            _ood_eval = evaluate_r2(
                self.model,
                self.trainer.datamodule.ood_treated_dataset,
                self.trainer.datamodule.ood_control_dataset.genes,
            )

            # Log or print final results
            # 'final_train_R2_mean', etc...
            self.log("final_train_R2_mean", _train_eval[0])
            self.log("final_test_R2_mean", _test_eval[0])
            self.log("final_ood_R2_mean", _ood_eval[0])
        self.train()

    def configure_optimizers(self):
        """Set up three optimizers + schedulers: autoencoder, adversaries, dosers."""
        has_covariates = self.model.num_covariates[0] > 0

        def get_params(module_, condition):
            return list(module_.parameters()) if condition else []

        # 1) Autoencoder
        #    encoder + decoder + drug_embedding_encoder + covariates_embeddings
        autoencoder_parameters = (
            get_params(self.model.encoder, True)
            + get_params(self.model.decoder, True)
            + get_params(self.model.drug_embedding_encoder, self.model.drug_embedding_encoder is not None)
        )
        for emb in getattr(self.model, "covariates_embeddings", []):
            autoencoder_parameters.extend(get_params(emb, has_covariates))

        optimizer_autoencoder = torch.optim.Adam(
            autoencoder_parameters,
            lr=self.model.hparams["autoencoder_lr"],
            weight_decay=self.model.hparams["autoencoder_wd"],
        )

        # 2) Adversaries
        adversaries_parameters = []
        # Drugs adversary
        if self.model.num_drugs > 0:
            adversaries_parameters += list(self.model.adversary_drugs.parameters())
        # Covariates adversary
        if has_covariates:
            for adv in self.model.adversary_covariates:
                adversaries_parameters.extend(list(adv.parameters()))

        optimizer_adversaries = torch.optim.Adam(
            adversaries_parameters,
            lr=self.model.hparams["adversary_lr"],
            weight_decay=self.model.hparams["adversary_wd"],
        )

        # 3) Dosers
        if hasattr(self.model, "dosers"):
            optimizer_dosers = torch.optim.Adam(
                self.model.dosers.parameters(),
                lr=self.model.hparams["dosers_lr"],
                weight_decay=self.model.hparams["dosers_wd"],
            )
        else:
            # Fallback if no 'dosers' in the model
            optimizer_dosers = torch.optim.Adam([], lr=0.0)

        # LR schedulers
        scheduler_autoencoder = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer_autoencoder,
                step_size=self.model.hparams["step_size_lr"],
                gamma=0.9,
            ),
            "name": "lr-autoencoder",
        }
        scheduler_adversaries = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer_adversaries,
                step_size=self.model.hparams["step_size_lr"],
                gamma=0.9,
            ),
            "name": "lr-adversaries",
        }
        scheduler_dosers = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer_dosers,
                step_size=self.model.hparams["step_size_lr"],
                gamma=0.9,
            ),
            "name": "lr-dosers",
        }

        # Return the 3 optimizers + 3 schedulers
        return (
            {"optimizer": optimizer_autoencoder, "lr_scheduler": scheduler_autoencoder},
            {"optimizer": optimizer_adversaries, "lr_scheduler": scheduler_adversaries},
            {"optimizer": optimizer_dosers, "lr_scheduler": scheduler_dosers},
        )


