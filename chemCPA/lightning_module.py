import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.demos import Transformer

from chemCPA.data.data import load_dataset_splits
from chemCPA.embedding import get_chemical_representation
from chemCPA.model import ComPert
from chemCPA.train import evaluate_disentanglement, evaluate_r2


class ChemCPA(L.LightningModule):
    def __init__(self, config, dataset_config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config

        additional_params = self.config["model"]["additional_params"]
        hparams = self.config["model"]["hparams"]

        self.drug_embeddings = get_chemical_representation(
            smiles=dataset_config["canon_smiles_unique_sorted"],
            embedding_model=self.config["model"]["embedding"],
            data_path=self.config["model"]["embedding"]["datapath"],
        )

        self.model = ComPert(
            dataset_config["num_genes"],
            dataset_config["num_drugs"],
            dataset_config["num_covariates"],
            **additional_params,
            hparams=hparams,
            drug_embeddings=self.drug_embeddings,
            use_drugs_idx=dataset_config["use_drugs_idx"],
            append_layer_width=None,
            device="cuda",
        )

        self.save_hyperparameters()

    def forward(self, batch, return_latent_basal=False):
        if self.model.use_drugs_idx:
            # batch has (genes, drugs_idx, dosages, degs, *covariates)
            genes = batch[0]
            drugs_idx = batch[1]
            dosages = batch[2]
            degs = batch[3]
            covariates = batch[4:]
            gene_reconstructions, cell_drug_embedding, latents = self.model.predict(
                genes=genes,
                drugs=None,
                drugs_idx=drugs_idx,
                dosages=dosages,
                covariates=covariates,
                return_latent_basal=True,
            )
        else:
            # batch has (genes, drugs, degs, *covariates)
            genes = batch[0]
            drugs = batch[1]
            degs = batch[2]
            covariates = batch[3:]
            gene_reconstructions, cell_drug_embedding, latents = self.model.predict(
                genes=genes,
                drugs=drugs,
                drugs_idx=None,
                dosages=None,
                covariates=covariates,
                return_latent_basal=True,
            )

        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = gene_reconstructions[:, dim:]
        if return_latent_basal:
            return (mean, var), latents
        return mean, var

    def training_step(self, batch, batch_idx):
        """Example training_step that handles both idx-based and multi-hot drugs."""

        # Predefine everything so we don't get "UnboundLocalError".
        genes, drugs_idx, dosages, drugs, degs, covariates = None, None, None, None, None, None

        # Unpack the batch based on use_drugs_idx
        if self.model.use_drugs_idx:
            # batch = (genes, drugs_idx, dosages, degs, *covariates)
            genes = batch[0]
            drugs_idx = batch[1]
            dosages = batch[2]
            degs = batch[3]
            covariates = batch[4:]
        else:
            # batch = (genes, drugs, degs, *covariates)
            genes = batch[0]
            drugs = batch[1]
            degs = batch[2]
            covariates = batch[3:]

        # Forward pass
        (mean, var), latents = self(batch, return_latent_basal=True)
        latent_basal = latents[0]

        # Reconstruction loss
        reconstruction_loss = self.model.loss_autoencoder(
            input=mean,
            target=genes,
            var=var
        )

        # Adversary (drugs)
        adversary_drugs_predictions = self.model.adversary_drugs(latent_basal)
        if self.model.use_drugs_idx:
            # CrossEntropy if we have integer drug indices
            adversary_drugs_loss = self.model.loss_adversary_drugs(
                adversary_drugs_predictions,
                drugs_idx
            )
        else:
            # BCE if we have multi-hot drug dosages
            # (dosage > 0) means "drug is present"
            adversary_drugs_loss = self.model.loss_adversary_drugs(
                adversary_drugs_predictions,
                (drugs > 0).float()
            )

        # Adversary (covariates)
        adversary_covariates_loss = torch.tensor([0.0], device=self.device)
        if self.model.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.model.adversary_covariates):
                adv = adv.to(self.model.device)
                prediction = adv(latent_basal)
                adversary_covariate_predictions.append(prediction)
                adversary_covariates_loss += self.model.loss_adversary_covariates[i](
                    prediction,
                    covariates[i].argmax(dim=1)
                )

        # Two placeholders for gradient penalties
        adv_drugs_grad_penalty = torch.tensor([0.0], device=self.device)
        adv_covs_grad_penalty = torch.tensor([0.0], device=self.device)

        # Retrieve optimizers in the order returned by configure_optimizers()
        optimizers = self.optimizers()

        # Adversary steps
        if (self.trainer.global_step % self.model.hparams["adversary_steps"]) == 0:
            optimizer_adversaries = optimizers[1]
            optimizer_adversaries.zero_grad()

            def compute_gradient_penalty(output, x_input):
                grads = torch.autograd.grad(output, x_input, create_graph=True)[0]
                return grads.pow(2).mean()

            # Compute penalty for drug adversary
            adv_drugs_grad_penalty = compute_gradient_penalty(
                adversary_drugs_predictions.sum(), 
                latent_basal
            )

            # Compute penalty for covariates adversary
            if self.model.num_covariates[0] > 0:
                for pred in adversary_covariate_predictions:
                    adv_covs_grad_penalty += compute_gradient_penalty(pred.sum(), latent_basal)

            loss = (
                adversary_drugs_loss
                + self.model.hparams["penalty_adversary"] * adv_drugs_grad_penalty
                + adversary_covariates_loss
                + self.model.hparams["penalty_adversary"] * adv_covs_grad_penalty
            )
            self.manual_backward(loss)
            self.clip_gradients(optimizer_adversaries, gradient_clip_val=1, gradient_clip_algorithm="norm")
            optimizer_adversaries.step()

            self.log("penalized_adversary_loss", loss)
        else:
            # Autoencoder & dosers step
            optimizer_autoencoder = optimizers[0]
            optimizer_dosers = optimizers[2]
            optimizer_autoencoder.zero_grad()
            optimizer_dosers.zero_grad()

            loss = (
                reconstruction_loss
                - self.model.hparams["reg_adversary"] * adversary_drugs_loss
                - self.model.hparams["reg_adversary_cov"] * adversary_covariates_loss
            )
            self.manual_backward(loss)
            self.clip_gradients(optimizer_autoencoder, gradient_clip_val=1, gradient_clip_algorithm="norm")
            self.clip_gradients(optimizer_dosers, gradient_clip_val=1, gradient_clip_algorithm="norm")
            optimizer_autoencoder.step()
            optimizer_dosers.step()

            self.log("penalized_reconstruction_loss", loss)

        # Optionally step LR schedulers at epoch-end or after last batch
        # (just an example with step after last batch)
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            for lr_s in self.lr_schedulers():
                lr_s.step()

        # Logging
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("adversary_drugs_loss", adversary_drugs_loss)
        self.log("adversary_covariates_loss", adversary_covariates_loss)

        return None


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def on_validation_epoch_end(self):
        assert not self.training
        result = evaluate_r2(
            self.model,
            self.trainer.datamodule.test_treated_dataset,
            self.trainer.datamodule.test_control_dataset.genes,
        )
        # _result = evaluate_r2(
        #     self.model,
        #     self.trainer.datamodule.test_treated_dataset,
        #     self.trainer.datamodule.test_control_dataset.genes,
        # )
        # assert np.allclose(result, _result)

        evaluation_stats = dict(zip(["R2_mean", "R2_mean_de", "R2_var", "R2_var_de"], result))
        self.log_dict(evaluation_stats, on_step=False, on_epoch=True)

        # self.trainer.save_checkpoint("test.ckpt")
        # model_2 = ChemCPA.load_from_checkpoint(checkpoint_path="test.ckpt")
        # sd_a = self.state_dict()
        # sd_b = model_2.state_dict()
        # for key in sd_a:
        #     if not torch.equal(sd_a[key], sd_b[key]):
        #         print(f"Difference found in layer: {key}")
        # model_2.eval()
        # result2 = evaluate_r2(
        #     model_2.model,
        #     self.trainer.datamodule.test_treated_dataset,
        #     self.trainer.datamodule.test_control_dataset.genes,
        # )
        # # _result2 = evaluate_r2(
        # #     model_2.model,
        # #     self.trainer.datamodule.test_treated_dataset,
        # #     self.trainer.datamodule.test_control_dataset.genes,
        # # )
        # # assert np.allclose(result2, _result2)
        # # check that result and result2 are close
        # if not np.allclose(result, result2):
        #     print(self.trainer.current_epoch)
        #     print(result)
        #     print(result2)

        # # for batch in self.trainer.datamodule.train_dataloader():
        # #     genes, drugs_idx, covariates = batch[0], batch[1], batch[4:]

        # #     (mean, var), latents = self(batch, return_latent_basal=True)
        # #     (mean2, var2), latents2 = model_2(batch, return_latent_basal=True)
        # #     break

    def on_train_end(self):
        self.eval()
        assert not self.training
        if self.config["training"]["run_eval_disentangle"]:
            dl_test = self.trainer.datamodule.val_dataloader()["test"]
            dataset_test = dl_test.dataset
            drug_names, drug_counts = np.unique(dataset_test.drugs_names, return_counts=True)
            disent_scores = evaluate_disentanglement(self.model, dataloader=dl_test)
            stats_disent_pert = disent_scores[0]
            # optimal score == always predicting the most common drug
            optimal_disent_score = max(drug_counts) / len(dataset_test)
            stats_disent_cov = disent_scores[1:]

            evaluation_stats = {
                "perturbation disentanglement": stats_disent_pert,
                "optimal for perturbations": optimal_disent_score,
            }
            for i, cov_type in enumerate(dataset_test.covariate_names.keys()):
                if dataset_test.num_covariates[i] > 0:
                    stats = {
                        f"optimal for {cov_type}": dataset_test.covariates[i].mean(axis=0).max().item(),
                        f"{cov_type} disentanglement": stats_disent_cov[i],
                    }
                    evaluation_stats.update(stats)
            # evaluation_stats = {
            #     "perturbation disentanglement": stats_disent_pert,
            #     "optimal for perturbations": optimal_disent_score,
            #     "covariate disentanglement": stats_disent_cov,
            #     "optimal for covariates": (
            #         [
            #             max(cov.mean(axis=0)).item() for cov in dataset_test.covariates
            #         ]  # mean over OHE embedding of covariates
            #         if dataset_test.num_covariates[0] > 0
            #         else None
            #     ),
            # }

            self.logger.experiment.log(evaluation_stats, step=self.trainer.global_step)

        if self.config["training"]["run_eval_r2"]:
            evaluation_stats = {}
            _result = evaluate_r2(
                self.model,
                self.trainer.datamodule.train_treated_dataset,
                self.trainer.datamodule.train_control_dataset.genes,
            )
            _metrics = ["final_train_R2_mean", "final_train_R2_mean_de", "final_train_R2_var", "final_train_R2_var_de"]
            evaluation_stats.update(dict(zip(_metrics, _result)))
            _result = evaluate_r2(
                self.model,
                self.trainer.datamodule.test_treated_dataset,
                self.trainer.datamodule.test_control_dataset.genes,
            )
            _metrics = ["final_test_R2_mean", "final_test_R2_mean_de", "final_test_R2_var", "final_test_R2_var_de"]
            evaluation_stats.update(dict(zip(_metrics, _result)))
            _result = evaluate_r2(
                self.model,
                self.trainer.datamodule.ood_treated_dataset,
                self.trainer.datamodule.ood_control_dataset.genes,
            )
            _metrics = ["final_ood_R2_mean", "final_ood_R2_mean_de", "final_ood_R2_var", "final_ood_R2_var_de"]
            evaluation_stats.update(dict(zip(_metrics, _result)))

            self.logger.experiment.log(evaluation_stats, step=self.trainer.global_step)

    def configure_optimizers(self):
        has_covariates = self.model.num_covariates[0] > 0

        get_params = lambda model, cond: list(model.parameters()) if cond else []

        autoencoder_parameters = (
            get_params(self.model.encoder, True)
            + get_params(self.model.decoder, True)
            + get_params(self.model.drug_embedding_encoder, True)
        )
        for emb in self.model.covariates_embeddings:
            autoencoder_parameters.extend(get_params(emb, has_covariates))
        optimizer_autoencoder = torch.optim.Adam(
            autoencoder_parameters,
            lr=self.model.hparams["autoencoder_lr"],
            weight_decay=self.model.hparams["autoencoder_wd"],
        )

        adversaries_parameters = get_params(self.model.adversary_drugs, True)
        for adv in self.model.adversary_covariates:
            adversaries_parameters.extend(get_params(adv, has_covariates))
        optimizer_adversaries = torch.optim.Adam(
            adversaries_parameters,
            lr=self.model.hparams["adversary_lr"],
            weight_decay=self.model.hparams["adversary_wd"],
        )

        optimizer_dosers = torch.optim.Adam(
            self.model.dosers.parameters(),
            lr=self.model.hparams["dosers_lr"],
            weight_decay=self.model.hparams["dosers_wd"],
        )

        # learning rate schedulers
        scheduler_autoencoder = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer_autoencoder,
                step_size=self.model.hparams["step_size_lr"],
                gamma=0.9,
            ),
            "name": "lr-autoencoder",
        }

        scheduler_adversary = {
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
        return (
            {"optimizer": optimizer_autoencoder, "lr_scheduler": scheduler_autoencoder},
            {"optimizer": optimizer_adversaries, "lr_scheduler": scheduler_adversary},
            {"optimizer": optimizer_dosers, "lr_scheduler": scheduler_dosers},
        )
