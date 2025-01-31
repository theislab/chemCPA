import json
import logging
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _move_inputs(*inputs, device="cuda"):
    def mv_input(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return [mv_input(y) for y in x]

    return [mv_input(x) for x in inputs]


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive.
        """
        dim = yhat.size(1) // 2
        mu = yhat[:, :dim]
        theta = yhat[:, dim:]  # inverse dispersion

        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - torch.lgamma(y + theta + eps)
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (
            y * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = t1 + t2
        final = _nan2inf(final)
        return torch.mean(final)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, gamma=3, reduction="mean") -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        from torchvision.ops import focal_loss

        loss = focal_loss.sigmoid_focal_loss(
            inputs,
            target,
            reduction=self.reduction,
            gamma=self.gamma,
            alpha=self.alpha,
        )
        return loss


class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns.
    """

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, yhat, y):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]

        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))
        return (term1 + term2).mean()


class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.

    If `append_layer_width` is provided, we add an extra "henc" or "hdec" layer:
      - append_layer_position="first" => "henc" layer mapping input_dim -> original_sizes[0]
      - append_layer_position="last"  => "hdec" layer mapping final_size -> append_layer_width
    This matches the approach of adding non-linear layers to shift from the new
    gene dimension to the pretraining dimension (or vice versa) for fine-tuning.
    """

    def __init__(
        self,
        sizes,
        batch_norm=True,
        last_layer_act="linear",
        append_layer_width=None,
        append_layer_position=None,
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU(),
            ]
        # Remove trailing "None" or "ReLU"
        layers = [l for l in layers if l is not None][:-1]

        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        # Build the MLP as an OrderedDict, optionally with an appended layer
        if append_layer_width:
            assert append_layer_position in ("first", "last")
            layers_dict = OrderedDict()

            if append_layer_position == "first":
                # This is the "henc" layer: new_dim -> original sizes[0]
                layers_dict["henc_linear"] = torch.nn.Linear(append_layer_width, sizes[0])
                layers_dict["henc_bn"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["henc_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                # Normal MLP first
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
                # Then the "hdec" layer: sizes[-1] -> new_dim
                layers_dict["hdec_bn"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["hdec_relu"] = torch.nn.ReLU()
                layers_dict["hdec_linear"] = torch.nn.Linear(sizes[-1], append_layer_width)
        else:
            # No appended layer
            layers_dict = OrderedDict({str(i): module for i, module in enumerate(layers)})

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        x = self.network(x)
        if self.activation == "ReLU":
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return x


class GeneralizedSigmoid(torch.nn.Module):
    """
    Optionally applies a (log-)sigmoid transform to dosage inputs.
    """

    def __init__(self, dim, device, nonlin="sigm"):
        super(GeneralizedSigmoid, self).__init__()
        assert nonlin in ("sigm", "logsigm", None)
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(torch.ones(1, dim, device=device), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, dim, device=device), requires_grad=True)

    def forward(self, x, idx=None):
        if self.nonlin == "logsigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (torch.log1p(x) * beta + bias).sigmoid() - c0

        elif self.nonlin == "sigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (x * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (x * beta + bias).sigmoid() - c0

        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class ComPert(torch.nn.Module):
    """
    Main ComPert autoencoder module with drug/covariate adversaries.

    If `append_layer_width` is given, we add an extra "henc" layer at the start
    of the encoder and an extra "hdec" layer at the end of the decoder. This is
    how we handle dimension mismatch when fine-tuning on a new gene set.
    """

    num_drugs: int
    use_drugs_idx: bool

    def __init__(
        self,
        num_genes: int,
        num_drugs: int,
        num_covariates: int,
        device="cpu",
        seed=0,
        patience=5,
        doser_type="logsigm",
        decoder_activation="linear",
        hparams="",
        drug_embeddings: Union[None, torch.nn.Embedding] = None,
        use_drugs_idx=False,
        append_layer_width=None,
        multi_task: bool = False,
        enable_cpa_mode=False,
    ):
        super(ComPert, self).__init__()
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.device = device
        self.seed = seed
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0
        self.use_drugs_idx = use_drugs_idx
        self.enable_cpa_mode = enable_cpa_mode

        # set hyperparameters
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(seed, hparams)

        # Autoencoder
        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]],
            append_layer_width=append_layer_width,      # henc
            append_layer_position="first",
        )
        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
            append_layer_width=(2 * append_layer_width) if append_layer_width else None,  # hdec
            append_layer_position="last",
        )
        if append_layer_width:
            # We keep track that we are effectively working with 'append_layer_width' genes
            # after the final "hdec" for our predictions. This can help in indexing etc.
            self.num_genes = append_layer_width

        # Adversaries
        if self.num_drugs > 0:
            self.adversary_drugs = MLP(
                [self.hparams["dim"]]
                + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                + [self.num_drugs]
            )
            if drug_embeddings is None:
                self.drug_embeddings = torch.nn.Embedding(self.num_drugs, self.hparams["dim"])
            else:
                self.drug_embeddings = drug_embeddings

            if self.enable_cpa_mode:
                self.drug_embedding_encoder = None
            else:
                self.drug_embedding_encoder = MLP(
                    [self.drug_embeddings.embedding_dim]
                    + [self.hparams["embedding_encoder_width"]] * self.hparams["embedding_encoder_depth"]
                    + [self.hparams["dim"]],
                    last_layer_act="linear",
                )

            self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()

            # Dosers
            assert doser_type in ("mlp", "sigm", "logsigm", "amortized", None)
            if doser_type == "mlp":
                self.dosers = torch.nn.ModuleList()
                for _ in range(self.num_drugs):
                    self.dosers.append(
                        MLP(
                            [1] + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"] + [1],
                            batch_norm=False,
                        )
                    )
            elif doser_type == "amortized":
                assert use_drugs_idx, "Amortized doser not implemented for `use_drugs_idx=False`"
                self.dosers = MLP(
                    [self.drug_embeddings.embedding_dim + 1]
                    + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                    + [1],
                )
            else:
                # 'sigm' or 'logsigm'
                self.dosers = GeneralizedSigmoid(self.num_drugs, self.device, nonlin=doser_type)
            self.doser_type = doser_type

        # Covariates
        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = nn.ModuleList()
            self.loss_adversary_covariates = nn.ModuleList()
            self.covariates_embeddings = nn.ModuleList()
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams["dim"]]
                        + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                        + [num_covariate]
                    )
                )
                self.loss_adversary_covariates.append(torch.nn.CrossEntropyLoss())
                self.covariates_embeddings.append(torch.nn.Embedding(num_covariate, self.hparams["dim"]))

        self.loss_autoencoder = torch.nn.GaussianNLLLoss()
        self.iteration = 0
        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) user overrides in JSON/dict `hparams`.
        """
        default = seed == 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            "dim": 256 if default else int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
        }
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)
        return self.hparams

    def compute_drug_embeddings_(self, drugs=None, drugs_idx=None, dosages=None):
        """
        Compute sum of drug embeddings, each scaled by a dose-response curve.
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None), (
            "Either `drugs` or (`drugs_idx` and `dosages`) must be provided."
        )

        drugs, drugs_idx, dosages = _move_inputs(drugs, drugs_idx, dosages, device=self.device)
        all_embeddings = self.drug_embeddings.weight  # shape [num_drugs, embedding_dim]

        if drugs_idx is not None:
            # Ensure 2D
            if drugs_idx.ndim == 1:
                drugs_idx = drugs_idx.unsqueeze(1)
                dosages = dosages.unsqueeze(1)
            batch_size, combo_drugs = drugs_idx.shape

            gathered_embeddings = all_embeddings[drugs_idx]  # [batch_size, combo_drugs, emb_dim]

            # scaled dosages
            if self.doser_type == "mlp":
                scaled_dosages_list = []
                for i in range(combo_drugs):
                    idx_i = drugs_idx[:, i]
                    dose_i = dosages[:, i].unsqueeze(1)
                    scaled_i = []
                    for b in range(batch_size):
                        d_idx = idx_i[b].item()
                        scaled_i.append(self.dosers[d_idx](dose_i[b]).sigmoid())
                    scaled_i = torch.cat(scaled_i, dim=0)
                    scaled_dosages_list.append(scaled_i.unsqueeze(1))
                scaled_dosages = torch.cat(scaled_dosages_list, dim=1)

            elif self.doser_type == "amortized":
                scaled_list = []
                for i in range(combo_drugs):
                    emb_i = gathered_embeddings[:, i, :]
                    dose_i = dosages[:, i].unsqueeze(-1)
                    cat_i = torch.cat([emb_i, dose_i], dim=1)
                    scaled_i = self.dosers(cat_i).sigmoid()
                    scaled_list.append(scaled_i)
                scaled_dosages = torch.stack(scaled_list, dim=1).squeeze(-1)

            elif self.doser_type in ("sigm", "logsigm"):
                scaled_list = []
                for i in range(combo_drugs):
                    dose_i = dosages[:, i]
                    drug_i = drugs_idx[:, i]
                    scaled_list.append(self.dosers(dose_i, drug_i).unsqueeze(1))
                scaled_dosages = torch.cat(scaled_list, dim=1)
            else:
                scaled_dosages = dosages

            # transform each embedding if needed
            if not self.enable_cpa_mode and self.drug_embedding_encoder is not None:
                transformed_list = []
                for i in range(combo_drugs):
                    emb_i = self.drug_embedding_encoder(gathered_embeddings[:, i, :])
                    transformed_list.append(emb_i.unsqueeze(1))
                transformed = torch.cat(transformed_list, dim=1)
            else:
                transformed = gathered_embeddings

            scaled_dosages_expanded = scaled_dosages.unsqueeze(-1)
            scaled_embeddings = transformed * scaled_dosages_expanded
            combo_embedding = scaled_embeddings.sum(dim=1)
            return combo_embedding
        else:
            # (drugs) => shape [batch_size, num_drugs]
            if self.doser_type == "mlp":
                scaled_list = []
                for d in range(self.num_drugs):
                    dose_d = drugs[:, d].unsqueeze(1)
                    scaled_d = self.dosers[d](dose_d).sigmoid()
                    scaled_list.append(scaled_d)
                scaled_dosages = torch.cat(scaled_list, dim=1)
            elif self.doser_type == "amortized":
                scaled_dosages = self.dosers(drugs).sigmoid()
            elif self.doser_type in ("sigm", "logsigm"):
                scaled_dosages = self.dosers(drugs)
            else:
                scaled_dosages = drugs

            if not self.enable_cpa_mode and self.drug_embedding_encoder is not None:
                transformed_embeddings = self.drug_embedding_encoder(all_embeddings)
            else:
                transformed_embeddings = all_embeddings

            drug_combo_emb = scaled_dosages @ transformed_embeddings
            return drug_combo_emb

    def predict(
        self,
        genes,
        drugs=None,
        drugs_idx=None,
        dosages=None,
        covariates=None,
        return_latent_basal=False,
    ):
        """
        Predict how gene expression in `genes` changes when treated with `drugs`.
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)
        genes, drugs, drugs_idx, dosages, covariates = _move_inputs(
            genes, drugs, drugs_idx, dosages, covariates, device=self.device
        )
        latent_basal = self.encoder(genes)
        latent_treated = latent_basal

        if self.num_drugs > 0:
            drug_embedding = self.compute_drug_embeddings_(drugs=drugs, drugs_idx=drugs_idx, dosages=dosages)
            latent_treated = latent_treated + drug_embedding

        if self.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].argmax(1)
                latent_treated = latent_treated + emb_cov(cov_idx)

        # Construct cell_drug_embedding for e.g. multi-task or logging
        if self.num_covariates[0] > 0 and self.num_drugs > 0:
            cell_drug_embedding = torch.cat([emb_cov(cov_idx), drug_embedding], dim=1)
        elif self.num_drugs > 0:
            cell_drug_embedding = drug_embedding
        else:
            cell_drug_embedding = torch.zeros_like(latent_basal)

        gene_reconstructions = self.decoder(latent_treated)
        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = F.softplus(gene_reconstructions[:, dim:])
        normalized_reconstructions = torch.cat([mean, var], dim=1)

        if return_latent_basal:
            return normalized_reconstructions, cell_drug_embedding, (latent_basal, drug_embedding, latent_treated)
        return normalized_reconstructions, cell_drug_embedding

    def early_stopping(self, score):
        if score is None:
            logging.warning("Early stopping score was None!")
        elif score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1
        return self.patience_trials > self.patience

    def update(
        self,
        genes,
        drugs=None,
        drugs_idx=None,
        dosages=None,
        degs=None,
        covariates=None,
    ):
        """
        (Optional) if you manually call a training step here; typically unused under Lightning.
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

        # ---- Forward pass (with debugging) ----
        gene_reconstructions, cell_drug_embedding, (latent_basal, drug_embedding, latent_treated) = self.predict(
            genes=genes,
            drugs=drugs,
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
            return_latent_basal=True,
        )
        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = gene_reconstructions[:, dim:]

        # Debug check for NaNs
        if torch.isnan(mean).any() or torch.isnan(var).any():
            print(
                f"NaN detected in mean/var:\n"
                f"  mean range [{mean.min().item()}, {mean.max().item()}]\n"
                f"  var range  [{var.min().item()}, {var.max().item()}]\n"
                f"  Some sample values:\n"
                f"    mean[:5] = {mean[:5]}\n"
                f"    var[:5]  = {var[:5]}"
            )

        # ---- Reconstruction loss ----
        reconstruction_loss = self.loss_autoencoder(input=mean, target=genes, var=var)

        # ---- Drug adversary loss (if used) ----
        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            # ...BCEWithLogitsLoss if multi-label...

        # ---- Covariates adversary loss (if used) ----
        adversary_covariates_loss = torch.tensor([0.0], device=self.device)
        if self.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.adversary_covariates):
                adv = adv.to(self.device)
                pred = adv(latent_basal)
                adversary_covariate_predictions.append(pred)
                adversary_covariates_loss += self.loss_adversary_covariates[i](
                    pred,
                    covariates[i].argmax(1),
                )

        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
        }

    @classmethod
    def defaults(cls):
        return cls.set_hparams_(cls, 0, "")

