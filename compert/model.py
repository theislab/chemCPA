# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
from typing import Union

import numpy as np
import torch


def _move_inputs(*inputs, device="cuda"):
    def mv_input(x):
        if type(x) == list:
            return [mv_input(y) for y in x]
        else:
            return x.to(device) if x is not None else None

    return [mv_input(x) for x in inputs]


class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        dim = yhat.size(1) // 2
        # means of the negative binomial (has to be positive support)
        mu = yhat[:, :dim]
        # inverse dispersion parameter (has to be positive support)
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y + 1.0)
            - torch.lgamma(y + theta + eps)
        )
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (
            y * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
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
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, device, nonlin="sigm"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        assert nonlin in ("sigm", "logsigm", None)
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device), requires_grad=True
        )

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
    Our main module, the ComPert autoencoder
    """

    num_drugs: int  # number of unique drugs in the dataset, including control
    use_drugs_idx: bool  # whether to except drugs coded by index or by OneHotEncoding

    def __init__(
        self,
        num_genes: int,
        num_drugs: int,
        num_covariates: int,
        device="cpu",
        seed=0,
        patience=5,
        loss_ae="gauss",
        doser_type="logsigm",
        decoder_activation="linear",
        hparams="",
        drug_embeddings: Union[None, torch.nn.Embedding] = None,
        use_drugs_idx=False,
    ):
        super(ComPert, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0
        self.use_drugs_idx = use_drugs_idx

        # set hyperparameters
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(seed, hparams)

        # set models
        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]]
        )

        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
        )
        if self.num_drugs > 0:
            self.adversary_drugs = MLP(
                [self.hparams["dim"]]
                + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                + [self.num_drugs]
            )
            if drug_embeddings is None:
                self.drug_embeddings = (
                    torch.nn.Embedding(  # TODO: Check to merge on gnn model
                        self.num_drugs, self.hparams["dim"]
                    )
                )
                embedding_requires_grad = True
            else:
                self.drug_embeddings = drug_embeddings
                embedding_requires_grad = False

            self.drug_embedding_encoder = MLP(
                [self.drug_embeddings.embedding_dim, self.hparams["dim"]],
                last_layer_act="linear",
            )

            if use_drugs_idx:
                # there will only ever be a single drug, so no binary cross entropy needed
                # careful: when the model is finetuned later with One-hot encodings, we'll have to
                # retrained the adversary classifiers.
                self.loss_adversary_drugs = torch.nn.CrossEntropyLoss()
            else:
                self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
            # set dosers
            if doser_type == "mlp":
                self.dosers = torch.nn.ModuleList()
                for _ in range(self.num_drugs):
                    self.dosers.append(
                        MLP(
                            [1]
                            + [self.hparams["dosers_width"]]
                            * self.hparams["dosers_depth"]
                            + [1],
                            batch_norm=False,
                        )
                    )
            else:
                self.dosers = GeneralizedSigmoid(
                    self.num_drugs, self.device, nonlin=doser_type
                )
            self.doser_type = doser_type

        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = []
            self.loss_adversary_covariates = []
            self.covariates_embeddings = (
                []
            )  # TODO: Continue with checking that dict assignment is possible via covaraites names and if dict are possible to use in optimisation
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams["dim"]]
                        + [self.hparams["adversary_width"]]
                        * self.hparams["adversary_depth"]
                        + [num_covariate]
                    )
                )
                self.loss_adversary_covariates.append(torch.nn.CrossEntropyLoss())
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams["dim"])
                )

        # losses
        if self.loss_ae == "nb":
            self.loss_autoencoder = NBLoss()
        else:
            self.loss_autoencoder = GaussianLoss()

        self.iteration = 0

        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        has_covariates = self.num_covariates[0] > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs and embedding_requires_grad)
            + get_params(self.drug_embedding_encoder, True)
        )
        for emb in self.covariates_embeddings:
            _parameters.extend(get_params(emb, has_covariates))

        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )

        _parameters = get_params(self.adversary_drugs, has_drugs)
        for adv in self.adversary_covariates:
            _parameters.extend(get_params(adv, has_covariates))

        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"],
        )

        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["dosers_lr"],
                weight_decay=self.hparams["dosers_wd"],
            )

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"]
        )

        if has_drugs:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.hparams["step_size_lr"]
            )

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
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
            "autoencoder_width": 512
            if default
            else int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else int(np.random.choice([3, 4, 5])),
            "adversary_width": 128
            if default
            else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3
            if default
            else float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3
            if default
            else float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6
            if default
            else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128
            if default
            else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def compute_drug_embeddings_(self, drugs=None, drugs_idx=None, dosages=None):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.

        Returns a tensor of shape [batch_size, drug_embedding_dimension]
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

        drugs, drugs_idx, dosages = _move_inputs(
            drugs, drugs_idx, dosages, device=self.device
        )

        latent_drugs = self.drug_embeddings.weight

        if drugs_idx is not None:
            assert drugs_idx.shape == dosages.shape and len(drugs_idx.shape) == 1
            # results in a tensor of shape [batchsize, drug_embedding_dimension]
            latent_drugs = latent_drugs[drugs_idx]

        if self.doser_type == "mlp":
            if drugs_idx is None:
                doses = []
                for d in range(drugs.size(1)):
                    this_drug = drugs[:, d].view(-1, 1)
                    doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
                scaled_dosages = torch.cat(doses, 1)
            else:
                scaled_dosages = []
                for idx, dosage in zip(drugs_idx, dosages):
                    scaled_dosages.append(
                        self.dosers[idx](dosage.unsqueeze(0)).sigmoid()
                    )
                scaled_dosages = torch.cat(scaled_dosages, 0)
        else:
            if drugs_idx is None:
                scaled_dosages = self.dosers(drugs)
            else:
                scaled_dosages = self.dosers(dosages, drugs_idx)

        if drugs_idx is None:
            return scaled_dosages @ latent_drugs
        else:
            return torch.einsum("b,be->be", [scaled_dosages, latent_drugs])

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
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)
        genes, drugs, drugs_idx, dosages, covariates = _move_inputs(
            genes, drugs, drugs_idx, dosages, covariates, device=self.device
        )

        latent_basal = self.encoder(genes)

        latent_treated = latent_basal

        if self.num_drugs > 0:
            drug_embedding = self.compute_drug_embeddings_(
                drugs=drugs, drugs_idx=drugs_idx, dosages=dosages
            )
            latent_treated = latent_treated + self.drug_embedding_encoder(
                drug_embedding
            )
        if self.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].argmax(1)
                latent_treated = latent_treated + emb_cov(cov_idx)

        gene_reconstructions = self.decoder(latent_treated)

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        gene_reconstructions[:, dim:] = (
            gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)
        )

        if self.loss_ae == "nb":
            gene_reconstructions[:, :dim] = (
                gene_reconstructions[:, :dim].exp().add(1).log().add(1e-4)
            )
            # gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            # gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-6, max=1e6)

        if return_latent_basal:
            return gene_reconstructions, latent_basal

        return gene_reconstructions

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        if self.num_drugs > 0:
            self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def update(self, genes, drugs=None, drugs_idx=None, dosages=None, covariates=None):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

        gene_reconstructions, latent_basal = self.predict(
            genes=genes,
            drugs=drugs,
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
            return_latent_basal=True,
        )
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            if self.use_drugs_idx:
                adversary_drugs_loss = self.loss_adversary_drugs(
                    adversary_drugs_predictions, drugs_idx
                )
            else:
                adversary_drugs_loss = self.loss_adversary_drugs(
                    adversary_drugs_predictions, drugs.gt(0).float()
                )

        adversary_covariates_loss = torch.tensor([0.0], device=self.device)
        if self.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.adversary_covariates):
                adv = adv.to(self.device)
                adversary_covariate_predictions.append(adv(latent_basal))
                adversary_covariates_loss += self.loss_adversary_covariates[i](
                    adversary_covariate_predictions[-1], covariates[i].argmax(1)
                )

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.tensor([0.0], device=self.device)
        adversary_covariates_penalty = torch.tensor([0.0], device=self.device)

        if self.iteration % self.hparams["adversary_steps"]:

            def compute_gradients(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            if self.num_drugs > 0:
                adversary_drugs_penalty = compute_gradients(
                    adversary_drugs_predictions.sum(), latent_basal
                )

            if self.num_covariates[0] > 0:
                adversary_covariates_penalty = torch.tensor([0.0], device=self.device)
                for pred in adversary_covariate_predictions:
                    adversary_covariates_penalty += compute_gradients(
                        pred.sum(), latent_basal
                    )  # TODO: Adding up tensor sum, is that right?

            self.optimizer_adversaries.zero_grad()
            (
                adversary_drugs_loss
                + self.hparams["penalty_adversary"] * adversary_drugs_penalty
                + adversary_covariates_loss
                + self.hparams["penalty_adversary"] * adversary_covariates_penalty
            ).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            if self.num_drugs > 0:
                self.optimizer_dosers.zero_grad()
            (
                reconstruction_loss
                - self.hparams["reg_adversary"] * adversary_drugs_loss
                - self.hparams["reg_adversary"] * adversary_covariates_loss
            ).backward()
            self.optimizer_autoencoder.step()
            if self.num_drugs > 0:
                self.optimizer_dosers.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_covariates": adversary_covariates_penalty.item(),
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for ComPert
        """

        return self.set_hparams_(self, 0, "")
