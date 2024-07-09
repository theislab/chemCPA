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
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - torch.lgamma(y + theta + eps)
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (y * (torch.log(theta + eps) - torch.log(mu + eps)))
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, gamma=3, reduction="mean") -> None:
        """Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .

        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        """Compute the FocalLoss

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        from torchvision.ops import focal_loss

        loss = focal_loss.sigmoid_focal_loss(
            inputs,
            target,
            reduction=self.reduction,
            gamma=self.gamma,
            alpha=self.alpha,
        )
        return loss


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

    Careful: if activation is set to ReLU, ReLU is only applied to the first half of NN outputs!
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

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        # We add another layer either at the front / back of the sequential model. It gets a different name
        # `append_XXX`. The naming of the other layers stays consistent.
        # This allows us to load the state dict of the "non_appended" MLP without errors.
        if append_layer_width:
            assert append_layer_position in ("first", "last")
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(append_layer_width, sizes[0])
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                layers_dict = OrderedDict({str(i): module for i, module in enumerate(layers)})
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["append_relu"] = torch.nn.ReLU()
                layers_dict["append_linear"] = torch.nn.Linear(sizes[-1], append_layer_width)
        else:
            layers_dict = OrderedDict({str(i): module for i, module in enumerate(layers)})

        self.network = torch.nn.Sequential(layers_dict)

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
            One of logsigm, sigm or None. If None, then the doser is disabled and just returns the dosage unchanged.
        """
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
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.device = device
        self.seed = seed
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0
        self.use_drugs_idx = use_drugs_idx
        # self.multi_task = multi_task
        self.enable_cpa_mode = enable_cpa_mode

        # set hyperparameters
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(seed, hparams)

        # store the variables used for initialization (allows restoring model later).
        self.init_args = {
            "num_genes": num_genes,
            "num_drugs": num_drugs,
            "num_covariates": num_covariates,
            "seed": seed,
            "patience": patience,
            "doser_type": doser_type,
            "decoder_activation": decoder_activation,
            "hparams": hparams,
            "use_drugs_idx": use_drugs_idx,
        }

        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]],
            append_layer_width=append_layer_width,
            append_layer_position="first",
        )

        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
            append_layer_width=2 * append_layer_width if append_layer_width else None,
            append_layer_position="last",
        )

        if append_layer_width:
            self.num_genes = append_layer_width

        # self.degs_predictor = None
        # if self.multi_task:
        #     self.degs_predictor = MLP(
        #         [2 * self.hparams["dim"]]
        #         + [2 * self.hparams["dim"]]
        #         + [self.num_genes],
        #         batch_norm=True,
        #     )
        #     self.loss_degs = FocalLoss()

        if self.num_drugs > 0:
            self.adversary_drugs = MLP(
                [self.hparams["dim"]]
                + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                + [self.num_drugs]
            )
            if drug_embeddings is None:
                self.drug_embeddings = torch.nn.Embedding(self.num_drugs, self.hparams["dim"])
                embedding_requires_grad = True
            else:
                self.drug_embeddings = drug_embeddings
                embedding_requires_grad = False

            if self.enable_cpa_mode:
                self.drug_embedding_encoder = None
            else:
                self.drug_embedding_encoder = MLP(
                    [self.drug_embeddings.embedding_dim]
                    + [self.hparams["embedding_encoder_width"]] * self.hparams["embedding_encoder_depth"]
                    + [self.hparams["dim"]],
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
                assert use_drugs_idx, "Amortized doser not yet implemented for `use_drugs_idx=False`"
                # should this also have `batch_norm=False`?
                self.dosers = MLP(
                    [self.drug_embeddings.embedding_dim + 1]
                    + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                    + [1],
                )
            else:
                assert doser_type == "sigm" or doser_type == "logsigm"
                self.dosers = GeneralizedSigmoid(self.num_drugs, self.device, nonlin=doser_type)
            self.doser_type = doser_type

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

        # self.to(self.device)

        # # optimizers
        # has_drugs = self.num_drugs > 0
        # has_covariates = self.num_covariates[0] > 0
        # get_params = lambda model, cond: list(model.parameters()) if cond else []
        # _parameters = (
        #     get_params(self.encoder, True)
        #     + get_params(self.decoder, True)
        #     + get_params(self.drug_embeddings, has_drugs and embedding_requires_grad)
        #     + get_params(self.degs_predictor, self.multi_task)
        #     + get_params(self.drug_embedding_encoder, not self.enable_cpa_mode)
        # )
        # for emb in self.covariates_embeddings:
        #     _parameters.extend(get_params(emb, has_covariates))

        # self.optimizer_autoencoder = torch.optim.Adam(
        #     _parameters,
        #     lr=self.hparams["autoencoder_lr"],
        #     weight_decay=self.hparams["autoencoder_wd"],
        # )

        # _parameters = get_params(self.adversary_drugs, has_drugs)
        # for adv in self.adversary_covariates:
        #     _parameters.extend(get_params(adv, has_covariates))

        # self.optimizer_adversaries = torch.optim.Adam(
        #     _parameters,
        #     lr=self.hparams["adversary_lr"],
        #     weight_decay=self.hparams["adversary_wd"],
        # )

        # if has_drugs:
        #     self.optimizer_dosers = torch.optim.Adam(
        #         self.dosers.parameters(),
        #         lr=self.hparams["dosers_lr"],
        #         weight_decay=self.hparams["dosers_wd"],
        #     )

        # # learning rate schedulers
        # self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_autoencoder,
        #     step_size=self.hparams["step_size_lr"],
        #     gamma=0.5,
        # )

        # self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_adversaries,
        #     step_size=self.hparams["step_size_lr"],
        #     gamma=0.5,
        # )

        # if has_drugs:
        #     self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
        #         self.optimizer_dosers,
        #         step_size=self.hparams["step_size_lr"],
        #         gamma=0.5,
        #     )

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

        If use_drugs_idx is True, then drugs_idx and dosages will be set.
        If use_drugs_idx is False, then drugs will be set.

        @param drugs: A vector of dim [batch_size, num_drugs], where each entry contains the dose of that drug.
        @param drugs_idx: A vector of dim [batch_size]. Each entry contains the index of the applied drug. The
            index is ∈ [0, num_drugs).
        @param dosages: A vector of dim [batch_size]. Each entry contains the dose of the applied drug.
        @return: a tensor of shape [batch_size, drug_embedding_dimension]
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

        drugs, drugs_idx, dosages = _move_inputs(drugs, drugs_idx, dosages, device=self.device)

        latent_drugs = self.drug_embeddings.weight

        if drugs is None:
            if len(drugs_idx.size()) == 0:
                drugs_idx = drugs_idx.unsqueeze(0)

            if len(dosages.size()) == 0:
                dosages = dosages.unsqueeze(0)

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
                    scaled_dosages.append(self.dosers[idx](dosage.unsqueeze(0)).sigmoid())
                scaled_dosages = torch.cat(scaled_dosages, 0)
        elif self.doser_type == "amortized":
            # dosages are 1D, so we unsqueeze them to be (N, 1) which allows using torch.concat().
            # after the dosers we squeeze them back to 1D
            scaled_dosages = self.dosers(
                torch.concat([latent_drugs, torch.unsqueeze(dosages, dim=-1)], dim=1)
            ).squeeze()
        else:
            if drugs_idx is None:
                scaled_dosages = self.dosers(drugs)
            else:
                scaled_dosages = self.dosers(dosages, drugs_idx)

        # unsqueeze if batch_size is 1
        if len(scaled_dosages.size()) == 0:
            scaled_dosages = scaled_dosages.unsqueeze(0)

        if not self.enable_cpa_mode:
            # Transform and adjust dimension to latent dims
            latent_drugs = self.drug_embedding_encoder(latent_drugs)
        else:
            # in CPAMode, we don't use the drug embedding encoder, as it
            # is not part of the CPA paper.
            assert latent_drugs.shape[-1] == self.hparams["dim"], f"{latent_drugs.shape[-1]} != {self.hparams['dim']}"

        if drugs_idx is None:
            return scaled_dosages @ latent_drugs
        else:
            # scale latent vector by scalar scaled_dosage
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
            drug_embedding = self.compute_drug_embeddings_(drugs=drugs, drugs_idx=drugs_idx, dosages=dosages)
            # latent_treated = latent_treated + self.drug_embedding_encoder(
            #     drug_embedding
            # )
            latent_treated = latent_treated + drug_embedding
        if self.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].argmax(1)
                latent_treated = latent_treated + emb_cov(cov_idx)

        cell_drug_embedding = torch.cat([emb_cov(cov_idx), drug_embedding], dim=1)

        gene_reconstructions = self.decoder(latent_treated)

        # convert variance estimates to a positive value in [0, \infty)
        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = F.softplus(gene_reconstructions[:, dim:])
        normalized_reconstructions = torch.concat([mean, var], dim=1)

        if return_latent_basal:
            return normalized_reconstructions, cell_drug_embedding, (latent_basal, drug_embedding, latent_treated)

        return normalized_reconstructions, cell_drug_embedding

    def early_stopping(self, score):
        """
        Possibly early-stops training.
        """
        if score is None:
            # TODO don't really know what to do here
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
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """
        assert (drugs is not None) or (drugs_idx is not None and dosages is not None)

        gene_reconstructions, cell_drug_embedding, latent_basal = self.predict(
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
        reconstruction_loss = self.loss_autoencoder(input=mean, target=genes, var=var)

        # multi_task_loss = torch.tensor([0.0], device=self.device)
        # if self.multi_task:
        #     degs_prediciton = self.degs_predictor(cell_drug_embedding)
        #     multi_task_loss = self.loss_degs(degs_prediciton, degs)

        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            if self.use_drugs_idx:
                adversary_drugs_loss = self.loss_adversary_drugs(adversary_drugs_predictions, drugs_idx)
            else:
                adversary_drugs_loss = self.loss_adversary_drugs(adversary_drugs_predictions, drugs.gt(0).float())

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
        adv_drugs_grad_penalty = torch.tensor([0.0], device=self.device)
        adv_covs_grad_penalty = torch.tensor([0.0], device=self.device)

        if (self.iteration % self.hparams["adversary_steps"]) == 0:

            def compute_gradient_penalty(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            if self.num_drugs > 0:
                adv_drugs_grad_penalty = compute_gradient_penalty(adversary_drugs_predictions.sum(), latent_basal)

            if self.num_covariates[0] > 0:
                adv_covs_grad_penalty = torch.tensor([0.0], device=self.device)
                for pred in adversary_covariate_predictions:
                    adv_covs_grad_penalty += compute_gradient_penalty(pred.sum(), latent_basal)

            self.optimizer_adversaries.zero_grad()
            (
                adversary_drugs_loss
                + self.hparams["penalty_adversary"] * adv_drugs_grad_penalty
                + adversary_covariates_loss
                + self.hparams["penalty_adversary"] * adv_covs_grad_penalty
            ).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            if self.num_drugs > 0:
                self.optimizer_dosers.zero_grad()
            (
                reconstruction_loss
                - self.hparams["reg_adversary"] * adversary_drugs_loss
                - self.hparams["reg_adversary_cov"] * adversary_covariates_loss
                # + self.hparams["reg_multi_task"] * multi_task_loss
            ).backward()
            self.optimizer_autoencoder.step()
            if self.num_drugs > 0:
                self.optimizer_dosers.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
            "penalty_adv_drugs": adv_drugs_grad_penalty.item(),
            "penalty_adv_covariates": adv_covs_grad_penalty.item(),
            "loss_multi_task": multi_task_loss.item(),
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for ComPert
        """

        return self.set_hparams_(self, 0, "")
