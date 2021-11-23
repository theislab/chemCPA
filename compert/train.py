# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torchmetrics import R2Score

from compert.model import ComPert


def bool2idx(x):
    """
    Returns the indices of the True-valued entries in a boolean array `x`
    """
    return np.where(x)[0]


def repeat_n(x, n):
    """
    Returns an n-times repeated version of the Tensor x,
    repetition dimension is axis 0
    """
    # copy tensor to device BEFORE replicating it n times
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)


def mean(x: list):
    """
    Returns mean of list `x`
    """
    return np.mean(x) if len(x) else -1


def compute_prediction(autoencoder: ComPert, genes, emb_drugs, emb_covs):
    """
    Computes the prediction of a ComPert `autoencoder` and
    directly splits into `mean` and `variance` predictions
    """
    if autoencoder.use_drugs_idx:
        assert len(emb_drugs) == 2
        genes_pred = autoencoder.predict(
            genes=genes,
            drugs_idx=emb_drugs[0],
            dosages=emb_drugs[1],
            covariates=emb_covs,
        ).detach()
    else:
        genes_pred = autoencoder.predict(
            genes=genes, drugs=emb_drugs, covariates=emb_covs
        ).detach()
    dim = genes.size(1)
    mean = genes_pred[:, :dim]
    var = genes_pred[:, dim:]
    return mean, var


def compute_r2(y_true, y_pred):
    """
    Computes the r2 score for `y_true` and `y_pred`,
    returns `-1` when `y_pred` contains nan values
    """
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    metric = R2Score().to(y_true.device)
    metric.update(y_pred, y_true)  # same as sklearn.r2_score(y_true, y_pred)
    r2 = 0 if torch.isnan(y_pred).any() else max(metric.compute().item(), 0)
    return r2


def evaluate_logfold_r2(autoencoder, ds_treated, ds_ctrl):
    logfold_score = []
    # assumes that `pert_categories` where constructed with first covariate type
    cov_type = ds_treated.covariate_keys[0]
    for pert_category in np.unique(ds_treated.pert_categories):
        covariate = pert_category.split("_")[0]
        # the next line is where it is spending all it's time. Why?
        idx_treated_all = bool2idx(ds_treated.pert_categories == pert_category)
        idx_treated, n_idx_treated = idx_treated_all[0], len(idx_treated_all)

        bool_ctrl_all = ds_ctrl.covariate_names[cov_type] == covariate
        idx_ctrl_all = bool2idx(bool_ctrl_all)
        n_idx_ctrl = len(idx_ctrl_all)
        # estimate metrics only for reasonably-sized drug/cell-type combos
        if n_idx_treated <= 5:
            continue

        emb_covs = [
            repeat_n(cov[idx_treated], n_idx_ctrl) for cov in ds_treated.covariates
        ]
        if ds_treated.use_drugs_idx:
            emb_drugs = (
                repeat_n(ds_treated.drugs_idx[idx_treated], n_idx_ctrl).squeeze(),
                repeat_n(ds_treated.dosages[idx_treated], n_idx_ctrl).squeeze(),
            )
        else:
            emb_drugs = repeat_n(ds_treated.drugs[idx_treated], n_idx_ctrl)

        genes_ctrl = ds_ctrl.genes[idx_ctrl_all].to(device="cuda")

        genes_pred, _ = compute_prediction(
            autoencoder,
            genes_ctrl,
            emb_drugs,
            emb_covs,
        )
        genes_true = ds_treated.genes[idx_treated_all, :].to(device="cuda")

        y_ctrl = genes_ctrl.mean(0)
        y_pred = genes_pred.mean(0)
        y_true = genes_true.mean(0)

        eps = 1e-5
        pred = torch.log2((y_pred + eps) / (y_ctrl + eps))
        true = torch.log2((y_true + eps) / (y_ctrl + eps))
        r2 = compute_r2(true, pred)

        logfold_score.append(r2)
    return mean(logfold_score)


def evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.

    """
    if dataset.use_drugs_idx:
        _, latent_basal = autoencoder.predict(
            genes=dataset.genes,
            drugs_idx=dataset.drugs_idx,
            dosages=dataset.dosages,
            covariates=dataset.covariates,
            return_latent_basal=True,
        )
    else:
        _, latent_basal = autoencoder.predict(
            genes=dataset.genes,
            drugs=dataset.drugs,
            covariates=dataset.covariates,
            return_latent_basal=True,
        )

    latent_basal = latent_basal.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            max_iter=3000,
            # n_jobs=-1,
            # verbose=2,
            tol=1e-2,
        )

    pert_scores, cov_scores = 0, []

    def compute_score(labels):
        scaler = StandardScaler().fit_transform(latent_basal)
        scorer = make_scorer(balanced_accuracy_score)
        return cross_val_score(clf, scaler, labels, scoring=scorer, cv=5, n_jobs=-1)

    if dataset.perturbation_key is not None:
        pert_scores = compute_score(dataset.drugs_names)
    for cov in list(dataset.covariate_names):
        cov_scores = []
        if len(np.unique(dataset.covariate_names[cov])) == 0:
            cov_scores = [0]
            break
        else:
            cov_scores.append(compute_score(dataset.covariate_names[cov]))
        return [np.mean(pert_scores), *[np.mean(cov_score) for cov_score in cov_scores]]


def evaluate_r2(autoencoder, dataset, genes_control):
    """
    Measures different quality metrics about an ComPert `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """
    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)

    for pert_category in np.unique(dataset.pert_categories):
        if dataset.perturbation_key is None:
            break
        # pert_category category contains: 'celltype_perturbation_dose' info
        bool_de = dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
        idx_de = bool2idx(bool_de)

        # spending a lot of time here, could this be precomputed?
        idx_all = bool2idx(dataset.pert_categories == pert_category)
        idx, n_idx = idx_all[0], len(idx_all)

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if n_idx <= 5:
            continue

        emb_covs = [repeat_n(cov[idx], num) for cov in dataset.covariates]
        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], num).squeeze(),
                repeat_n(dataset.dosages[idx], num).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], num)
        mean_pred, var_pred = compute_prediction(
            autoencoder,
            genes_control,
            emb_drugs,
            emb_covs,
        )

        # copies just the needed genes to GPU
        y_true = dataset.genes[idx_all, :].to(device="cuda")

        # true means and variances
        yt_m = y_true.mean(dim=0)
        yt_v = y_true.var(dim=0)
        # predicted means and variances
        yp_m = mean_pred.mean(dim=0)
        yp_v = var_pred.mean(dim=0)

        r2_m = compute_r2(yt_m, yp_m)
        r2_v = compute_r2(yt_v, yp_v)
        r2_m_de = compute_r2(yt_m[idx_de], yp_m[idx_de])
        r2_v_de = compute_r2(yt_v[idx_de], yp_v[idx_de])

        mean_score.append(r2_m)
        var_score.append(r2_v)
        mean_score_de.append(r2_m_de)
        var_score_de.append(r2_v_de)
    print(f"Number of different r2 computations: {len(mean_score)}")
    return [mean(s) for s in [mean_score, mean_score_de, var_score, var_score_de]]


def evaluate(autoencoder, datasets, disentangle=False):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """
    start_time = time.time()
    autoencoder.eval()
    with torch.no_grad():
        if disentangle:
            disent_scores = evaluate_disentanglement(autoencoder, datasets["test"])
            stats_disent_pert = disent_scores[0]
            stats_disent_cov = disent_scores[1:]
        else:
            stats_disent_pert = [0]
            stats_disent_cov = [0]

        evaluation_stats = {
            "training": evaluate_r2(
                autoencoder,
                datasets["training_treated"],
                datasets["training_control"].genes,
            ),
            "test": evaluate_r2(
                autoencoder, datasets["test_treated"], datasets["test_control"].genes
            ),
            "ood": evaluate_r2(
                autoencoder, datasets["ood"], datasets["test_control"].genes
            ),
            "training_logfold": evaluate_logfold_r2(
                autoencoder, datasets["training_treated"], datasets["training_control"]
            ),
            "test_logfold": evaluate_logfold_r2(
                autoencoder, datasets["test_treated"], datasets["test_control"]
            ),
            "ood_logfold": evaluate_logfold_r2(
                autoencoder, datasets["ood"], datasets["test_control"]
            ),
            "perturbation disentanglement": stats_disent_pert,
            "optimal for perturbations": 1 / datasets["test"].num_drugs
            if datasets["test"].num_drugs > 0
            else None,
            "covariate disentanglement": stats_disent_cov,
            "optimal for covariates": [
                1 / num for num in datasets["test"].num_covariates
            ]
            if datasets["test"].num_covariates[0] > 0
            else None,
        }
    autoencoder.train()
    ellapsed_minutes = (time.time() - start_time) / 60
    print(f"\nTook {ellapsed_minutes:.1f} min for evaluation.\n")
    return evaluation_stats


def custom_collate(batch):
    transposed = zip(*batch)
    concat_batch = []
    for samples in transposed:
        if samples[0] is None:
            concat_batch.append(None)
        else:
            # we move to CUDA here so that prefetching in the DataLoader already yields
            # ready-to-process CUDA tensors
            concat_batch.append(torch.stack(samples, 0).to("cuda"))
    return concat_batch
