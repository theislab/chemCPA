import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score
from tqdm.auto import tqdm

import chemCPA.data
from chemCPA.data import SubDataset
from chemCPA.model import MLP, ComPert


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
        )[0].detach()
    else:
        genes_pred = autoencoder.predict(genes=genes, drugs=emb_drugs, covariates=emb_covs)[0].detach()
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
    return metric.compute().item()


def evaluate_logfold_r2(autoencoder: ComPert, ds_treated: SubDataset, ds_ctrl: SubDataset):
    logfold_score = []
    signs_score = []
    # assumes that `pert_categories` where constructed with first covariate type
    cov_type = ds_treated.covariate_keys[0]
    treated_pert_cat_index = pd.Index(ds_treated.pert_categories, dtype="category")
    ctrl_cov_cat_index = pd.Index(ds_ctrl.covariate_names[cov_type], dtype="category")
    for cell_drug_dose_comb, category_count in zip(*np.unique(ds_treated.pert_categories, return_counts=True)):
        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if "dmso" in cell_drug_dose_comb.lower() or "control" in cell_drug_dose_comb.lower():
            continue

        covariate = cell_drug_dose_comb.split("_")[0]

        bool_pert_categoy = treated_pert_cat_index.get_loc(cell_drug_dose_comb)
        idx_treated_all = bool2idx(bool_pert_categoy)
        idx_treated = idx_treated_all[0]

        # this doesn't work on LINCS. Often `covariate` will not exist at all in the `ds_ctrl` (example: ASC.C)
        # this means we get `n_idx_ctrl == 0`, which results in all kinds of NaNs later on.
        # Once we figured out how to deal with this we can replace this `==` matching with an index lookup.
        bool_ctrl_all = ds_ctrl.covariate_names[cov_type] == covariate
        idx_ctrl_all = bool2idx(bool_ctrl_all)
        n_idx_ctrl = len(idx_ctrl_all)

        bool_de = ds_treated.var_names.isin(np.array(ds_treated.de_genes[cell_drug_dose_comb]))
        idx_de = bool2idx(bool_de)

        if n_idx_ctrl == 1:
            print(
                f"For covariate {covariate} we have only one control in current set of observations. Skipping {cell_drug_dose_comb}."
            )
            continue

        emb_covs = [repeat_n(cov[idx_treated], n_idx_ctrl) for cov in ds_treated.covariates]
        if ds_treated.use_drugs_idx:
            emb_drugs = (
                repeat_n(ds_treated.drugs_idx[idx_treated], n_idx_ctrl).squeeze(),
                repeat_n(ds_treated.dosages[idx_treated], n_idx_ctrl).squeeze(),
            )
        else:
            emb_drugs = repeat_n(ds_treated.drugs[idx_treated], n_idx_ctrl)

        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)
        genes_ctrl = ds_ctrl.genes[idx_ctrl_all].to(device="cuda")

        genes_pred, _ = compute_prediction(
            autoencoder,
            genes_ctrl,
            emb_drugs,
            emb_covs,
        )
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)
        genes_true = ds_treated.genes[idx_treated_all, :].to(device="cuda")

        y_ctrl = genes_ctrl.mean(0)[idx_de]
        y_pred = genes_pred.mean(0)[idx_de]
        y_true = genes_true.mean(0)[idx_de]

        eps = 1e-5
        pred = torch.log2((y_pred + eps) / (y_ctrl + eps))
        true = torch.log2((y_true + eps) / (y_ctrl + eps))
        r2 = compute_r2(true, pred)
        acc_signs = ((pred * true) > 0).sum() / len(true)

        logfold_score.append(r2)
        signs_score.append(acc_signs.item())
    return mean(logfold_score), mean(signs_score)


def evaluate_disentanglement(autoencoder: ComPert, dataloader: DataLoader):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.

    """

    # generate random indices to subselect the dataset
    print(f"Size of disentanglement testdata: {len(dataloader)*dataloader.batch_size}")
    data = dataloader.dataset
    with torch.no_grad():
        autoencoder.to(autoencoder.device)
        latent_basal = []
        for batch in dataloader:
            genes, drugs_idx, dosages, degs, covariates = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4:],
            )

            _, _, _latents = autoencoder.predict(
                genes=genes,
                drugs=None,
                drugs_idx=drugs_idx,
                dosages=dosages,
                covariates=covariates,
                return_latent_basal=True,
            )
            _latent_basal = _latents[0]
            latent_basal.append(_latent_basal)
        latent_basal = torch.cat(latent_basal, dim=0)
        # _, _, latent_basal = autoencoder.predict(
        #     genes=data.genes,
        #     drugs_idx=data.drugs_idx,
        #     dosages=data.dosages,
        #     covariates=data.covariates,
        #     return_latent_basal=True,
        # )

    mean = latent_basal.mean(dim=0, keepdim=True)
    stddev = latent_basal.std(0, unbiased=False, keepdim=True)
    normalized_basal = (latent_basal - mean) / stddev

    criterion = nn.CrossEntropyLoss()
    pert_score, cov_scores = 0, []

    def compute_score(labels):
        unique_labels = set(labels)
        label_to_idx = {labels: idx for idx, labels in enumerate(unique_labels)}
        labels_tensor = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long, device="cuda")
        assert normalized_basal.size(0) == len(labels_tensor)
        dataset = torch.utils.data.TensorDataset(normalized_basal, labels_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        # 2 non-linear layers of size <input_dimension>
        # followed by a linear layer.
        disentanglement_classifier = MLP(
            [normalized_basal.size(1)] + [normalized_basal.size(1) for _ in range(2)] + [len(unique_labels)]
        ).to("cuda")
        optimizer = torch.optim.Adam(disentanglement_classifier.parameters(), lr=1e-2)

        for epoch in tqdm(range(400)):
            for X, y in data_loader:
                pred = disentanglement_classifier(X)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            pred = disentanglement_classifier(normalized_basal).argmax(dim=1)
            acc = torch.sum(pred == labels_tensor) / len(labels_tensor)
        return acc.item()

    if data.perturbation_key is not None:
        pert_score = compute_score(data.drugs_names)
    for cov in list(data.covariate_names):
        cov_scores = []
        if len(np.unique(data.covariate_names[cov])) == 0:
            cov_scores = [0]
            break
        else:
            cov_scores.append(compute_score(data.covariate_names[cov]))
        return [pert_score] + cov_scores


def evaluate_r2(model: ComPert, dataset: SubDataset, genes_control: torch.Tensor):
    """
    Measures different quality metrics about an ComPert `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """
    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    n_rows = genes_control.size(0)
    genes_control = genes_control.to(model.device)

    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
    for cell_drug_dose_comb, category_count in zip(*np.unique(dataset.pert_categories, return_counts=True)):
        if dataset.perturbation_key is None:
            break

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if "dmso" in cell_drug_dose_comb.lower() or "control" in cell_drug_dose_comb.lower():
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        bool_de = dataset.var_names.isin(np.array(dataset.de_genes[cell_drug_dose_comb]))
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]

        emb_covs = [repeat_n(cov[idx], n_rows) for cov in dataset.covariates]
        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_rows).squeeze(),
                repeat_n(dataset.dosages[idx], n_rows).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_rows)
        mean_pred, var_pred = compute_prediction(
            model,
            genes_control,
            emb_drugs,
            emb_covs,
        )

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)
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

        # to be investigated
        if r2_m_de == float("-inf") or r2_v_de == float("-inf"):
            continue

        mean_score.append(max(r2_m, 0.0))
        var_score.append(max(r2_v, 0.0))
        mean_score_de.append(max(r2_m_de, 0.0))
        var_score_de.append(max(r2_v_de, 0.0))
    if len(mean_score) > 0:
        return [np.mean(s) for s in [mean_score, mean_score_de, var_score, var_score_de]]
    else:
        return []


def evaluate_r2_sc(autoencoder: ComPert, dataset: SubDataset):
    """
    Measures quality metric about an ComPert `autoencoder`. Computing
    the reconstruction of a single datapoint in terms of the R2 score.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """
    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []

    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
    inf_combinations = set()
    for cell_drug_dose_comb, category_count in zip(*np.unique(dataset.pert_categories, return_counts=True)):
        if dataset.perturbation_key is None:
            break

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if "dmso" in cell_drug_dose_comb.lower() or "control" in cell_drug_dose_comb.lower():
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        bool_de = dataset.var_names.isin(np.array(dataset.de_genes[cell_drug_dose_comb]))
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device="cuda")
        n_obs = y_true.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]
        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)

        mean_pred, var_pred = compute_prediction(
            autoencoder,
            y_true,
            emb_drugs,
            emb_covs,
        )

        # mean of r2 scores for current cell_drug_dose_comb
        r2_m = torch.Tensor([compute_r2(y_true[i], mean_pred[i]) for i in range(n_obs)]).mean()
        r2_m_de = torch.Tensor([compute_r2(y_true[i, idx_de], mean_pred[i, idx_de]) for i in range(n_obs)]).mean()
        # r2 score for predicted variance of obs in current cell_drug_dose_comb
        yt_v = y_true.var(dim=0)
        yp_v = var_pred.mean(dim=0)
        r2_v = compute_r2(yt_v, yp_v)
        r2_v_de = compute_r2(yt_v[idx_de], yp_v[idx_de])

        # if r2_m_de == float("-inf") or r2_v_de == float("-inf"):
        #     continue

        mean_score.append(r2_m) if not r2_m == float("-inf") else inf_combinations.add(cell_drug_dose_comb)
        var_score.append(r2_v) if not r2_v == float("-inf") else inf_combinations.add(cell_drug_dose_comb)
        mean_score_de.append(r2_m_de) if not r2_m_de == float("-inf") else inf_combinations.add(cell_drug_dose_comb)
        var_score_de.append(r2_v_de) if not r2_v_de == float("-inf") else inf_combinations.add(cell_drug_dose_comb)
    print(f"{len(inf_combinations)} combinations had '-inf' R2 scores:\n\t {inf_combinations}")
    if len(mean_score) > 0:
        return [np.mean(s, dtype=float) for s in [mean_score, mean_score_de, var_score, var_score_de]]
    else:
        return []


def evaluate(
    autoencoder,
    datasets,
    eval_stats,
    run_disentangle=False,
    run_r2=True,
    # run_r2_sc=False,
    # run_logfold=False,
):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.

    eval_stats is the default evaluation dictionary that is updated with the missing scores
    """
    start_time = time.time()
    autoencoder.eval()
    if run_disentangle:
        logging.info("Running disentanglement evaluation")
        drug_names, drug_counts = np.unique(datasets["test"].drugs_names, return_counts=True)
        disent_scores = evaluate_disentanglement(autoencoder, datasets["test"])
        stats_disent_pert = disent_scores[0]
        # optimal score == always predicting the most common drug
        optimal_disent_score = max(drug_counts) / len(datasets["test"])
        stats_disent_cov = disent_scores[1:]
    else:
        stats_disent_pert = [0]
        optimal_disent_score = 0
        stats_disent_cov = [0]

    with torch.no_grad():
        evaluation_stats = {
            "perturbation disentanglement": stats_disent_pert,
            "optimal for perturbations": optimal_disent_score,
            "covariate disentanglement": stats_disent_cov,
            "optimal for covariates": (
                [
                    max(cov.mean(axis=0)).item() for cov in datasets["test"].covariates
                ]  # mean over OHE embedding of covariates
                if datasets["test"].num_covariates[0] > 0
                else None
            ),
        }

        if run_r2:
            logging.info("Running R2 evaluation")
            evaluation_stats["training"] = evaluate_r2(
                autoencoder,
                datasets["training_treated"],
                datasets["training_control"].genes,
            )
            if "test" in eval_stats:
                evaluation_stats["test"] = eval_stats["test"]
            else:
                evaluation_stats["test"] = evaluate_r2(
                    autoencoder,
                    datasets["test_treated"],
                    datasets["test_control"].genes,
                )
            evaluation_stats["ood"] = evaluate_r2(
                autoencoder,
                datasets["ood"],
                datasets["test_control"].genes,
            )

        # if run_r2_sc:
        #     logging.info("Running single-cell R2 evaluation")
        #     evaluation_stats["training_sc"] = evaluate_r2_sc(
        #         autoencoder, datasets["training_treated"]
        #     )
        #     if "test_sc" in eval_stats:
        #         evaluation_stats["test_sc"] = eval_stats["test_sc"]
        #     else:
        #         evaluation_stats["test_sc"] = evaluate_r2_sc(
        #             autoencoder, datasets["test_treated"]
        #         )
        #     evaluation_stats["ood_sc"] = (evaluate_r2_sc(autoencoder, datasets["ood"]),)

        # if run_logfold:
        #     logging.info("Running logfold evaluation")
        #     evaluation_stats["training_logfold"] = evaluate_logfold_r2(
        #         autoencoder,
        #         datasets["training_treated"],
        #         datasets["training_control"],
        #     )
        #     evaluation_stats["test_logfold"] = evaluate_logfold_r2(
        #         autoencoder,
        #         datasets["test_treated"],
        #         datasets["test_control"],
        #     )
        #     evaluation_stats["ood_logfold"] = evaluate_logfold_r2(
        #         autoencoder,
        #         datasets["ood"],
        #         datasets["test_control"],
        #     )

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
