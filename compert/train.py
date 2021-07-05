# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from compert.graph_model.graph_model import Drugemb
import os
import json
import argparse

import torch
import numpy as np
from collections import defaultdict

from compert.data import load_dataset_splits
from compert.model import ComPert

from sklearn.metrics import r2_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import time


def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


def evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.

    """
    _, latent_basal = autoencoder.predict(
        dataset.genes,
        dataset.drugs,
        dataset.covariates,
        return_latent_basal=True,
    )

    latent_basal = latent_basal.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(solver="liblinear", multi_class="auto", max_iter=10000)

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
    tasked to translate some `genes_control` into each of the drug/cell_type
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)

    total_cells = len(dataset)

    for pert_category in np.unique(dataset.pert_categories):
        if dataset.perturbation_key is None:
            break
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx = np.where(
            dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
        )[0]

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > 30:
            emb_drugs = dataset.drugs[idx][0].view(1, -1).repeat(num, 1).clone()
            emb_cts = (
                dataset.cell_types[idx][0].view(1, -1).repeat(num, 1).clone()
            )  # TODO: Adjust evaluation to covariates

            genes_predict = (
                autoencoder.predict(genes_control, emb_drugs, emb_cts).detach().cpu()
            )

            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]

            # estimate metrics only for reasonably-sized drug/cell-type combos

            y_true = dataset.genes[idx, :].numpy()

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)
            # predicted means and variances
            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        stats_test = evaluate_r2(
            autoencoder, datasets["test_treated"], datasets["test_control"].genes
        )

        disent_scores = evaluate_disentanglement(autoencoder, datasets["test"])
        stats_disent_pert = disent_scores[0]
        stats_disent_cov = disent_scores[1:]

        evaluation_stats = {
            "training": evaluate_r2(
                autoencoder,
                datasets["training_treated"],
                datasets["training_control"].genes,
            ),
            "test": stats_test,
            "ood": evaluate_r2(
                autoencoder, datasets["ood"], datasets["test_control"].genes
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
    return evaluation_stats


def prepare_compert(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_dataset_splits(
        args["dataset_path"],
        args["perturbation_key"],
        args["dose_key"],
        args["covariate_keys"],
        args["smiles_key"],
        args["split_key"],
        args["mol_featurizer"],
    )
    if args["gnn_model"] is not None:
        drug_embeddings = Drugemb(
            dim=256,  # TODO: This is set only in Compert model
            gnn_model=args["gnn_model"],
            graph_feats_shape=datasets["training"].graph_feats_shape,
            idx_wo_smiles=datasets["training"].idx_wo_smiles,
            batched_graph_collection=datasets["training"].batched_graph_collection,
            device=device,
        )
    else:
        drug_embeddings = None

    autoencoder = ComPert(
        datasets["training"].num_genes,
        datasets["training"].num_drugs,
        datasets["training"].num_covariates,
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        doser_type=args["doser_type"],
        patience=args["patience"],
        hparams=args["hparams"],
        decoder_activation=args["decoder_activation"],
        drug_embeddings=drug_embeddings,
    )
    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)

    return autoencoder, datasets


def custom_collate(batch):
    transposed = zip(*batch)
    concat_batch = []
    for samples in transposed:
        if samples[0] is None:
            concat_batch.append(None)
        else:
            concat_batch.append(torch.stack(samples, 0))
    return concat_batch


def train_compert(args, return_model=False, ignore_evaluation=True):
    """
    Trains a ComPert autoencoder
    """

    autoencoder, datasets = prepare_compert(args)

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=autoencoder.hparams["batch_size"],
                collate_fn=custom_collate,
                shuffle=True,
            )
        }
    )

    pjson({"training_args": args})
    pjson({"autoencoder_params": autoencoder.hparams})

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for data in datasets["loader_tr"]:
            genes, drugs, covariates = data[0], data[1], data[2:]
            minibatch_training_stats = autoencoder.update(genes, drugs, covariates)

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(val)
        autoencoder.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        autoencoder.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or (
            epoch == args["max_epochs"] - 1
        )

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            if not ignore_evaluation:
                evaluation_stats = evaluate(autoencoder, datasets)
                for key, val in evaluation_stats.items():
                    if not (key in autoencoder.history.keys()):
                        autoencoder.history[key] = []
                    autoencoder.history[key].append(val)
                autoencoder.history["stats_epoch"].append(epoch)
            else:
                evaluation_stats = {}

            pjson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            torch.save(
                (autoencoder.state_dict(), args, autoencoder.history),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            pjson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )
            if not ignore_evaluation:
                stop = stop or autoencoder.early_stopping(
                    np.mean(evaluation_stats["test"])
                )
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description="Drug combinations.")
    # dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--perturbation_key", type=str, default="condition")
    parser.add_argument("--dose_key", type=str, default="dose_val")
    parser.add_argument("--covariate_keys", type=str, default="cell_type")
    parser.add_argument("--split_key", type=str, default="split")
    parser.add_argument("--loss_ae", type=str, default="gauss")
    parser.add_argument("--doser_type", type=str, default="sigm")
    parser.add_argument("--decoder_activation", type=str, default="linear")

    # ComPert arguments (see set_hparams_() in compert.model.ComPert)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--max_minutes", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint_freq", type=int, default=20)

    # output folder
    parser.add_argument("--save_dir", type=str, required=True)
    # number of trials when executing compert.sweep
    parser.add_argument("--sweep_seeds", type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    CLI = False
    if CLI:
        train_compert(parse_arguments())
    else:
        max_minutes = 1  # testing
        gnn_models = ["AttentiveFP", "GAT", "GCN", "MPNN", "weave"]
        for model in gnn_models:
            args = {
                "dataset_path": "datasets/trapnell_cpa_subset.h5ad",  # full path to the anndata dataset
                "covariate_keys": [
                    "cell_type"
                ],  # necessary field for cell types. Fill it with a dummy variable if no celltypes present.
                "split_key": "split",  # necessary field for train, test, ood splits.
                "perturbation_key": "condition",  # necessary field for perturbations
                "dose_key": "dose",  # necessary field for dose. Fill in with dummy variable if dose is the same.
                "gnn_model": model,
                "smiles_key": "SMILES",
                "mol_featurizer": "canonical",
                "checkpoint_freq": 40,  # checkoint frequencty to save intermediate results
                "hparams": "",  # autoencoder architecture
                "max_epochs": 20,  # maximum epochs for training
                "max_minutes": max_minutes,  # maximum computation time
                "patience": 20,  # patience for early stopping
                "loss_ae": "gauss",  # loss (currently only gaussian loss is supported)
                "doser_type": None,  # non-linearity for doser function
                "save_dir": "notebooks/tmp_save_dir/",  # directory to save the model
                "decoder_activation": "linear",  # last layer of the decoder
                "seed": 0,  # random seed
                "sweep_seeds": 0,
            }
            autoencoder, datasets = train_compert(
                args, return_model=True, ignore_evaluation=False
            )
