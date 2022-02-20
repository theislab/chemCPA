import numpy as np
import pandas as pd
import scanpy as sc
import seml
import torch
from tqdm.auto import tqdm

from compert.data import canonicalize_smiles, drug_names_to_once_canon_smiles
from compert.embedding import get_chemical_representation
from compert.model import ComPert
from compert.paths import CHECKPOINT_DIR
from compert.train import bool2idx, compute_prediction, compute_r2, repeat_n


def load_config(seml_collection, model_hash):
    seml_collection = "finetuning_num_genes"  # "sciplex_hparam"

    results_df = seml.get_results(
        seml_collection,
        to_data_frame=True,
        fields=["config", "config_hash"],
        states=["COMPLETED"],
        filter_dict={"config_hash": model_hash},
    )
    experiment = results_df.apply(
        lambda exp: {
            "hash": exp["config_hash"],
            "seed": exp["config.seed"],
            "_id": exp["_id"],
        },
        axis=1,
    )
    assert len(experiment) == 1
    experiment = experiment[0]
    collection = seml.database.get_collection(seml_collection)
    config = collection.find_one({"_id": experiment["_id"]})["config"]
    assert config["dataset"]["data_params"]["use_drugs_idx"]
    assert config["model"]["additional_params"]["doser_type"] == "amortized"
    config["config_hash"] = model_hash
    return config


def load_dataset(config):
    perturbation_key = config["dataset"]["data_params"]["perturbation_key"]
    smiles_key = config["dataset"]["data_params"]["smiles_key"]
    dataset = sc.read(config["dataset"]["data_params"]["dataset_path"])
    key_dict = {
        "perturbation_key": perturbation_key,
        "smiles_key": smiles_key,
    }
    return dataset, key_dict


def load_smiles(config, dataset, key_dict):
    perturbation_key = key_dict["perturbation_key"]
    smiles_key = key_dict["smiles_key"]

    # this is how the `canon_smiles_unique_sorted` is generated inside compert.data.Dataset
    # we need to have the same ordering of SMILES, else the mapping to pathways will be off
    # when we load the Vanilla embedding. For the other embeddings it's not as important.
    drugs_names = np.array(dataset.obs[perturbation_key].values)
    drugs_names_unique = set()
    for d in drugs_names:
        [drugs_names_unique.add(i) for i in d.split("+")]
    drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
    canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
        list(drugs_names_unique_sorted), dataset, perturbation_key, smiles_key
    )

    smiles_to_pathway_map = {
        canonicalize_smiles(smiles): pathway
        for smiles, pathway in dataset.obs.groupby(
            [config["dataset"]["data_params"]["smiles_key"], "pathway_level_1"]
        ).groups.keys()
    }
    smiles_to_drug_map = {
        canonicalize_smiles(smiles): drug
        for smiles, drug in dataset.obs.groupby(
            [config["dataset"]["data_params"]["smiles_key"], perturbation_key]
        ).groups.keys()
    }

    return canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map


def load_model(config, canon_smiles_unique_sorted):
    model_hash = config["config_hash"]
    model_checkp = CHECKPOINT_DIR / (model_hash + ".pt")

    embedding_model = config["model"]["embedding"]["model"]
    if embedding_model == "vanilla":
        embedding = None
    else:
        embedding = get_chemical_representation(
            smiles=canon_smiles_unique_sorted,
            embedding_model=config["model"]["embedding"]["model"],
            data_dir=config["model"]["embedding"]["directory"],
            device="cuda",
        )
    state_dict, cov_state_dicts, init_args, history = torch.load(model_checkp)
    append_layer_width = (
        config["dataset"]["n_vars"]
        if (config["model"]["append_ae_layer"] and config["model"]["load_pretrained"])
        else None
    )

    if embedding_model != "vanilla":
        state_dict.pop("drug_embeddings.weight")
    model = ComPert(
        **init_args, drug_embeddings=embedding, append_layer_width=append_layer_width
    )
    model = model.eval()

    incomp_keys = model.load_state_dict(state_dict, strict=False)
    if embedding_model == "vanilla":
        assert (
            len(incomp_keys.unexpected_keys) == 0 and len(incomp_keys.missing_keys) == 0
        )
    else:
        # make sure we didn't accidentally load the embedding from the state_dict
        torch.testing.assert_allclose(model.drug_embeddings.weight, embedding.weight)
        assert (
            len(incomp_keys.missing_keys) == 1
            and "drug_embeddings.weight" in incomp_keys.missing_keys
        ), incomp_keys.missing_keys
        assert len(incomp_keys.unexpected_keys) == 0, incomp_keys.unexpected_keys

    return model, embedding


def compute_drug_embeddings(model, embedding, dosage=1e4):
    all_drugs_idx = torch.tensor(list(range(len(embedding.weight))))
    dosages = dosage * torch.ones((len(embedding.weight),))
    # dosages = torch.ones((len(embedding.weight),))
    with torch.no_grad():
        # scaled the drug embeddings using the doser
        scaled_embeddings = model.compute_drug_embeddings_(
            drugs_idx=all_drugs_idx, dosages=dosages
        )
        # apply drug embedder
        transf_embeddings = model.drug_embedding_encoder(scaled_embeddings)
    return transf_embeddings


def compute_pred(model, dataset, dosages=[1e4], genes_control=None):
    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")

    predictions_dict = {}
    drug_r2 = {}
    for cell_drug_dose_comb, category_count in tqdm(
        zip(*np.unique(dataset.pert_categories, return_counts=True))
    ):
        if dataset.perturbation_key is None:
            break

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes[cell_drug_dose_comb])
        )
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device="cuda")
        if genes_control is None:
            n_obs = y_true.size(0)
        else:
            assert isinstance(genes_control, torch.Tensor)
            n_obs = genes_control.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]

        if dataset.dosages[idx] not in dosages:
            continue

        if (dataset.covariates[0][idx] != torch.Tensor([0, 0, 1])).any():
            continue

        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)

        if genes_control is None:
            # print("Predicting AE alike.")
            mean_pred, _ = compute_prediction(
                model,
                y_true,
                emb_drugs,
                emb_covs,
            )
        else:
            # print("Predicting counterfactuals.")
            mean_pred, _ = compute_prediction(
                model,
                genes_control,
                emb_drugs,
                emb_covs,
            )

        y_pred = mean_pred.mean(0)
        y_true = y_true.mean(0)
        r2_m_de = compute_r2(y_true[idx_de].cuda(), y_pred[idx_de].cuda())
        print(f"{cell_drug_dose_comb}: {r2_m_de:.2f}")
        predictions_dict[cell_drug_dose_comb] = [y_true, y_pred, idx_de]
        drug_r2[cell_drug_dose_comb.split("_")[1]] = r2_m_de
    return drug_r2, predictions_dict
