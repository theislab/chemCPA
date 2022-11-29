# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %%
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap.plot
from anndata import AnnData
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from utils import (  # load_config,
    compute_drug_embeddings,
    compute_pred,
    compute_pred_ctrl,
    load_dataset,
    load_model,
    load_smiles,
)

from chemCPA.data import load_dataset_splits
from chemCPA.paths import DATA_DIR, FIGURE_DIR, PROJECT_DIR, ROOT

# %%
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %%
BLACK = False
SAVEFIG = False

# %%
if BLACK:
    plt.style.use("dark_background")
else:
    matplotlib.style.use("fivethirtyeight")
    matplotlib.style.use("seaborn-talk")
    matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
    sns.set_style("whitegrid")

matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 300
sns.set_context("poster")

# %%
seml_collection = "baseline_comparison"

cpa_A549 = "044c4dba0c8719985c3622834f2cbd58"
cpa_K562 = "5ea85d5dd7abd5962d1d3eeff1b8c1ff"
cpa_MCF7 = "4d98e7d857f497d870e19c6d12175aaa"

chemCPA_A549_pretrained = "3326f900c45faaf99ca4400f78c58847"
chemCPA_A549 = "8779ff45830000c6bc8e22023bb1cb2c"

chemCPA_K562_pretrained = "6388fa373386c11e40dceb5e2e8a113d"
chemCPA_K562 = "34fd06018d6e2662ccd5da7a16b57334"

chemCPA_MCF7_pretrained = "2075a457bafdca5948ab671b77757974"
chemCPA_MCF7 = "6ad52ba3939397521c5050ca1dd89a4c"


# %%
def load_config(seml_collection, model_hash):
    file_path = PROJECT_DIR / f"{seml_collection}.json"  # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)

    for _config in tqdm(file_data):
        if _config["config_hash"] == model_hash:
            # print(config)
            config = _config["config"]
            config["config_hash"] = _config["config_hash"]
    return config


# %%
config = load_config(seml_collection, chemCPA_MCF7_pretrained)

config["dataset"]["data_params"]["dataset_path"] = (
    ROOT / config["dataset"]["data_params"]["dataset_path"]
)
config["model"]["embedding"]["directory"] = (
    ROOT / config["model"]["embedding"]["directory"]
)

dataset, key_dict = load_dataset(config)

config["dataset"]["n_vars"] = dataset.n_vars

canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict, True
)
model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %%
data_params = config["dataset"]["data_params"]
datasets = load_dataset_splits(**data_params, return_dataset=False)

# %%
dosages = [1e1, 1e2, 1e3, 1e4]
cell_lines = ["A549", "K562", "MCF7"]
use_DEGs = True


# %%
def get_baseline_predictions(
    hash,
    seml_collection="baseline_comparison",
    smiles=None,
    dosages=[1e1, 1e2, 1e3, 1e4],
    cell_lines=["A549", "K562", "MCF7"],
    use_DEGs=False,
    verbose=False,
    name_tag=None,
):
    if smiles is None:
        smiles = canon_smiles_unique_sorted

    config = load_config(seml_collection, hash)
    config["dataset"]["n_vars"] = dataset.n_vars
    config["dataset"]["data_params"]["dataset_path"] = (
        ROOT / config["dataset"]["data_params"]["dataset_path"]
    )
    config["model"]["embedding"]["directory"] = (
        ROOT / config["model"]["embedding"]["directory"]
    )
    data_params = config["dataset"]["data_params"]
    datasets = load_dataset_splits(**data_params, return_dataset=False)

    predictions, _ = compute_pred_ctrl(
        datasets["ood"],
        dataset_ctrl=datasets["test_control"],
        # dataset_ctrl=datasets["training_control"],
        dosages=dosages,
        cell_lines=cell_lines,
        use_DEGs=use_DEGs,
        verbose=verbose,
    )

    predictions = pd.DataFrame.from_dict(predictions, orient="index", columns=["R2"])
    if name_tag:
        predictions["model"] = name_tag
    predictions["genes"] = "degs" if use_DEGs else "all"
    return predictions


# %%
seml_collection = "baseline_comparison"

base_hash_dict = dict(
    baseline_A549="044c4dba0c8719985c3622834f2cbd58",
    baseline_K562="5ea85d5dd7abd5962d1d3eeff1b8c1ff",
    baseline_MCF7="4d98e7d857f497d870e19c6d12175aaa",
)

# %%
# drug_r2_baseline_degs, _ = compute_pred_ctrl(
#     dataset=datasets["ood"],
#     dataset_ctrl=datasets["test_control"],
#     dosages=dosages,
#     cell_lines=cell_lines,
#     use_DEGs=True,
#     verbose=False,
# )

# drug_r2_baseline_all, _ = compute_pred_ctrl(
#     dataset=datasets["ood"],
#     dataset_ctrl=datasets["test_control"],
#     dosages=dosages,
#     cell_lines=cell_lines,
#     use_DEGs=False,
#     verbose=False,
# )
predictions = []

predictions.extend(
    [
        get_baseline_predictions(_hash, name_tag=name_tag, use_DEGs=True)
        for name_tag, _hash in base_hash_dict.items()
    ]
)

predictions.extend(
    [
        get_baseline_predictions(_hash, name_tag=name_tag, use_DEGs=False)
        for name_tag, _hash in base_hash_dict.items()
    ]
)

# %%
predictions

# %%
datasets["test_control"].genes.size()

# %%
adata = sc.read(DATA_DIR / "adata_baseline.h5ad")
pd.crosstab(adata.obs["control"], adata.obs["split_baseline_A549"])

# %%
pd.crosstab(adata.obs["control"], adata.obs["split_baseline_K562"])

# %%
pd.crosstab(adata.obs["control"], adata.obs["split_baseline_MCF7"])


# %%
def get_model_predictions(
    hash,
    seml_collection="baseline_comparison",
    smiles=None,
    dosages=[1e1, 1e2, 1e3, 1e4],
    cell_lines=["A549", "K562", "MCF7"],
    use_DEGs=False,
    verbose=False,
    name_tag=None,
):
    if smiles is None:
        smiles = canon_smiles_unique_sorted

    config = load_config(seml_collection, hash)
    config["dataset"]["n_vars"] = dataset.n_vars
    config["dataset"]["data_params"]["dataset_path"] = (
        ROOT / config["dataset"]["data_params"]["dataset_path"]
    )
    config["model"]["embedding"]["directory"] = (
        ROOT / config["model"]["embedding"]["directory"]
    )
    data_params = config["dataset"]["data_params"]
    datasets = load_dataset_splits(**data_params, return_dataset=False)

    model, embedding = load_model(config, smiles)
    predictions, _ = compute_pred(
        model,
        datasets["ood"],
        genes_control=datasets["test_control"].genes,
        # genes_control=datasets["training_control"].genes,
        dosages=dosages,
        cell_lines=cell_lines,
        use_DEGs=use_DEGs,
        verbose=verbose,
    )

    predictions = pd.DataFrame.from_dict(predictions, orient="index", columns=["R2"])
    if name_tag:
        predictions["model"] = name_tag
    predictions["genes"] = "degs" if use_DEGs else "all"
    return predictions


# %%
seml_collection = "baseline_comparison"

hash_dict = dict(
    cpa_A549="044c4dba0c8719985c3622834f2cbd58",
    cpa_K562="5ea85d5dd7abd5962d1d3eeff1b8c1ff",
    cpa_MCF7="4d98e7d857f497d870e19c6d12175aaa",
    chemCPA_A549_pretrained="3326f900c45faaf99ca4400f78c58847",
    chemCPA_A549="8779ff45830000c6bc8e22023bb1cb2c",
    chemCPA_K562_pretrained="6388fa373386c11e40dceb5e2e8a113d",
    chemCPA_K562="34fd06018d6e2662ccd5da7a16b57334",
    chemCPA_MCF7_pretrained="2075a457bafdca5948ab671b77757974",
    chemCPA_MCF7="6ad52ba3939397521c5050ca1dd89a4c",
)

# %%
# df_degs = pd.DataFrame.from_dict(drug_r2_baseline_degs, orient="index", columns=["R2"])
# df_degs["model"] = "baseline"
# df_degs["genes"] = "degs"

# df_all = pd.DataFrame.from_dict(drug_r2_baseline_all, orient="index", columns=["R2"])
# df_all["model"] = "baseline"
# df_all["genes"] = "all"

# predictions = [df_degs, df_all]

# %%
predictions.extend(
    [
        get_model_predictions(_hash, name_tag=name_tag, use_DEGs=True)
        for name_tag, _hash in hash_dict.items()
    ]
)
predictions.extend(
    [
        get_model_predictions(_hash, name_tag=name_tag, use_DEGs=False)
        for name_tag, _hash in hash_dict.items()
    ]
)


# %%
def rename_model(str):
    str_list = str.split("_")
    if len(str_list) == 2:
        return str_list[0]
    else:
        assert len(str_list) == 3
        return "_".join([str_list[0], str_list[2]])


# %%
predictions = pd.concat(predictions)
predictions.reset_index(inplace=True)
predictions["cell_type"] = predictions["index"].apply(lambda s: s.split("_")[0])
predictions["condition"] = predictions["index"].apply(lambda s: s.split("_")[1])
predictions["dose"] = predictions["index"].apply(lambda s: s.split("_")[2])
predictions["model_ct"] = predictions["model"]
predictions["model"] = predictions["model"].apply(rename_model)

# %%
predictions

# %%
predictions.groupby(["model", "genes"]).mean()

# %%
predictions.to_parquet("cpa_predictions.parquet")

# %%
predictions.groupby(["model", "genes"]).mean()

# %%
predictions.groupby(["model", "genes"]).std()

# %%
# predictions = pd.read_parquet("baseline_predictions.parquet")

# %%
