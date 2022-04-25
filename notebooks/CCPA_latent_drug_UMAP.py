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

# %% [markdown]
# **Requirements:**
# * Trained models
#
# **Outputs:**
# * none
# ___
# # Imports

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sn
import umap.plot
from utils import (
    compute_drug_embeddings,
    compute_pred,
    load_config,
    load_dataset,
    load_model,
    load_smiles,
)

from chemCPA.paths import FIGURE_DIR

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
sn.set_context("poster")


# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Plotting function

# %%
def plot_umap(
    mapper,
    canon_smiles_unique_sorted,
    smiles_to_pathway_map,
    smiles_to_drug_map,
    groups=[
        "Epigenetic regulation",
        "Tyrosine kinase signaling",
        "Cell cycle regulation",
    ],
    ood_drugs=[],
):
    # important to use the same ordering of SMILES as was used for getting the embedding!
    fig, ax = plt.subplots(figsize=(12, 8))

    # groups=["Tyrosine kinase signaling"]
    pathway = [
        smiles_to_pathway_map[s]
        for s in canon_smiles_unique_sorted
        if smiles_to_pathway_map[s]
    ]
    drugs = [
        smiles_to_drug_map[s]
        for s in canon_smiles_unique_sorted
        if smiles_to_drug_map[s]
    ]
    pathway = np.where(pd.Series(pathway).isin(groups), pathway, "other")

    cmap = [(0.7, 0.7, 0.7)]
    cmap.extend(list(plt.get_cmap("tab20").colors))
    cmap = tuple(cmap)
    sn.scatterplot(x=mapper[:, 0], y=mapper[:, 1], hue=pathway, palette=cmap, ax=ax)

    shift = 0.05
    for i, label in enumerate(drugs):
        if drugs[i] in ood_drugs:
            ax.text(
                x=mapper[i, 0] + shift,
                y=mapper[i, 1] + shift,
                s=label,
                fontdict=dict(color="black", alpha=0.9, size=12),
            )
    bbox = (1, 1)

    #     plt.legend(bbox_to_anchor=bbox)
    plt.legend(loc="best")


# %% [markdown]
# # Load and analyse model
# * Define `seml_collection` and `model_hash` to load data and model

# %%
seml_collection = "finetuning_num_genes"

model_hash_pretrained = (
    "70290e4f42ac4cb19246fafa0b75ccb6"  # "config.model.load_pretrained": true,
)
model_hash_scratch = (
    "ed3bc586a5fcfe3c4dbb0157cd67d0d9"  # "config.model.load_pretrained": false,
)

# %% [markdown]
# ___
# ## Pretrained model

# %% [markdown]
# ### Load model

# %%
config = load_config(seml_collection, model_hash_pretrained)
dataset, key_dict = load_dataset(config)
config["dataset"]["n_vars"] = dataset.n_vars
canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict
)
model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %% [markdown]
# #### Define which drugs should be annotaded with list `ood_drugs`

# %%
split_key = config["dataset"]["data_params"]["split_key"]
ood_drugs = (
    dataset.obs.condition[dataset.obs[split_key].isin(["ood"])].unique().to_list()
)

# %% [markdown]
# #### Get pathway level 2 annotation for clustering of drug embeddings

# %%
smiles_to_pw_level2_map = {}
pw1_to_pw2 = {}

for (drug, pw1, pw2), df in dataset.obs.groupby(
    ["SMILES", "pathway_level_1", "pathway_level_2"]
):
    smiles_to_pw_level2_map[drug] = pw2
    if pw1 in pw1_to_pw2:
        pw1_to_pw2[pw1].add(pw2)
    else:
        pw1_to_pw2[pw1] = {pw2}

# %%
groups = ["Epigenetic regulation"]

groups_pw2 = [pw2 for pw in groups for pw2 in pw1_to_pw2[pw]]
groups_pw2

# %% [markdown]
# ### Compute UMAP

# %%
transf_embeddings_pretrained = compute_drug_embeddings(
    model_pretrained, embedding_pretrained
)
mapper_pretrained = umap.UMAP(n_neighbors=25, min_dist=0.5).fit_transform(
    transf_embeddings_pretrained
)

# %% [markdown]
# ### Plot UMAP

# %%
plot_umap(
    mapper_pretrained,
    canon_smiles_unique_sorted,
    smiles_to_pathway_map,
    smiles_to_drug_map,
    groups=[
        "Epigenetic regulation",
        "Tyrosine kinase signaling",
        "Cell cycle regulation",
    ],
    ood_drugs=ood_drugs,
)

plt.savefig(
    FIGURE_DIR / "UMAP_embedding_pretrained.eps", format="eps", bbox_inches="tight"
)

# %% [markdown]
# ___
# ___
# ## Non-pretrained model

# %% [markdown]
# ### Load model

# %%
config = load_config(seml_collection, model_hash_scratch)
dataset, key_dict = load_dataset(config)
config["dataset"]["n_vars"] = dataset.n_vars
canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict
)
model_scratch, embedding_scratch = load_model(config, canon_smiles_unique_sorted)

# %% [markdown]
# ### Compute UMAP

# %%
transf_embeddings_scratch = compute_drug_embeddings(model_scratch, embedding_scratch)
mapper_scratch = umap.UMAP(n_neighbors=25, min_dist=0.5).fit_transform(
    transf_embeddings_scratch
)

# %% [markdown]
# ### Plot UMAP

# %%
plot_umap(
    mapper_scratch,
    canon_smiles_unique_sorted,
    smiles_to_pathway_map,
    smiles_to_drug_map,
    groups=[
        "Epigenetic regulation",
        "Tyrosine kinase signaling",
        "Cell cycle regulation",
    ],
    ood_drugs=ood_drugs,
)

plt.savefig(
    FIGURE_DIR / "UMAP_embedding_scratch.eps", format="eps", bbox_inches="tight"
)

# %%
# from chemCPA.data import load_dataset_splits

# data_params = config['dataset']['data_params']
# data_params['split_key'] = 'split_ho_epigenetic'
# datasets = load_dataset_splits(**data_params, return_dataset=False)

# %%
# datasets

# %%
# predictions_dict = compute_pred(model, datasets['training'])

# %%
# predictions_dict = compute_pred(model, datasets['ood']) # non-pretrained

# %%
