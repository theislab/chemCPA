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

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 60
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
    pathway = []
    drugs = []

    for s in canon_smiles_unique_sorted:
        if s in smiles_to_pathway_map:
            pathway.append(smiles_to_pathway_map[s])
            drugs.append(smiles_to_drug_map[s])
        else:
            pathway.append("other")
            drugs.append("unknown")

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
                fontdict=dict(color="black", alpha=0.5, size=6),
            )
    bbox = (1, 1)

    plt.legend(bbox_to_anchor=bbox)


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
# config = load_config(seml_collection, model_hash_pretrained)
# dataset_, key_dict = load_dataset(config)
# config['dataset']['n_vars'] = dataset.n_vars
# canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(config, dataset_, key_dict, True)
# model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %% [markdown]
# #### Define which drugs should be annotaded with list `ood_drugs`

# %%
ood_drugs = (
    dataset.obs.condition[dataset.obs.split_ho_pathway.isin(["ood"])].unique().to_list()
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
    smiles_to_pw_level2_map,
    smiles_to_drug_map,
    groups=groups_pw2,
    ood_drugs=ood_drugs,
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
    config, dataset, key_dict, return_pathway_map=True
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
    smiles_to_pw_level2_map,
    smiles_to_drug_map,
    groups=groups_pw2,
    ood_drugs=ood_drugs,
)

# %%
# from compert.data import load_dataset_splits

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


# %%

# %%
dataset.obs_names.isin(adata_lincs_small.obs_names).sum()

# %%

# %%

# %%
cond = pd.Series(canon_smiles_unique_sorted).isin(smiles_subset)

# %%

# %% [markdown]
# ### Plot UMAP

# %%
# plot_umap(
#     mapper_pretrained,
#     canon_smiles_unique_sorted,
#     smiles_to_pw_level2_map,
#     smiles_to_drug_map,
#     groups = groups_pw2,
#     ood_drugs=ood_drugs
# )

fig, ax = plt.subplots(figsize=(12, 8))

# drugs = [smiles_to_drug_map[s] for s in canon_smiles_unique_sorted if smiles_to_drug_map[s]]
# pathway = np.where(pd.Series(pathway).isin(groups), pathway, "other")

# cmap = [(0.7,0.7,0.7)]
# cmap.extend(list(plt.get_cmap('tab20').colors))
# cmap = tuple(cmap)
sn.scatterplot(x=mapper_pretrained[:, 0], y=mapper_pretrained[:, 1], ax=ax)

# shift = 0.05
# for i, label in enumerate(drugs):
#     if drugs[i] in ood_drugs:
#         ax.text(x=mapper[i,0]+shift, y=mapper[i,1]+shift,s=label,fontdict=dict(color='black', alpha=0.5, size=6))
# bbox = (1, 1)

# plt.legend(bbox_to_anchor=bbox)

# %% [markdown]
# ____
#
# from rdkit import Chem

# %%
from compert.paths import PROJECT_DIR

# rdkit, LINCS
seml_collection = "lincs_rdkit_hparam"
model_hash_rdkit = "4f061dbfc7af05cf84f06a724b0c8563"
model_hash_grover = "ff420aea264fca7668ecb147f60762a1"

# %%
config = load_config(seml_collection, model_hash_grover)

# %%
# dataset, key_dict = load_dataset(config)

# %%
key_dict["perturbation_key"] = "pert_id"
key_dict["smiles_key"] = "canonical_smiles"

# %%
config["dataset"]["n_vars"] = dataset.n_vars
config["model"]["append_ae_layer"] = False

# %%
dataset

# %% tags=[]
# canon_smiles_unique_sorted, smiles_to_drug_map = load_smiles(config, dataset, key_dict)

# %%
adata_lincs_small = (
    sc.read(PROJECT_DIR / "datasets" / "lincs_small_.h5ad")
    if adata_lincs_small is None
    else adata_lincs_small
)

# obs_lincs_small = pd.Series(adata_lincs_small.obs.index).apply(lambda x: "-".join(x.split("-")[:-1]))
# smiles_subset = dataset.obs.loc[dataset.obs_names.isin(obs_lincs_small), 'canonical_smiles'].unique().to_list()
# smiles_subset = [Chem.CanonSmiles(s) for s in smiles_subset]

# %%
smiles_list = list(set(canon_smiles_unique_sorted + smiles_subset))
model_pretrained, embedding_pretrained = load_model(config, smiles_list)

# %%
transf_embeddings_pretrained = compute_drug_embeddings(
    model_pretrained, embedding_pretrained
)  # [cond]

mapper_pretrained = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(
    transf_embeddings_pretrained
)

# %%
sciplex_drugs = dataset_.obs.condition.unique().to_list()

# %%
plot_umap(
    mapper_pretrained,
    smiles_list,
    smiles_to_pathway_map,
    smiles_to_drug_map,
    #     groups = groups_pw2,
    ood_drugs=sciplex_drugs,
)

# %%
