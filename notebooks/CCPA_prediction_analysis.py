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

from compert.data import load_dataset_splits

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
# # Load and analyse model
# * Define `seml_collection` and `model_hash` to load data and model

# %%
seml_collection = "finetuning_num_genes"
# split_ho_pathway, append_ae_layer: true
model_hash_pretrained = (
    "70290e4f42ac4cb19246fafa0b75ccb6"  # "config.model.load_pretrained": true,
)
model_hash_scratch = (
    "00e7e9c7979f90d1325f25f9ff4e3fcb"  # "config.model.load_pretrained": false,
)

# split_ood_finetuning, append_ae_layer: true
model_hash_pretrained = (
    "bd001c8d557edffe9df9e6bf09dc4120"  # "config.model.load_pretrained": true,
)
model_hash_scratch = (
    "6e9d00880375aa450a8e5de60250659f"  # "config.model.load_pretrained": false,
)

seml_collection = "sciplex_hparam"
# rdkit
# split_ood_finetuning, append_ae_layer: false
model_hash_pretrained = (
    "d9ee464c93a0d2d947e9115f8d834f22"  # "config.model.load_pretrained": true,
)
model_hash_scratch = (
    "0a929eab639127e304271036fe478e0b"  # "config.model.load_pretrained": false,
)

# grover
# split_ood_finetuning, append_ae_layer: false
model_hash_pretrained = (
    "bacf2e0b3f9dee9078a97c5216bf7f1c"  # "config.model.load_pretrained": true,
)
model_hash_scratch = (
    "d635df7c184dfff217e09ca93395604b"  # "config.model.load_pretrained": false,
)

# %% [markdown]
# ## Load config

# %%
config = load_config(seml_collection, model_hash_pretrained)
dataset, key_dict = load_dataset(config)
config["dataset"]["n_vars"] = dataset.n_vars

# %% [markdown]
# ### Load smiles info

# %%
canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict
)

# %% [markdown]
# #### Define which drugs should be annotaded with list `ood_drugs`

# %%
ood_drugs = (
    dataset.obs.condition[
        dataset.obs[config["dataset"]["data_params"]["split_key"]].isin(["ood"])
    ]
    .unique()
    .to_list()
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
# ## Load dataset splits

# %%
config["dataset"]["data_params"]

# %%
data_params = config["dataset"]["data_params"]

# #Overwrite split_key
# data_params['split_key'] = 'split_ho_epigenetic'

datasets = load_dataset_splits(**data_params, return_dataset=False)

# %% [markdown]
# ___
# ## Pretrained model

# %%
dosages = [1e3, 1e4]
cell_lines = ["A549", "K562", "MCF7"]  # ["A549", "K562", "MCF7"]
# cell_lines = ['MCF7']

# %%
config = load_config(seml_collection, model_hash_pretrained)
config["dataset"]["n_vars"] = dataset.n_vars
model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %%
drug_r2_pretrained, _ = compute_pred(
    model_pretrained,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    cell_lines=cell_lines,
    dosages=dosages,
)

# %% [markdown]
# ## Non-pretrained model

# %%
config = load_config(seml_collection, model_hash_scratch)
config["dataset"]["n_vars"] = dataset.n_vars
model_scratch, embedding_scratch = load_model(config, canon_smiles_unique_sorted)

# %%
drug_r2_scratch, _ = compute_pred(
    model_scratch,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
)  # non-pretrained

# %%
np.mean([max(v, 0) for v in drug_r2_scratch.values()])

# %%
np.mean([max(v, 0) for v in drug_r2_pretrained.values()])

# %%
from utils import evaluate_r2

# %%
evaluate_r2(model_pretrained, datasets["ood"], datasets["test_control"].genes)

import torch

# %%
from compert.paths import CHECKPOINT_DIR

model_checkp = CHECKPOINT_DIR / (model_hash_pretrained + ".pt")

state_dict, cov_state_dicts, init_args, history = torch.load(model_checkp)

# %%
model_pretrained

# %%
cov_state_dicts

# %%
model_pretrained.covariates_embeddings[0].weight

# %%
[k for k in state_dict.keys() if "emb" in k]

# %%
