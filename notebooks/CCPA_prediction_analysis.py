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
    "ed3bc586a5fcfe3c4dbb0157cd67d0d9"  # "config.model.load_pretrained": false,
)

# split_ood_finetuning, append_ae_layer: true
model_hash_pretrained = (
    "bd001c8d557edffe9df9e6bf09dc4120"  # "config.model.load_pretrained": true,
)
model_hash_scratch = (
    "6e9d00880375aa450a8e5de60250659f"  # "config.model.load_pretrained": false,
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
dosages = [1e1, 1e2, 1e3, 1e4]

# %%
config = load_config(seml_collection, model_hash_pretrained)
config["dataset"]["n_vars"] = dataset.n_vars
model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %%
drug_r2_pretrained, _ = compute_pred(
    model_pretrained,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
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
)  # non-pretrained

# %%
dataset.obs.loc[
    dataset.obs.split_ood_finetuning == "ood", "condition"
].unique().to_list()

# %%
np.mean([max(v, 0) for v in drug_r2_scratch.values()])

# %%
np.mean([max(v, 0) for v in drug_r2_pretrained.values()])

# %% [markdown]
# ____
#
# import json

# %%
import torch

file_name = "test.pt"
torch.save(
    (
        model_scratch.state_dict(),
        # adversary covariates are saved as a list attr on the autoencoder
        # which PyTorch doesn't include in the autoencoder's state dict
        [
            adversary_covariates.state_dict()
            for adversary_covariates in model_scratch.adversary_covariates
        ],
        #                 TODO I haven't checked that this actually works
        [
            covariate_embedding.state_dict()
            for covariate_embedding in model_scratch.covariates_embeddings
        ],
        model_scratch.init_args,
        model_scratch.history,
    ),
    file_name,
)
pjson = lambda s: print(json.dumps(s), flush=True)
pjson({"model_saved": file_name})


# %%
def load_torch_model(file_path, append_ae_layer=False):
    dumped_model = torch.load(file_path)
    if len(dumped_model) == 3:
        # old version
        state_dict, model_config, history = dumped_model
    else:
        # new version
        assert len(dumped_model) == 5
        (
            state_dict,
            adversary_cov_state_dicts,
            cov_embeddings_state_dicts,
            model_config,
            history,
        ) = dumped_model
        assert len(cov_embeddings_state_dicts) == 1
        print("hi")
    #     # sanity check
    #     if append_ae_layer:
    #         assert model_config["num_genes"] < self.datasets["training"].num_genes
    #     else:
    #         assert model_config["num_genes"] == self.datasets["training"].num_genes
    #     assert model_config["use_drugs_idx"]
    #     keys = list(state_dict.keys())
    #     for key in keys:
    #         # remove all components which we will train from scratch
    #         # the drug embedding is saved in the state_dict for some reason, but we don't need it
    #         if key.startswith("adversary_drugs") or key == "drug_embeddings.weight":
    #             state_dict.pop(key)
    #     if self.embedding_model_type == "vanilla":
    #         # for Vanilla CPA, we also train the amortized doser & drug_embedding_encoder anew
    #         keys = list(state_dict.keys())
    #         for key in keys:
    #             if key.startswith("dosers") or key.startswith("drug_embedding_encoder"):
    #                 state_dict.pop(key)
    return state_dict, cov_embeddings_state_dicts, model_config


# %%
model_scratch.drug_embeddings.weight

# %%
append_ae_layer = False
append_layer_width = datasets["training"].num_genes if append_ae_layer else None
in_out_size = (
    model_config["num_genes"] if append_ae_layer else datasets["training"].num_genes
)
# idea: Reconstruct the ComPert model as pretrained (hence the "old" in_out_size)
# then add the append_layer (the "new" in_out_size)

from compert.embedding import get_chemical_representation

embedding = get_chemical_representation(
    smiles=canon_smiles_unique_sorted,
    embedding_model=config["model"]["embedding"]["model"],
    data_dir=config["model"]["embedding"]["directory"],
    device="cuda",
)
state_dict, cov_embeddings_state_dicts, model_config = load_torch_model("test.pt")
append_layer_width = (
    config["dataset"]["n_vars"]
    if (config["model"]["append_ae_layer"] and config["model"]["load_pretrained"])
    else None
)

if config["model"]["embedding"]["model"] != "vanilla":
    state_dict.pop("drug_embeddings.weight")

from compert.model import ComPert

autoencoder = ComPert(
    **model_config, drug_embeddings=embedding, append_layer_width=append_layer_width
)
incomp_keys = autoencoder.load_state_dict(state_dict, strict=False)
for embedding, state_dict in zip(
    autoencoder.covariates_embeddings, cov_embeddings_state_dicts
):
    embedding.load_state_dict(state_dict)
autoencoder.eval()
print(f"INCOMP_KEYS (make sure these contain what you expected):\n{incomp_keys}")

# %%
x = torch.randn((3, 2000))

(model_scratch.encoder.network[0](x) == autoencoder.encoder.network[0](x)).all()

# %%
drug_r2_scratch, _ = compute_pred(
    autoencoder,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
)  # non-pretrained

# %%
