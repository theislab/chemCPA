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
# # UMAP Plots of the latent drug embedding space of CCPA

# %%
import logging
import os
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sn
import seml
import torch
import umap.plot

from compert.data import Dataset, canonicalize_smiles, drug_names_to_once_canon_smiles
from compert.embedding import get_chemical_representation
from compert.model import ComPert

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
sn.set_context("poster")

# %% [markdown]
# ## Setting up the model

# %%
model_checkpoints_dir = Path(
    "/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/sweeps/checkpoints"
)
# model is specified via the model hash (this is some seq2seq model that I picked randomly)
model_hash = "14b0557bb351b024fa5abcaae90be37c"
model_checkp = model_checkpoints_dir / (model_hash + ".pt")

# %% [markdown]
# Load the config used to train the model from the mongoDB

# %%
seml_collection = "sciplex_hparam"
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

# %% [markdown]
# Load the dataset that was used by the model (this could be modified to just load the subset of the dataset) and extract the SMILES + pathways.

# %%
perturbation_key = config["dataset"]["data_params"]["perturbation_key"]
smiles_key = config["dataset"]["data_params"]["smiles_key"]
dataset = sc.read(config["dataset"]["data_params"]["dataset_path"])

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

# %% [markdown]
# Load the embedding that was used by the model. The embedding is returned in the same order as the smiles. For Vanilla load the embedding from the `state_dict` later on.

# %%
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

# %%
state_dict, cov_state_dicts, init_args, history = torch.load(model_checkp)
if embedding_model != "vanilla":
    state_dict.pop("drug_embeddings.weight")
model = ComPert(**init_args, drug_embeddings=embedding)
model = model.eval()

# %%
incomp_keys = model.load_state_dict(state_dict, strict=False)
if embedding_model == "vanilla":
    assert len(incomp_keys.unexpected_keys) == 0 and len(incomp_keys.missing_keys) == 0
else:
    # make sure we didn't accidentally load the embedding from the state_dict
    torch.testing.assert_allclose(model.drug_embeddings.weight, embedding.weight)
    assert (
        len(incomp_keys.missing_keys) == 1
        and "drug_embeddings.weight" in incomp_keys.missing_keys
    )
    assert len(incomp_keys.unexpected_keys) == 0

# %%
all_drugs_idx = torch.tensor(list(range(len(embedding.weight))))
# TODO Check whether 1.0 is actually the max dosage.
dosages = torch.ones((len(embedding.weight),))
with torch.no_grad():
    # scaled the drug embeddings using the doser
    scaled_embeddings = model.compute_drug_embeddings_(
        drugs_idx=all_drugs_idx, dosages=dosages
    )
    # apply drug embedder
    transf_embeddings = model.drug_embedding_encoder(scaled_embeddings)

# %%
mapper = umap.UMAP(n_neighbors=8, min_dist=0.15).fit_transform(transf_embeddings)

# %%
# important to use the same ordering of SMILES as was used for getting the embedding!
pathway = [smiles_to_pathway_map[s] for s in canon_smiles_unique_sorted]
sn.scatterplot(x=mapper[:, 0], y=mapper[:, 1], hue=pathway, palette="tab20")
bbox = (1, -0.1)
plt.legend(bbox_to_anchor=bbox)
