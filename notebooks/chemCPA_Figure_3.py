# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
# ---

# %% [markdown]
# **Requirements:**
# * Trained models
# * GROVER:
#      * fine-tuned:      `'a50dc68191a3776694ce8f34ad55e7e0'`
#      * non-pretrained: `'0807497c5407f4e0c8a52207f36a185f'`
#
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
import seaborn as sns
import torch
import umap.plot
from utils import (
    compute_drug_embeddings,
    compute_pred,
    load_config,
    load_dataset,
    load_model,
    load_smiles,
)

from chemCPA.paths import FIGURE_DIR, ROOT

# %%
BLACK = False
SAVEFIG = True

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
    ax=None,
):
    # important to use the same ordering of SMILES as was used for getting the embedding!
    if ax == None:
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

    #     cmap = [(0.7,0.7,0.7)]
    # #     cmap.extend(list(plt.get_cmap('tab20').colors))
    # #     cmap = tuple(cmap)

    #     sns.scatterplot(x=mapper[:,0], y=mapper[:,1], hue=pathway, palette=cmap, ax=ax)
    cond = pathway != "other"
    sns.scatterplot(
        x=mapper[cond, 0],
        y=mapper[cond, 1],
        hue=pathway[cond],
        ax=ax,
        palette="tab10" if BLACK else None,
    )
    sns.scatterplot(
        x=mapper[~cond, 0], y=mapper[~cond, 1], ax=ax, color="grey", alpha=0.3
    )

    shift = 0.05
    for i, label in enumerate(drugs):
        if drugs[i] in ood_drugs:
            ax.text(
                x=mapper[i, 0] + shift,
                y=mapper[i, 1] + shift,
                s=label,
                fontdict=dict(
                    color="white" if BLACK else "black", alpha=1, size=12, weight=600
                ),
                bbox=dict(facecolor="black" if BLACK else "lightgrey", alpha=0.3),
            )


# %% [markdown]
# # Load and analyse model
# * Define `seml_collection` and `model_hash` to load data and model

# %%
seml_collection = "multi_task"

model_hash_pretrained_rdkit = "c824e42f7ce751cf9a8ed26f0d9e0af7"  # Fine-tuned
model_hash_scratch_rdkit = "59bdaefb1c1adfaf2976e3fdf62afa21"  # Non-pretrained

model_hash_pretrained_grover = "c30016a7469feb78a8ee9ebb18ed9b1f"  # Fine-tuned
model_hash_scratch_grover = "60e4b40e8d67bff2d5efc5e22e265820"  # Non-pretrained

model_hash_pretrained_jtvae = "915345a522c29fa709b995d6149083b9"  # Fine-tuned
model_hash_scratch_jtvae = "934c89b742a6309ad6bb2e1cf90c5e50"  # Non-pretrained

# %%
model_hash_pretrained = model_hash_pretrained_rdkit

# %% [markdown]
# ___
# ## Pretrained model

# %% [markdown]
# ### Load model

# %%
config = load_config(seml_collection, model_hash_pretrained)
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
dataset

# %%
config

# %% [markdown]
# #### Define which drugs should be annotaded with list `ood_drugs`

# %%
ood_drugs = (
    dataset.obs.condition[dataset.obs.split_ood_multi_task.isin(["ood"])]
    .unique()
    .to_list()
)

# %%
ood_drugs

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
for s, pw in smiles_to_pw_level2_map.items():
    if pw == "Histone deacetylation":
        smiles_to_pathway_map[s] = pw

# %%
groups = [
    "Histone deacetylation",
    "Tyrosine kinase signaling",
    "Cell cycle regulation",
    "DNA damage & DNA repair",
]

# groups_pw2 = [pw2 for pw in groups for pw2 in pw1_to_pw2[pw]]
# groups_pw2

# %% [markdown]
# ### Compute UMAP

# %%
transf_embeddings_pretrained_high = compute_drug_embeddings(
    model_pretrained, embedding_pretrained, dosage=1e4
)
mapper_pretrained_high = umap.UMAP(
    n_neighbors=25, min_dist=1, spread=2, metric="euclidean"
).fit_transform(transf_embeddings_pretrained_high)

transf_embeddings_pretrained_low = compute_drug_embeddings(
    model_pretrained, embedding_pretrained, dosage=10
)
mapper_pretrained_low = umap.UMAP(n_neighbors=25, min_dist=1, spread=2).fit_transform(
    transf_embeddings_pretrained_low
)

# %% [markdown]
# ### Plot UMAP

# %%
fig, ax = plt.subplots(1, 2, figsize=(21, 5))

plot_umap(
    mapper_pretrained_high,
    canon_smiles_unique_sorted,
    smiles_to_pathway_map,
    smiles_to_drug_map,
    groups=groups,
    ood_drugs=ood_drugs,
    ax=ax[1],
)

plot_umap(
    mapper_pretrained_low,
    canon_smiles_unique_sorted,
    smiles_to_pathway_map,
    smiles_to_drug_map,
    groups=groups,
    ood_drugs=ood_drugs,
    ax=ax[0],
)
ax[0].set(xticklabels=[], yticklabels=[])
ax[0].set_xlabel(f"UMAP of $z_d$ for a dosage of $10\,$nM")

ax[1].set(xticklabels=[], yticklabels=[])
ax[1].set_xlabel(f"UMAP of $z_d$ for a dosage of $10\,\mu$M")

ax[0].grid(False)
ax[1].grid(False)
ax[0].get_legend().remove()
ax[1].legend(
    title="Pathway",
    fontsize=18,
    title_fontsize=22,
    loc="upper left",
    bbox_to_anchor=(1, 1),
)
plt.tight_layout()

if SAVEFIG:
    if BLACK:
        plt.savefig(FIGURE_DIR / "umap_drug_embedding_black.png", format="png")
    else:
        plt.savefig(FIGURE_DIR / "umap_drug_embedding.png", format="png")

# %% [markdown]
# ___

# %%
