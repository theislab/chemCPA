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
# * RDKit:
#      * fine-tuned:      `'c824e42f7ce751cf9a8ed26f0d9e0af7'`
#      * non-pretrained: `'59bdaefb1c1adfaf2976e3fdf62afa21'`
#
# Here everything is in setting 1 (identical gene sets)
#
# **Outputs:**
# * **Figure 2 for RDKit**
# * Figure 5 with DEGs for RDKit
# * Supplement Figures 10 & 11 for RDKit
# ___
# # Imports

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap.plot
from utils import (
    compute_drug_embeddings,
    compute_pred,
    compute_pred_ctrl,
    load_config,
    load_dataset,
    load_model,
    load_smiles,
)

from chemCPA.data import load_dataset_splits
from chemCPA.paths import FIGURE_DIR, ROOT

# %%
BLACK = False

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
# # Load model configs and dataset
# * Define `seml_collection` and `model_hash` to load data and model

# %%
seml_collection = "multi_task"

# # RDKit
model_hash_pretrained = "c824e42f7ce751cf9a8ed26f0d9e0af7"  # Fine-tuned
model_hash_scratch = "59bdaefb1c1adfaf2976e3fdf62afa21"  # Non-pretrained

# %% [markdown]
# ## Load config and SMILES

# %%
config = load_config(seml_collection, model_hash_pretrained)

config["dataset"]["data_params"]["dataset_path"] = (
    ROOT / config["dataset"]["data_params"]["dataset_path"]
)

dataset, key_dict = load_dataset(config)
config["dataset"]["n_vars"] = dataset.n_vars

# %%
canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict, True
)

# %% [markdown]
# Get list of drugs that are ood in `ood_drugs`

# %%
ood_drugs = (
    dataset.obs.condition[
        dataset.obs[config["dataset"]["data_params"]["split_key"]].isin(["ood"])
    ]
    .unique()
    .to_list()
)

# %% [markdown]
# ## Load dataset splits

# %%
config["dataset"]["data_params"]

# %%
data_params = config["dataset"]["data_params"]
datasets = load_dataset_splits(**data_params, return_dataset=False)

# %% [markdown]
# ___
# # Run models
# ## Baseline model

# %%
dosages = [1e1, 1e2, 1e3, 1e4]
cell_lines = ["A549", "K562", "MCF7"]
use_DEGs = True

# %%
drug_r2_baseline_degs, _ = compute_pred_ctrl(
    dataset=datasets["ood"],
    dataset_ctrl=datasets["test_control"],
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)

drug_r2_baseline_all, _ = compute_pred_ctrl(
    dataset=datasets["ood"],
    dataset_ctrl=datasets["test_control"],
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)

# %% [markdown]
# ## Pretrained model

# %%
ood_drugs

# %%
config = load_config(seml_collection, model_hash_pretrained)

config["dataset"]["n_vars"] = dataset.n_vars
config["model"]["embedding"]["directory"] = (
    ROOT / config["model"]["embedding"]["directory"]
)

model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %%
drug_r2_pretrained_degs, _ = compute_pred(
    model_pretrained,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)

drug_r2_pretrained_all, _ = compute_pred(
    model_pretrained,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)

# %% [markdown]
# ## Non-pretrained model

# %%
config = load_config(seml_collection, model_hash_scratch)

config["dataset"]["n_vars"] = dataset.n_vars
config["model"]["embedding"]["directory"] = (
    ROOT / config["model"]["embedding"]["directory"]
)

model_scratch, embedding_scratch = load_model(config, canon_smiles_unique_sorted)

# %%
drug_r2_scratch_degs, _ = compute_pred(
    model_scratch,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)  # non-pretrained

drug_r2_scratch_all, _ = compute_pred(
    model_scratch,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)  # non-pretrained


# %% [markdown]
# # Combine results and create dataframe

# %%
def create_df(drug_r2_baseline, drug_r2_pretrained, drug_r2_scratch):
    df_baseline = pd.DataFrame.from_dict(
        drug_r2_baseline, orient="index", columns=["r2_de"]
    )
    df_baseline["type"] = "baseline"
    df_pretrained = pd.DataFrame.from_dict(
        drug_r2_pretrained, orient="index", columns=["r2_de"]
    )
    df_pretrained["type"] = "pretrained"
    df_scratch = pd.DataFrame.from_dict(
        drug_r2_scratch, orient="index", columns=["r2_de"]
    )
    df_scratch["type"] = "non-pretrained"

    df = pd.concat([df_pretrained, df_scratch, df_baseline])

    df["r2_de"] = df["r2_de"].apply(lambda x: max(x, 0))
    df["cell_line"] = pd.Series(df.index.values).apply(lambda x: x.split("_")[0]).values
    df["drug"] = pd.Series(df.index.values).apply(lambda x: x.split("_")[1]).values
    df["dose"] = pd.Series(df.index.values).apply(lambda x: x.split("_")[2]).values
    df["dose"] = df["dose"].astype(float)

    df["combination"] = df.index.values
    assert (
        df[df.type == "pretrained"].combination
        == df[df.type == "non-pretrained"].combination
    ).all()

    delta = (
        df[df.type == "pretrained"].r2_de - df[df.type == "non-pretrained"].r2_de
    ).values
    df["delta"] = list(delta) + list(-delta) + [0] * len(delta)

    df = df.reset_index()
    return df


# %%
df_degs = create_df(
    drug_r2_baseline_degs, drug_r2_pretrained_degs, drug_r2_scratch_degs
)
df_all = create_df(drug_r2_baseline_all, drug_r2_pretrained_all, drug_r2_scratch_all)

# %% [markdown]
# # Plot Figure 2 with RDKit

# %%
SAVEFIG = True

# %%
fig, ax = plt.subplots(1, 2, figsize=(21, 6))

PROPS = {
    "boxprops": {"edgecolor": "white"},
    "medianprops": {"color": "white"},
    "whiskerprops": {"color": "white"},
    "capprops": {"color": "white"},
    "flierprops": {"markerfacecolor": "lightgrey", "markeredgecolor": "lightgrey"},
}

if BLACK:
    sns.boxplot(
        data=df_all,
        x="dose",
        y="r2_de",
        hue="type",
        whis=1.5,
        ax=ax[0],
        palette="tab10",
        **PROPS
    )  # [(df.r2_de > 0) & (df.delta != 0)]
    sns.boxplot(
        data=df_degs,
        x="dose",
        y="r2_de",
        hue="type",
        whis=1.5,
        ax=ax[1],
        palette="tab10",
        **PROPS
    )  # [(df.r2_de > 0) & (df.delta != 0)]
else:
    sns.boxplot(data=df_all, x="dose", y="r2_de", hue="type", whis=1.5, ax=ax[0])
    sns.boxplot(data=df_degs, x="dose", y="r2_de", hue="type", whis=1.5, ax=ax[1])

for j, axis in enumerate(ax):
    x_labels = axis.get_xticklabels()
    dose_labels = ["0.01", "0.1", "1", "10"]
    [label.set_text(dose_labels[i]) for i, label in enumerate(x_labels)]
    axis.set_xticklabels(x_labels)
    axis.set_ylabel("$E[r^2]$ on DEGs") if j == 1 else None
    axis.set_ylabel("$E[r^2]$ on all genes") if j == 0 else None
    axis.set_xlabel("Dosage in $\mu$M")
    axis.grid(".", color="darkgrey", axis="y")

ax[0].legend().remove()
ax[1].legend(
    title="Model type",
    fontsize=18,
    title_fontsize=24,
    loc="upper left",
    bbox_to_anchor=(1, 1),
)

plt.tight_layout()

if SAVEFIG:
    if BLACK:
        plt.savefig(
            FIGURE_DIR / "RDKit_shared_gene_set_black.pdf", format="pdf"
        )  # BLACK:
    else:
        plt.savefig(FIGURE_DIR / "RDKit_shared_gene_set.pdf", format="pdf")  # WHITE


# %% [markdown]
# ________

# %% [markdown]
# # Additional: Supplement Figure 10/11 and Figure 5

# %%
ood_drugs

# %% [markdown]
# ## Supplement Figure 11 for RDKit
# **Parameters**
# * DEGs
# * Shared genes

# %%
df = df_degs.copy()
df.dose = df.dose * 10

rows, cols = 3, 3
fig, ax = plt.subplots(rows, cols, figsize=(8 * cols, 4.5 * rows))


for i, drug in enumerate(ood_drugs):
    axis = ax[i // cols, i % cols]
    sns.lineplot(
        x="dose",
        y="r2_de",
        data=df[(df.drug == drug)],
        hue="type",
        ax=axis,
        palette="tab10" if BLACK else None,
    )  # & (df.type!="baseline") & (df.cell_line ==cell_line)
    axis.set_title(drug)
    #     ax[i].set()
    axis.set_ylim([0, 1])
    axis.legend().remove()
    axis.set_ylabel("$E[r^2]$ on DEGs")
    axis.set_ylabel("$E[r^2]$ on DEGs")
    axis.set_xlabel("Dosage in $\mu$M")
    axis.set_xscale("log")

ax[0, 2].legend(
    title="Model type",
    fontsize=18,
    title_fontsize=24,
    loc="lower left",
    bbox_to_anchor=(1, 0.2),
)

plt.tight_layout()


if SAVEFIG:
    if BLACK:
        plt.savefig(
            FIGURE_DIR / "all_drug_examples_rdkit_shared_degs_black.png", format="png"
        )
    else:
        plt.savefig(
            FIGURE_DIR / "all_drug_examples_rdkit_shared_degs.png", format="png"
        )


# %% [markdown]
# ## Figure 5 for RDKit (DEGs)

# %%
df = df_degs.copy()
df.dose = df.dose * 10

STACKED = True

if STACKED:
    # Portrait
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
else:
    # Landscape
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)


for i, drug in enumerate(["Dacinostat", "Hesperadin"]):
    sns.lineplot(
        x="dose",
        y="r2_de",
        data=df[(df.drug == drug)],
        hue="type",
        ax=ax[i],
        palette="tab10" if BLACK else None,
    )  # & (df.type!="baseline") & (df.cell_line ==cell_line)
    ax[i].set_title(drug)
    #     ax[i].set()
    ax[i].set_ylim([0, 1])
    ax[i].legend(
        title="Model type", fontsize=18, title_fontsize=24, loc="lower left"
    )  # , bbox_to_anchor=(1, 1)
    ax[i].grid(".", color="darkgrey")

ax[0].set_ylabel("$E[r^2]$ on DEGs")
ax[1].set_ylabel("$E[r^2]$ on DEGs")
ax[1].set_xlabel("Dosage in $\mu$M")
ax[0].set_xlabel("Dosage in $\mu$M")
ax[0].legend().remove()
ax[0].set_xscale("log")
ax[1].set_xscale("log")
plt.tight_layout()

if SAVEFIG:
    if BLACK:
        plt.savefig(
            FIGURE_DIR / "drug_examples_rdkit_shared_degs_black.png", format="png"
        )  # BLACK
    else:
        plt.savefig(FIGURE_DIR / "drug_examples_rdkit_shared_degs.png", format="png")

# %% [markdown]
# ## Supplement Figure 10 for RDKit
#
# **Parameters**
# * All genes
# * Shared genes

# %%
df = df_all.copy()
df.dose = df.dose * 10

rows, cols = 3, 3
fig, ax = plt.subplots(rows, cols, figsize=(8 * cols, 4.5 * rows))


for i, drug in enumerate(ood_drugs):
    axis = ax[i // cols, i % cols]
    sns.lineplot(
        x="dose",
        y="r2_de",
        data=df[(df.drug == drug)],
        hue="type",
        ax=axis,
        palette="tab10" if BLACK else None,
    )  # & (df.type!="baseline") & (df.cell_line ==cell_line)
    axis.set_title(drug)
    #     ax[i].set()
    axis.set_ylim([0, 1])
    axis.legend().remove()
    axis.set_ylabel("$E[r^2]$ on all genes")
    axis.set_ylabel("$E[r^2]$ on all genes")
    axis.set_xlabel("Dosage in $\mu$M")
    axis.set_xscale("log")

ax[0, 2].legend(
    title="Model type",
    fontsize=18,
    title_fontsize=24,
    loc="lower left",
    bbox_to_anchor=(1, 0.2),
)

plt.tight_layout()


if SAVEFIG:
    if BLACK:
        plt.savefig(
            FIGURE_DIR / "all_drug_examples_rdkit_shared_all_genes_black.png",
            format="png",
        )
    else:
        plt.savefig(
            FIGURE_DIR / "all_drug_examples_rdkit_shared_all_genes.png", format="png"
        )


# %% [markdown]
# ## Figure 5 for RDKit (All genes)

# %%
df = df_all.copy()
df.dose = df.dose * 10


STACKED = False

if STACKED:
    # Portrait
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
else:
    # Landscape
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)


for i, drug in enumerate(["Dacinostat", "Hesperadin"]):
    sns.lineplot(
        x="dose",
        y="r2_de",
        data=df[(df.drug == drug)],
        hue="type",
        ax=ax[i],
        palette="tab10" if BLACK else None,
    )  # & (df.type!="baseline") & (df.cell_line ==cell_line)
    ax[i].set_title(drug)
    #     ax[i].set()
    ax[i].set_ylim([0, 1])
    ax[i].legend(
        title="Model type", fontsize=18, title_fontsize=24, loc="lower left"
    )  # , bbox_to_anchor=(1, 1)
    ax[i].grid(".", color="darkgrey")

ax[0].set_ylabel("$E[r^2]$ on all genes")
ax[1].set_ylabel("$E[r^2]$ on all genes")
ax[1].set_xlabel("Dosage in $\mu$M")
ax[0].set_xlabel("Dosage in $\mu$M")
ax[0].legend().remove()
ax[0].set_xscale("log")
ax[1].set_xscale("log")
plt.tight_layout()

if SAVEFIG:
    if BLACK:
        plt.savefig(
            FIGURE_DIR / "drug_examples_rdkit_shared_all_genes_black.png", format="png"
        )  # BLACK
    else:
        plt.savefig(
            FIGURE_DIR / "drug_examples_rdkit_shared_all_genes.png", format="png"
        )

# %% [markdown]
# ___

# %%
