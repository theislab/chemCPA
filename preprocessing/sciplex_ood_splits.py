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
# **Requires**
# * `'trapnell_cpa_lincs_genes.h5ad'`
# * `'trapnell_cpa.h5ad'`
#
# **Output**
# * `'sciplex_complete.h5ad'`
# * `'sciplex_complete_lincs_genes.h5ad'`
#
# ## Imports

# %%
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sfaira
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

from chemCPA.paths import DATA_DIR, PROJECT_DIR

IPythonConsole.ipython_useSVG = False
matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
os.getcwd()
pd.set_option("display.max_columns", 100)
sc.set_figure_params(dpi=80, frameon=False)
sc.logging.print_header()
sns.set_context("poster")

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Load data

# %%
adata_sciplex_lincs_genes = sc.read(
    PROJECT_DIR / "datasets" / "trapnell_cpa_lincs_genes.h5ad"
)
adata_sciplex = sc.read(PROJECT_DIR / "datasets" / "trapnell_cpa.h5ad")

# %%
sc.pp.highly_variable_genes(adata_sciplex_lincs_genes, n_top_genes=977)

# %%
sc.pp.highly_variable_genes(adata_sciplex, n_top_genes=2000)


# %% [markdown]
# ## Compute UMAP

# %%
# ! mamba list | grep pynndescent
# ! mamba list | grep umap

# %%
def preprocess_adata(adata, n_comps=25, n_neighbors=50):
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric="cosine")
    sc.tl.umap(adata, min_dist=0.1)
    return None


def preprocess_adata_subset_type(adata, cell_type, n_comps=25):
    adata_new = adata[adata.obs.cell_type == cell_type].copy()
    sc.pp.pca(adata_new, n_comps=n_comps)
    sc.pp.neighbors(adata_new, n_neighbors=50, metric="cosine")
    sc.tl.umap(adata_new, min_dist=0.1)
    return adata_new


# %%
preprocess_adata(adata_sciplex, n_comps=25, n_neighbors=50)

# %%
preprocess_adata(adata_sciplex_lincs_genes)

# %% [markdown]
# ## Load or create subsetted adata objects

# %%
fname = PROJECT_DIR / "datasets" / "adata_MCF7.h5ad"
if not fname.exists():
    adata_MCF7 = preprocess_adata_subset_type(adata_sciplex, "MCF7")
    sc.write(fname, adata_MCF7)
else:
    adata_MCF7 = sc.read(fname)

fname = PROJECT_DIR / "datasets" / "adata_MCF7_lincs_genes.h5ad"
if not fname.exists():
    adata_MCF7_lincs_genes = preprocess_adata_subset_type(
        adata_sciplex_lincs_genes, "MCF7"
    )
    sc.write(fname, adata_MCF7_lincs_genes)
else:
    adata_MCF7_lincs_genes = sc.read(fname)

fname = PROJECT_DIR / "datasets" / "adata_K562.h5ad"
if not fname.exists():
    adata_K562 = preprocess_adata_subset_type(adata_sciplex, "K562")
    sc.write(fname, adata_K562)
else:
    adata_K562 = sc.read(fname)

fname = PROJECT_DIR / "datasets" / "adata_K562_lincs_genes.h5ad"
if not fname.exists():
    adata_K562_lincs_genes = preprocess_adata_subset_type(
        adata_sciplex_lincs_genes, "K562"
    )
    sc.write(fname, adata_K562_lincs_genes)
else:
    adata_K562_lincs_genes = sc.read(fname)

fname = PROJECT_DIR / "datasets" / "adata_A549.h5ad"
if not fname.exists():
    adata_A549 = preprocess_adata_subset_type(adata_sciplex, "A549")
    sc.write(fname, adata_A549)
else:
    adata_A549 = sc.read(fname)

fname = PROJECT_DIR / "datasets" / "adata_A549_lincs_genes.h5ad"
if not fname.exists():
    adata_A549_lincs_genes = preprocess_adata_subset_type(
        adata_sciplex_lincs_genes, "A549"
    )
    sc.write(fname, adata_A549_lincs_genes)
else:
    adata_A549_lincs_genes = sc.read(fname)

# %% [markdown]
# ## Plot pathways for different cell lines

# %%
pathways = [
    #     'Antioxidant',
    "Apoptotic regulation",
    "Cell cycle regulation",
    "DNA damage & DNA repair",
    "Epigenetic regulation",
    #     'Focal adhesion signaling',
    "HIF signaling",
    "JAK/STAT signaling",
    #     'Metabolic regulation',
    #     'Neuronal signaling',
    "Nuclear receptor signaling",
    #     'Other',
    "PKC signaling",
    "Protein folding & Protein degradation",
    #     'TGF/BMP signaling',
    "Tyrosine kinase signaling",
    #     'Vehicle'
]


# %% [markdown]
# ### LINCS genes

# %%
dose = 1e4

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sc.pl.umap(
    adata_A549_lincs_genes[adata_A549_lincs_genes.obs.dose == dose].copy(),
    color="pathway_level_1",
    groups=pathways,
    legend_fontsize="xx-small",
    show=False,
    ax=ax[0],
)
sc.pl.umap(
    adata_K562_lincs_genes[adata_K562_lincs_genes.obs.dose == dose].copy(),
    color="pathway_level_1",
    groups=pathways,
    legend_fontsize="xx-small",
    show=False,
    ax=ax[1],
)
sc.pl.umap(
    adata_MCF7_lincs_genes[adata_MCF7_lincs_genes.obs.dose == dose].copy(),
    color="pathway_level_1",
    groups=pathways,
    legend_fontsize="xx-small",
    show=False,
    ax=ax[2],
)
ax[0].get_legend().remove()
ax[1].get_legend().remove()
plt.tight_layout()

# %% [markdown]
# ### All genes

# %%
dose = 1e4

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sc.pl.umap(
    adata_A549[adata_A549.obs.dose == dose].copy(),
    color="pathway_level_1",
    groups=pathways,
    legend_fontsize="xx-small",
    show=False,
    ax=ax[0],
)
sc.pl.umap(
    adata_K562[adata_K562.obs.dose == dose].copy(),
    color="pathway_level_1",
    groups=pathways,
    legend_fontsize="xx-small",
    show=False,
    ax=ax[1],
)
sc.pl.umap(
    adata_MCF7[adata_MCF7.obs.dose == dose].copy(),
    color="pathway_level_1",
    groups=pathways,
    legend_fontsize="xx-small",
    show=False,
    ax=ax[2],
)
ax[0].get_legend().remove()
ax[1].get_legend().remove()
plt.tight_layout()

# %%
dose = 1e4
cond_A549 = adata_A549_lincs_genes.obs.dose == dose
cond_K562 = adata_K562_lincs_genes.obs.dose == dose
cond_MCF7 = adata_MCF7_lincs_genes.obs.dose == dose

cols = 3
rows = len(pathways)
size = 7
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))

for i, pw in enumerate(pathways):
    sc.pl.umap(
        adata_A549_lincs_genes[cond_A549].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 0],
        legend_fontsize="xx-small",
        size=size,
        label=None,
    )

    sc.pl.umap(
        adata_K562_lincs_genes[cond_K562].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 1],
        legend_fontsize="xx-small",
        size=size,
        label=None,
    )
    sc.pl.umap(
        adata_MCF7_lincs_genes[cond_MCF7].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 2],
        legend_fontsize="xx-small",
        size=size,
    )
    ax[i, 0].get_legend().remove()
    ax[i, 1].get_legend().remove()

plt.tight_layout()

# %% [markdown]
# ## Distribution of pathways for perturbations with maximal dosage

# %% [markdown]
# ### Lincs genes

# %%
dose = 1e4
cond_A549 = adata_A549_lincs_genes.obs.dose == dose
cond_K562 = adata_K562_lincs_genes.obs.dose == dose
cond_MCF7 = adata_MCF7_lincs_genes.obs.dose == dose

cols = 3
rows = len(pathways)
size = 7
fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))

for i, pw in enumerate(pathways):
    sc.pl.umap(
        adata_A549_lincs_genes[cond_A549].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 0],
        legend_fontsize="xx-small",
        size=size,
    )
    sc.pl.umap(
        adata_K562_lincs_genes[cond_K562].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 1],
        legend_fontsize="xx-small",
        size=size,
    )
    sc.pl.umap(
        adata_MCF7_lincs_genes[cond_MCF7].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 2],
        legend_fontsize="xx-small",
        size=size,
    )
    ax[i, 0].get_legend().remove()
    ax[i, 1].get_legend().remove()

plt.tight_layout()

# %% [markdown]
# ### All genes

# %%
dose = 1e4
cond_A549 = adata_A549.obs.dose == dose
cond_K562 = adata_K562.obs.dose == dose
cond_MCF7 = adata_MCF7.obs.dose == dose

cols = 3
rows = len(pathways)
size = 7
fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))

for i, pw in enumerate(pathways):
    sc.pl.umap(
        adata_A549[cond_A549].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 0],
        legend_fontsize="xx-small",
        size=size,
    )
    sc.pl.umap(
        adata_K562[cond_K562].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 1],
        legend_fontsize="xx-small",
        size=size,
    )
    sc.pl.umap(
        adata_MCF7[cond_MCF7].copy(),
        color="pathway_level_1",
        groups=pw,
        show=False,
        ax=ax[i, 2],
        legend_fontsize="xx-small",
        size=size,
    )
    ax[i, 0].get_legend().remove()
    ax[i, 1].get_legend().remove()

plt.tight_layout()

# %% [markdown]
# ## Distribution of drugs with maximal dosage

# %% [markdown]
# ### Identifying significant drugs for perturbations
#
# The relevant information is take from Fig.S6 from the [supplement material](https://www.science.org/doi/suppl/10.1126/science.aax6234/suppl_file/aax6234-srivatsan-sm.pdf) of the orignial [paper](https://www.science.org/doi/full/10.1126/science.aax6234)

# %%
epigenetic_drugs = [
    "Dacinostat",
    "Quisinostat",
    "CUDC-907",
    "Abexinostat",
    "Panobinostat",
    "Belinostat",
    "Givinostat",
    "Mocetinostat",  # no_ood
    "Pracinostat",  # no_ood
    "AR-42",
    "Entinostat",  # no_ood
    "Tucidinostat",  # no_ood
    "Tacedinaline",  # no_ood
    "Trichostatin",
    "CUDC-101",
    "M344",
    "Resminostat",
]

dna_damage_drugs = [
    "Raltitrexed",  # no_ood
    "Pirarubicin",
]

cell_cycle_drugs = [
    "Epothilone",
    "Patupilone",  # no_ood
    "Flavopiridol",
    "Hesperadin",
    "GSK1070916",  # no_ood
]

apoptosis_drugs = ["JNJ-26854165"]  # no_ood

tyrosine_drugs = ["Trametinib", "TAK-901", "Dasatinib"]  # no_ood  # no_ood  # no_ood

protein_drugs = ["Alvespimycin", "Tanespimycin", "Luminespib"]


# Create list of potential ood_drugs
ood_drugs = ["control"]
ood_drugs.extend(epigenetic_drugs)
ood_drugs.extend(dna_damage_drugs)
ood_drugs.extend(cell_cycle_drugs)
ood_drugs.extend(apoptosis_drugs)
ood_drugs.extend(tyrosine_drugs)
ood_drugs.extend(protein_drugs)

# %% [markdown]
# Create pathway dependent colour palette for more informative plotting

# %%
grey_palette = dict(zip(adata_A549.obs.condition.cat.categories.values, 188 * ["red"]))

colours = (
    ["grey"]
    + len(epigenetic_drugs) * ["008fd5"]
    + len(dna_damage_drugs) * ["fc4f30"]
    + len(cell_cycle_drugs) * ["e5ae38"]
    + len(apoptosis_drugs) * ["6d904f"]
    + len(tyrosine_drugs) * ["8b8b8b"]
    + len(protein_drugs) * ["810f7c"]
)

palette = dict(zip(ood_drugs, colours))
for drug, colour in palette.items():
    if drug == "control":
        continue
    if isinstance(colour, str):
        grey_palette[drug] = "#" + colour

# %% [markdown]
# ### LINCS Genes

# %%
dose = 1e4
cond_A549 = (adata_A549_lincs_genes.obs.dose == dose) | (
    adata_A549_lincs_genes.obs.condition == "control"
)
cond_K562 = (adata_K562_lincs_genes.obs.dose == dose) | (
    adata_K562_lincs_genes.obs.condition == "control"
)
cond_MCF7 = (adata_MCF7_lincs_genes.obs.dose == dose) | (
    adata_MCF7_lincs_genes.obs.condition == "control"
)

cols = 3
rows = len(ood_drugs)
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))

for i, drug in enumerate(ood_drugs):
    sc.pl.umap(
        adata_A549_lincs_genes[cond_A549].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 0],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    sc.pl.umap(
        adata_K562_lincs_genes[cond_K562].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 1],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    sc.pl.umap(
        adata_MCF7_lincs_genes[cond_MCF7].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 2],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    ax[i, 0].get_legend().remove()
    ax[i, 1].get_legend().remove()

plt.tight_layout()


# %% [markdown]
# ### All genes

# %%
dose = 1e4
cond_A549 = (adata_A549.obs.dose == dose) | (adata_A549.obs.condition == "control")
cond_K562 = (adata_K562.obs.dose == dose) | (adata_K562.obs.condition == "control")
cond_MCF7 = (adata_MCF7.obs.dose == dose) | (adata_MCF7.obs.condition == "control")

cols = 3
rows = len(ood_drugs)
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))

for i, drug in enumerate(ood_drugs):
    sc.pl.umap(
        adata_A549[cond_A549].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 0],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    sc.pl.umap(
        adata_K562[cond_K562].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 1],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    sc.pl.umap(
        adata_MCF7[cond_MCF7].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 2],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    ax[i, 0].get_legend().remove()
    ax[i, 1].get_legend().remove()

plt.tight_layout()

# %% [markdown]
# ## Create data split

# %% [markdown]
# ### Divide into `'train'`, `'test'`, and `'ood'`

# %%
validation_drugs = [
    "Alvespimycin",
    "Luminespib",
    "Epothilone",
    "Flavopiridol",
    "Quisinostat",
    "Abexinostat",
    "Panobinostat",
    "AR-42",
    "Trichostatin",
    "M344",
    "Resminostat",
    "Belinostat",  # ood
    "Mocetinostat",  # no_ood
    "Pracinostat",  # no_ood
    "Entinostat",  # no_ood
    "Tucidinostat",  # no_ood
    "Tacedinaline",  # no_ood
    "Patupilone",  # no_ood
    "GSK1070916",  # no_ood
    "JNJ-26854165" "TAK-901",  # no_ood  # no_ood
    "Dasatinib",  # no_ood
]


ood_drugs = [
    "Dacinostat",  # ood
    "CUDC-907",  # ood
    "Givinostat",  # ood
    "CUDC-101",  # ood
    "Pirarubicin",  # ood
    "Hesperadin",  # ood
    "Tanespimycin",  # ood
    "Trametinib",  # ood
    "Raltitrexed",  # no_ood
]

additional_validation_drugs = [
    "YM155",  # apoptosis
    "Barasertib",  # cell cycle
    "Fulvestrant",  # nuclear receptor
    "Nintedanib",  # tyrosine
    "Rigosertib",  # tyrosine
    "BMS-754807",  # tyrosine
    "KW-2449",  # tyrosine
    "Crizotinib",  # tyrosin
    "ENMD-2076",  # cell cycle
    "Alisertib",  # cell cycle
    "JQ1",  # epigenetic
]

validation_drugs.extend(additional_validation_drugs)

# %% [markdown]
# ### Plot additonal validation drugs on all genes data

# %%
dose = 1e4
cond_A549 = (adata_A549.obs.dose == dose) | (adata_A549.obs.condition == "control")
cond_K562 = (adata_K562.obs.dose == dose) | (adata_K562.obs.condition == "control")
cond_MCF7 = (adata_MCF7.obs.dose == dose) | (adata_MCF7.obs.condition == "control")

cols = 3
rows = len(additional_validation_drugs)
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))

for i, drug in enumerate(additional_validation_drugs):
    sc.pl.umap(
        adata_A549[cond_A549].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 0],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    sc.pl.umap(
        adata_K562[cond_K562].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 1],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    sc.pl.umap(
        adata_MCF7[cond_MCF7].copy(),
        color="condition",
        groups=drug,
        show=False,
        ax=ax[i, 2],
        legend_fontsize="xx-small",
        palette=grey_palette,
        size=20,
    )
    ax[i, 0].get_legend().remove()
    ax[i, 1].get_legend().remove()

plt.tight_layout()

# %%
# train
adata_sciplex.obs["split_ood_finetuning"] = "train"

# ood
adata_sciplex.obs.loc[
    adata_sciplex.obs.condition.isin(ood_drugs), "split_ood_finetuning"
] = "ood"

# test
validation_cond = (adata_sciplex.obs.condition.isin(validation_drugs)) & (
    adata_sciplex.obs.dose.isin([1e3, 1e4])
)
val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.4, copy=True).obs.index
adata_sciplex.obs.loc[val_idx, "split_ood_finetuning"] = "test"

validation_cond = (adata_sciplex.obs.condition.isin(validation_drugs)) & (
    adata_sciplex.obs.dose.isin([1e1, 1e2])
)
val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.2, copy=True).obs.index
adata_sciplex.obs.loc[val_idx, "split_ood_finetuning"] = "test"

validation_cond = adata_sciplex.obs.split_ood_finetuning == "train"
val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.04, copy=True).obs.index
adata_sciplex.obs.loc[val_idx, "split_ood_finetuning"] = "test"

validation_cond = (adata_sciplex.obs.split_ood_finetuning == "train") & (
    adata_sciplex.obs.control.isin([1])
)
val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.05, copy=True).obs.index
adata_sciplex.obs.loc[val_idx, "split_ood_finetuning"] = "test"

# %%
adata_sciplex.obs.condition.value_counts()

# %%
adata_sciplex.obs["split_ood_finetuning"].value_counts()

# %%
pd.crosstab(
    adata_sciplex.obs["split_ood_finetuning"],
    adata_sciplex.obs["condition"][adata_sciplex.obs["condition"].isin(ood_drugs)],
)

# %%
pd.crosstab(
    adata_sciplex.obs.loc[adata_sciplex.obs["split_ood_finetuning"] == "ood", "dose"],
    adata_sciplex.obs.loc[
        adata_sciplex.obs["split_ood_finetuning"] == "ood", "condition"
    ],
)

# %%
pd.crosstab(
    adata_sciplex.obs["split_ood_finetuning"],
    adata_sciplex.obs["condition"][
        adata_sciplex.obs["condition"].isin(validation_drugs)
    ],
)

# %%
pd.crosstab(
    adata_sciplex.obs.loc[adata_sciplex.obs["split_ood_finetuning"] == "test", "dose"],
    adata_sciplex.obs.loc[
        adata_sciplex.obs["split_ood_finetuning"] == "test", "condition"
    ],
)

# %% [markdown]
# ## Add epigenetic holdout split

# %%
# train
adata_sciplex.obs["split_ood_finetuning"] = "train"

# ood
adata_sciplex.obs.loc[
    adata_sciplex.obs.condition.isin(ood_drugs), "split_ood_finetuning"
] = "ood"


validation_cond = adata_sciplex.obs.split_ood_finetuning == "train"
val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.04, copy=True).obs.index
adata_sciplex.obs.loc[val_idx, "split_ood_finetuning"] = "test"

validation_cond = (adata_sciplex.obs.split_ood_finetuning == "train") & (
    adata_sciplex.obs.control.isin([1])
)
val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.05, copy=True).obs.index
adata_sciplex.obs.loc[val_idx, "split_ood_finetuning"] = "test"

# %%
adata_sciplex.obs.condition.isin(epigenetic_drugs_all).sum()

# %%
from sklearn.model_selection import train_test_split

epigenetic_drugs_all = adata_sciplex.obs.condition[
    adata_sciplex.obs.pathway_level_1 == "Epigenetic regulation"
].unique()

epigenetic_drugs = [
    "Dacinostat",
    "Quisinostat",
    "CUDC-907",
    "Abexinostat",
    "Panobinostat",
    "Belinostat",
    "Givinostat",
    "AR-42",
    "Trichostatin",
    "CUDC-101",
    "M344",
    "Resminostat",
    "Entinostat",  # no_ood
    "Tucidinostat",  # no_ood
    "Tacedinaline",  # no_ood
    "Mocetinostat",  # no_ood
    "Pracinostat",  # no_ood
]

ood_epi_drugs = [
    "Dacinostat",
    "Quisinostat",
    "CUDC-907",
    "Abexinostat",
    "Panobinostat",
    "Belinostat",
    "Givinostat",
    "AR-42",
]

val_epi_drugs = [
    "Trichostatin",
    "CUDC-101",
    "M344",
    "Resminostat",
    "Entinostat",  # no_ood
    "Tucidinostat",  # no_ood
    "Tacedinaline",  # no_ood
    "Mocetinostat",  # no_ood
    "Pracinostat",  # no_ood
]


# %%
if "split_ho_epigenetic" not in list(adata_sciplex.obs):
    print("Addig 'split_ho_epigenetic' to 'adata_sciplex.obs'.")
    obs_train, obs_val = train_test_split(adata_sciplex.obs.index, test_size=0.05)
    #     obs_val, obs_test = train_test_split(obs_tmp, test_size=0.5)

    adata_sciplex.obs["split_ho_epigenetic"] = "train"
    adata_sciplex.obs.loc[
        adata_sciplex.obs.index.isin(obs_val), "split_ho_epigenetic"
    ] = "test"

    # test
    validation_cond = (adata_sciplex.obs.condition.isin(val_epi_drugs)) & (
        adata_sciplex.obs.dose.isin([1e3, 1e4])
    )
    val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.4, copy=True).obs.index
    adata_sciplex.obs.loc[val_idx, "split_ho_epigenetic"] = "test"

    validation_cond = (adata_sciplex.obs.condition.isin(val_epi_drugs)) & (
        adata_sciplex.obs.dose.isin([1e1, 1e2])
    )
    val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.4, copy=True).obs.index
    adata_sciplex.obs.loc[val_idx, "split_ho_epigenetic"] = "test"

    validation_cond = adata_sciplex.obs.condition.isin(
        epigenetic_drugs_all[~epigenetic_drugs_all.isin(epigenetic_drugs)]
    )
    val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.4, copy=True).obs.index
    adata_sciplex.obs.loc[val_idx, "split_ho_epigenetic"] = "test"

    validation_cond = adata_sciplex.obs.control.isin([True])
    val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.4, copy=True).obs.index
    adata_sciplex.obs.loc[val_idx, "split_ho_epigenetic"] = "test"

    adata_sciplex.obs.loc[
        adata_sciplex.obs.condition.isin(ood_epi_drugs), "split_ho_epigenetic"
    ] = "ood"

# %%
if "split_ho_epigenetic_all" not in list(adata_sciplex.obs):
    print("Addig 'split_ho_epigenetic_all' to 'adata_sciplex.obs'.")
    obs_train, obs_val = train_test_split(adata_sciplex.obs.index, test_size=0.05)
    #     obs_val, obs_test = train_test_split(obs_tmp, test_size=0.5)

    adata_sciplex.obs["split_ho_epigenetic_all"] = "train"
    adata_sciplex.obs.loc[
        adata_sciplex.obs.index.isin(obs_val), "split_ho_epigenetic_all"
    ] = "test"

    validation_cond = adata_sciplex.obs.condition.isin(
        epigenetic_drugs_all[~epigenetic_drugs_all.isin(epigenetic_drugs)]
    )
    val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.5, copy=True).obs.index
    adata_sciplex.obs.loc[val_idx, "split_ho_epigenetic_all"] = "test"

    validation_cond = adata_sciplex.obs.control.isin([True])
    val_idx = sc.pp.subsample(adata_sciplex[validation_cond], 0.4, copy=True).obs.index
    adata_sciplex.obs.loc[val_idx, "split_ho_epigenetic_all"] = "test"

    adata_sciplex.obs.loc[
        adata_sciplex.obs.condition.isin(epigenetic_drugs), "split_ho_epigenetic_all"
    ] = "ood"


# %%
pd.crosstab(adata_sciplex.obs.split_ho_epigenetic, adata_sciplex.obs.control)

# %%
pd.crosstab(adata_sciplex.obs.split_ho_epigenetic_all, adata_sciplex.obs.control)

# %% [markdown]
# ## Add random split

# %%
if "split_random" not in list(adata_sciplex.obs):
    print("Addig 'split_random' to 'adata_sciplex.obs'.")
    obs_train, obs_tmp = train_test_split(adata_sciplex.obs.index, test_size=0.3)
    obs_val, obs_test = train_test_split(obs_tmp, test_size=0.5)

    adata_sciplex.obs["split_random"] = "train"
    adata_sciplex.obs.loc[
        adata_sciplex.obs.index.isin(obs_val), "split_random"
    ] = "test"
    adata_sciplex.obs.loc[
        adata_sciplex.obs.index.isin(obs_test), "split_random"
    ] = "ood"

# %%
pd.crosstab(adata_sciplex.obs.split_random, adata_sciplex.obs.control)

# %% [markdown]
# ## Save `adata_sciplex` and `adata_sciplex_lincs_genes`

# %%
assert (adata_sciplex.obs.index == adata_sciplex_lincs_genes.obs.index).all()

adata_sciplex_lincs_genes.obs["split_ood_finetuning"] = adata_sciplex.obs[
    "split_ood_finetuning"
]
adata_sciplex_lincs_genes.obs["split_ho_epigenetic"] = adata_sciplex.obs[
    "split_ho_epigenetic"
]
adata_sciplex_lincs_genes.obs["split_ho_epigenetic_all"] = adata_sciplex.obs[
    "split_ho_epigenetic_all"
]
adata_sciplex_lincs_genes.obs["split_random"] = adata_sciplex.obs["split_random"]

# %%
sc.write(PROJECT_DIR / "datasets" / "sciplex_complete.h5ad", adata_sciplex)

# %%
sc.write(
    PROJECT_DIR / "datasets" / "sciplex_complete_lincs_genes.h5ad",
    adata_sciplex_lincs_genes,
)

# %% [markdown]
# ______

# %%
adata_sciplex = sc.read(PROJECT_DIR / "datasets" / "sciplex_complete.h5ad")
adata_sciplex

# %%
adata_sciplex_lincs_genes = sc.read(
    PROJECT_DIR / "datasets" / "sciplex_complete_lincs_genes.h5ad"
)
adata_sciplex_lincs_genes

# %% [markdown]
# _________

# %% [markdown]
# ### Check splits

# %%
pd.crosstab(adata_sciplex.obs.split_ood_finetuning, adata_sciplex.obs.control)

# %%
pd.crosstab(adata_sciplex.obs.split_ho_pathway, adata_sciplex.obs.control)

# %% [markdown]
# ____

# %% [markdown]
# ## Create small sciplex dataset

# %%
adatas = []

for perturbation in np.unique(adata.obs.condition):
    tmp = adata[adata.obs.condition == perturbation].copy()
    tmp = sc.pp.subsample(tmp, n_obs=40, copy=True)
    adatas.append(tmp)

adata_subset = adatas[0].concatenate(adatas[1:])
adata_subset.uns = adata.uns.copy()

adata_subset

# %%
if "split" not in list(adata_subset.obs):
    print("Addig 'split' to 'adata_subset.obs'.")
    obs_train, obs_tmp = train_test_split(adata_subset.obs.index, test_size=0.3)
    obs_val, obs_test = train_test_split(obs_tmp, test_size=0.5)

    adata_subset.obs["split"] = "train"
    adata_subset.obs.loc[adata_subset.obs.index.isin(obs_val), "split"] = "test"
    adata_subset.obs.loc[adata_subset.obs.index.isin(obs_test), "split"] = "ood"

# %%
adata_subset.obs.split.value_counts()

# %%
sc.write(
    PROJECT_DIR / "datasets" / "sciplex_complete_subset_lincs_genes.h5ad", adata_subset
)

# %% [markdown]
# ____

# %% [markdown]
# ## Create middle sized sci-Plex subset

# %%
delete_idx = []
for drug, df in adata_sciplex.obs.groupby("condition"):
    # Low dose
    cond = df.dose.isin([10])
    idx = df[cond].sample(frac=0.6).index
    delete_idx.extend(idx)
    # Small dose
    cond = df.dose.isin([100])
    idx = df[cond].sample(frac=0.5).index
    delete_idx.extend(idx)
    # Middle dose
    cond = df.dose.isin([1000])
    idx = df[cond].sample(frac=0.3).index
    delete_idx.extend(idx)
    # High dose
    cond = df.dose.isin([10000])
    idx = df[cond].sample(frac=0.15).index
    delete_idx.extend(idx)

# %%
cond = ~pd.Series(adata_sciplex.obs.index).isin(delete_idx)

# %%
cond.sum()

# %%
cond = cond.to_list()

# %%
assert (adata_sciplex.obs.index == adata_sciplex_lincs_genes.obs.index).all()

# %%
adata = adata_sciplex[cond].copy()
adata_lincs_genes = adata_sciplex_lincs_genes[cond].copy()

# %%
pd.crosstab(adata.obs.split_ood_finetuning, adata.obs.control)

# %%
pd.crosstab(adata.obs.split_ho_pathway, adata.obs.control)

# %%
sc.write(PROJECT_DIR / "datasets" / "sciplex_complete_middle_subset.h5ad", adata)

# %%
sc.write(
    PROJECT_DIR / "datasets" / "sciplex_complete_middle_subset_lincs_genes.h5ad",
    adata_lincs_genes,
)

# %%
