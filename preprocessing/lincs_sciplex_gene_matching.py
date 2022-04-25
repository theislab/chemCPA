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
# * `'lincs_full_smiles.h5ad'`
# * `'sciplex_raw_chunk_{i}.h5ad'` with $i \in \{0,1,2,3,4\}$
#
# **Output**
# * `'sciplex3_matched_genes_lincs.h5ad'`
# * Only with genes that are shared with `lincs`: `'sciplex3_lincs_genes.h5ad'`
# * Only with genes that are shared with `sciplex`: `'lincs_full_smiles_sciplex_genes.h5ad'`
#
# ## Imports

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sfaira

sc.set_figure_params(dpi=80, frameon=False)
sc.logging.print_header()
os.getcwd()

from chemCPA.paths import DATA_DIR, PROJECT_DIR

pd.set_option("display.max_columns", 100)

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown] tags=[]
# ## Load data

# %% [markdown]
# Load lincs

# %% tags=[]
adata_lincs = sc.read(PROJECT_DIR / "datasets" / "lincs_full_smiles.h5ad")

# %% [markdown]
# Load trapnell

# %%
adatas = []
for i in range(5):
    adatas.append(sc.read(PROJECT_DIR / "datasets" / f"sciplex_raw_chunk_{i}.h5ad"))
adata = adatas[0].concatenate(adatas[1:])

# %% [markdown]
# Add gene_id to trapnell

# %%
adata.var["gene_id"] = adata.var.id.str.split(".").str[0]

# %% [markdown] tags=[]
# ### Get gene ids from symbols via sfaira

# %% [markdown]
# Load genome container with sfaira

# %%
genome_container = sfaira.versions.genomes.GenomeContainer(
    organism="homo_sapiens", release="82"
)

# %% [markdown]
# Extend symbols dict with unknown symbol

# %%
symbols_dict = genome_container.symbol_to_id_dict
symbols_dict.update({"PLSCR3": "ENSG00000187838"})

# %% [markdown]
# Identify genes that are shared between lincs and trapnell

# %% tags=[]
# For lincs
adata_lincs.var["gene_id"] = adata_lincs.var_names.map(symbols_dict)
adata_lincs.var["in_sciplex"] = adata_lincs.var.gene_id.isin(adata.var.gene_id)

# %%
# For trapnell
adata.var["in_lincs"] = adata.var.gene_id.isin(adata_lincs.var.gene_id)

# %% [markdown] tags=[]
# ## Preprocess sciplex dataset

# %% [markdown]
# See `sciplex3.ipynb`

# %% [markdown]
# The original CPA implementation required to subset the data due to scaling limitations.
# In this version we expect to be able to handle the full sciplex dataset.

# %%
SUBSET = False

if SUBSET:
    sc.pp.subsample(adata, fraction=0.5)

# %%
sc.pp.normalize_per_cell(adata)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata, n_top_genes=1032, subset=False)

# %% [markdown] tags=[]
# ### Combine HVG with lincs genes
#
# Union of genes that are considered highly variable and those that are shared with lincs

# %%
((adata.var.in_lincs) | (adata.var.highly_variable)).sum()

# %% [markdown]
# Subset to that union of genes

# %%
adata = adata[:, (adata.var.in_lincs) | (adata.var.highly_variable)].copy()

# %% [markdown] tags=[]
# ### Create additional meta data

# %% [markdown]
# Normalise dose values

# %%
adata.obs["dose_val"] = adata.obs.dose.astype(float) / np.max(
    adata.obs.dose.astype(float)
)
adata.obs.loc[adata.obs["product_name"].str.contains("Vehicle"), "dose_val"] = 1.0

# %%
adata.obs["dose_val"].value_counts()

# %% [markdown]
# Change `product_name`

# %%
adata.obs["product_name"] = [x.split(" ")[0] for x in adata.obs["product_name"]]
adata.obs.loc[
    adata.obs["product_name"].str.contains("Vehicle"), "product_name"
] = "control"

# %% [markdown]
# Create copy of `product_name` with column name `control`

# %%
adata.obs["condition"] = adata.obs.product_name.copy()

# %% [markdown]
# Add combinations of drug (`condition`), dose (`dose_val`), and cell_type (`cell_type`)

# %%
adata.obs["drug_dose_name"] = (
    adata.obs.condition.astype(str) + "_" + adata.obs.dose_val.astype(str)
)
adata.obs["cov_drug_dose_name"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.drug_dose_name.astype(str)
)
adata.obs["cov_drug"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)

# %% [markdown]
# Add `control` columns with vale `1` where only the vehicle was used

# %%
adata.obs["control"] = [
    1 if x == "control_1.0" else 0 for x in adata.obs.drug_dose_name.values
]

# %% [markdown] tags=[]
# ## Compute DE genes

# %%
from chemCPA.helper import rank_genes_groups_by_cov

rank_genes_groups_by_cov(
    adata,
    groupby="cov_drug",
    covariate="cell_type",
    control_group="control",
    key_added="all_DEGs",
)

# %%
adata_subset = adata[:, adata.var.in_lincs].copy()
rank_genes_groups_by_cov(
    adata_subset,
    groupby="cov_drug",
    covariate="cell_type",
    control_group="control",
    key_added="lincs_DEGs",
)
adata.uns["lincs_DEGs"] = adata_subset.uns["lincs_DEGs"]

# %% [markdown]
# ### Map all unique `cov_drug_dose_name` to the computed DEGs, independent of the dose value
#
# Create mapping between names with dose and without dose

# %%
cov_drug_dose_unique = adata.obs.cov_drug_dose_name.unique()

# %%
remove_dose = lambda s: "_".join(s.split("_")[:-1])
cov_drug = pd.Series(cov_drug_dose_unique).apply(remove_dose)
dose_no_dose_dict = dict(zip(cov_drug_dose_unique, cov_drug))

# %% [markdown]
# ### Compute new dicts for DEGs

# %%
uns_keys = ["all_DEGs", "lincs_DEGs"]

# %%
for uns_key in uns_keys:
    new_DEGs_dict = {}

    df_DEGs = pd.Series(adata.uns[uns_key])

    for key, value in dose_no_dose_dict.items():
        if "control" in key:
            continue
        new_DEGs_dict[key] = df_DEGs.loc[value]
    adata.uns[uns_key] = new_DEGs_dict

# %%
adata

# %% [markdown] tags=[]
# ## Create sciplex splits
#
# This is not the right configuration fot the experiments we want but for the moment this is okay

# %% [markdown] tags=[]
# ### OOD in Pathways

# %% tags=[]
adata.obs["split_ho_pathway"] = "train"  # reset

ho_drugs = [
    # selection of drugs from various pathways
    "Azacitidine",
    "Carmofur",
    "Pracinostat",
    "Cediranib",
    "Luminespib",
    "Crizotinib",
    "SNS-314",
    "Obatoclax",
    "Momelotinib",
    "AG-14361",
    "Entacapone",
    "Fulvestrant",
    "Mesna",
    "Zileuton",
    "Enzastaurin",
    "IOX2",
    "Alvespimycin",
    "XAV-939",
    "Fasudil",
]

ho_drug_pathway = adata.obs["condition"].isin(ho_drugs)
adata.obs.loc[ho_drug_pathway, "pathway_level_1"].value_counts()

# %% tags=[]
ho_drug_pathway.sum()

# %%
adata.obs.loc[
    ho_drug_pathway & (adata.obs["dose_val"] == 1.0), "split_ho_pathway"
] = "ood"

test_idx = sc.pp.subsample(
    adata[adata.obs["split_ho_pathway"] != "ood"], 0.15, copy=True
).obs.index
adata.obs.loc[test_idx, "split_ho_pathway"] = "test"

# %%
pd.crosstab(
    adata.obs.pathway_level_1,
    adata.obs["condition"][adata.obs.condition.isin(ho_drugs)],
)

# %%
adata.obs["split_ho_pathway"].value_counts()

# %%
adata[adata.obs.split_ho_pathway == "ood"].obs.condition.value_counts()

# %%
adata[adata.obs.split_ho_pathway == "test"].obs.condition.value_counts()

# %% [markdown] tags=[]
# ### OOD drugs in epigenetic regulation, Tyrosine kinase signaling, cell cycle regulation

# %%
adata.obs["pathway_level_1"].value_counts()

# %% [markdown] tags=[]
# ___
#
# #### Tyrosine signaling

# %%
adata.obs.loc[
    adata.obs.pathway_level_1.isin(["Tyrosine kinase signaling"]), "condition"
].value_counts()

# %%
tyrosine_drugs = adata.obs.loc[
    adata.obs.pathway_level_1.isin(["Tyrosine kinase signaling"]), "condition"
].unique()

# %%
adata.obs["split_tyrosine_ood"] = "train"

test_idx = sc.pp.subsample(
    adata[adata.obs.pathway_level_1.isin(["Tyrosine kinase signaling"])],
    0.20,
    copy=True,
).obs.index
adata.obs.loc[test_idx, "split_tyrosine_ood"] = "test"

adata.obs.loc[
    adata.obs.condition.isin(
        ["Cediranib", "Crizotinib", "Motesanib", "BMS-754807", "Nintedanib"]
    ),
    "split_tyrosine_ood",
] = "ood"

# %%
adata.obs.split_tyrosine_ood.value_counts()

# %% tags=[]
pd.crosstab(
    adata.obs.split_tyrosine_ood,
    adata.obs["condition"][adata.obs.condition.isin(tyrosine_drugs)],
)

# %%
pd.crosstab(adata.obs.split_tyrosine_ood, adata.obs.dose_val)

# %% [markdown] tags=[]
# ____
#
# #### Epigenetic regulation

# %%
adata.obs.loc[
    adata.obs.pathway_level_1.isin(["Epigenetic regulation"]), "condition"
].value_counts()

# %%
epigenetic_drugs = adata.obs.loc[
    adata.obs.pathway_level_1.isin(["Epigenetic regulation"]), "condition"
].unique()

# %%
adata.obs["split_epigenetic_ood"] = "train"

test_idx = sc.pp.subsample(
    adata[adata.obs.pathway_level_1.isin(["Epigenetic regulation"])], 0.20, copy=True
).obs.index
adata.obs.loc[test_idx, "split_epigenetic_ood"] = "test"

adata.obs.loc[
    adata.obs.condition.isin(
        ["Azacitidine", "Pracinostat", "Trichostatin", "Quisinostat", "Tazemetostat"]
    ),
    "split_epigenetic_ood",
] = "ood"

# %%
adata.obs.split_epigenetic_ood.value_counts()

# %% tags=[]
pd.crosstab(
    adata.obs.split_epigenetic_ood,
    adata.obs["condition"][adata.obs.condition.isin(epigenetic_drugs)],
)

# %%
pd.crosstab(adata.obs.split_tyrosine_ood, adata.obs.dose_val)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# __________
#
# #### Cell cycle regulation

# %%
adata.obs.loc[
    adata.obs.pathway_level_1.isin(["Cell cycle regulation"]), "condition"
].value_counts()

# %%
cell_cycle_drugs = adata.obs.loc[
    adata.obs.pathway_level_1.isin(["Cell cycle regulation"]), "condition"
].unique()

# %%
adata.obs["split_cellcycle_ood"] = "train"

test_idx = sc.pp.subsample(
    adata[adata.obs.pathway_level_1.isin(["Cell cycle regulation"])], 0.20, copy=True
).obs.index
adata.obs.loc[test_idx, "split_cellcycle_ood"] = "test"

adata.obs.loc[
    adata.obs.condition.isin(["SNS-314", "Flavopiridol", "Roscovitine"]),
    "split_cellcycle_ood",
] = "ood"

# %%
adata.obs.split_cellcycle_ood.value_counts()

# %% tags=[]
pd.crosstab(
    adata.obs.split_cellcycle_ood,
    adata.obs["condition"][adata.obs.condition.isin(cell_cycle_drugs)],
)

# %%
pd.crosstab(adata.obs.split_cellcycle_ood, adata.obs.dose_val)

# %%
[c for c in adata.obs.columns if "split" in c]

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Further splits
#
# **We omit these split as we design our own splits - for referece this is commented out for the moment**
#
# Also a split which sees all data:

# %%
# adata.obs['split_all'] = 'train'
# test_idx = sc.pp.subsample(adata, .10, copy=True).obs.index
# adata.obs.loc[test_idx, 'split_all'] = 'test'

# %%
# adata.obs['ct_dose'] = adata.obs.cell_type.astype('str') + '_' + adata.obs.dose_val.astype('str')

# %% [markdown]
# Round robin splits: dose and cell line combinations will be held out in turn.

# %%
# i = 0
# split_dict = {}

# %%
# # single ct holdout
# for ct in adata.obs.cell_type.unique():
#     for dose in adata.obs.dose_val.unique():
#         i += 1
#         split_name = f'split{i}'
#         split_dict[split_name] = f'{ct}_{dose}'

#         adata.obs[split_name] = 'train'
#         adata.obs.loc[adata.obs.ct_dose == f'{ct}_{dose}', split_name] = 'ood'

#         test_idx = sc.pp.subsample(adata[adata.obs[split_name] != 'ood'], .16, copy=True).obs.index
#         adata.obs.loc[test_idx, split_name] = 'test'

#         display(adata.obs[split_name].value_counts())

# %%
# # double ct holdout
# for cts in [('A549', 'MCF7'), ('A549', 'K562'), ('MCF7', 'K562')]:
#     for dose in adata.obs.dose_val.unique():
#         i += 1
#         split_name = f'split{i}'
#         split_dict[split_name] = f'{cts[0]}+{cts[1]}_{dose}'

#         adata.obs[split_name] = 'train'
#         adata.obs.loc[adata.obs.ct_dose == f'{cts[0]}_{dose}', split_name] = 'ood'
#         adata.obs.loc[adata.obs.ct_dose == f'{cts[1]}_{dose}', split_name] = 'ood'

#         test_idx = sc.pp.subsample(adata[adata.obs[split_name] != 'ood'], .16, copy=True).obs.index
#         adata.obs.loc[test_idx, split_name] = 'test'

#         display(adata.obs[split_name].value_counts())

# %%
# # triple ct holdout
# for dose in adata.obs.dose_val.unique():
#     i += 1
#     split_name = f'split{i}'

#     split_dict[split_name] = f'all_{dose}'
#     adata.obs[split_name] = 'train'
#     adata.obs.loc[adata.obs.dose_val == dose, split_name] = 'ood'

#     test_idx = sc.pp.subsample(adata[adata.obs[split_name] != 'ood'], .16, copy=True).obs.index
#     adata.obs.loc[test_idx, split_name] = 'test'

#     display(adata.obs[split_name].value_counts())

# %%
# adata.uns['all_DEGs']

# %% [markdown] tags=[]
# ## Save adata

# %% [markdown]
# Reindex the lincs dataset

# %%
sciplex_ids = pd.Index(adata.var.gene_id)

lincs_idx = [
    sciplex_ids.get_loc(_id)
    for _id in adata_lincs.var.gene_id[adata_lincs.var.in_sciplex]
]

# %%
non_lincs_idx = [
    sciplex_ids.get_loc(_id)
    for _id in adata.var.gene_id
    if not adata_lincs.var.gene_id.isin([_id]).any()
]

lincs_idx.extend(non_lincs_idx)

# %%
adata = adata[:, lincs_idx].copy()

# %% tags=[]
fname = PROJECT_DIR / "datasets" / "sciplex3_matched_genes_lincs.h5ad"

sc.write(fname, adata)

# %% [markdown]
# Check that it worked

# %%
sc.read(fname)

# %% [markdown]
# ## Subselect to shared only shared genes

# %% [markdown]
# Subset to shared genes

# %% tags=[]
adata_lincs = adata_lincs[:, adata_lincs.var.in_sciplex].copy()

# %%
adata = adata[:, adata.var.in_lincs].copy()

# %%
adata_lincs.var_names

# %%
adata.var_names

# %% [markdown]
# ## Save adata objects with shared genes only
# Index of lincs has also been reordered accordingly

# %%
fname = PROJECT_DIR / "datasets" / "sciplex3_lincs_genes.h5ad"

sc.write(fname, adata)

# %% [markdown]
# ____

# %%
# fname_lincs = PROJECT_DIR/'datasets'/'lincs_full_smiles_sciplex_genes.h5ad'

# sc.write(fname_lincs, adata_lincs)

# %%

# %%
