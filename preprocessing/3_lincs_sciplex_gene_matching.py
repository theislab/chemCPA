# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 3 LINCS SCIPLEX GENE MATCHING
#
# **Requires**
# * `'lincs_full_smiles.h5ad'`
# * `'sciplex_raw_chunk_{i}.h5ad'` with $i \in \{0,1,2,3,4\}$
#
# **Output**
# * `'sciplex3_matched_genes_lincs.h5ad'`
# * `lincs`: `'sciplex3_lincs_genes.h5ad'`
# * `sciplex`: `'lincs_full_smiles_sciplex_genes.h5ad'`
#
# ## Description 
#
# The goal of this notebook is to match and merge genes between the LINCS and SciPlex datasets, resulting in the creation of three new datasets:
#
# ### Created datasets
#
# - **`sciplex3_matched_genes_lincs.h5ad`**: Contains **SciPlex observations**. **Genes are limited to the intersection** of the genes found in both LINCS and SciPlex datasets.
#
# - **`sciplex3_lincs_genes.h5ad`**: Contains **SciPlex data**, but filtered to include **only the genes that are shared with the LINCS dataset**.
#
# - **`lincs_full_smiles_sciplex_genes.h5ad`**: Contains **LINCS data**, but filtered to include **only the genes that are shared with the SciPlex dataset**.
#
# To create these datasets, we need to match the genes between the two datasets, which is done as follows:
#
# ### Gene Matching
#
# 1. **Gene ID Assignment**: SciPlex gene names are standardized to Ensembl gene IDs by extracting the primary identifier and using either **sfaira** or a predefined mapping (`symbols_dict.json`). The LINCS dataset is already standardized.
#
# 2. **Identifying Shared Genes**: We then compute the intersection of the gene IDs (`gene_id`) inside LINCS and SciPlex. Both datasets are then filtered to retain only these shared genes.
#
# 3. **Reindexing**: The LINCS dataset is reindexed to match the order of genes in the SciPlex dataset.
#
#

# +
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sfaira
import warnings
os.getcwd()

from chemCPA.paths import DATA_DIR, PROJECT_DIR

pd.set_option('display.max_columns', 100)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath('')))
sys.path.append(PROJECT_DIR)
import raw_data.datasets as datasets
import logging

logging.basicConfig(level=logging.INFO)
from notebook_utils import suppress_output

import scanpy as sc
with suppress_output():
    sc.set_figure_params(dpi=80, frameon=False)
    sc.logging.print_header()
    warnings.filterwarnings('ignore')
# -

# %load_ext autoreload
# %autoreload 2

# ## Load data

# Load lincs

adata_lincs = sc.read(DATA_DIR/'lincs_full_smiles.h5ad' )

# Load sciplex 

# +
from tqdm import tqdm
from chemCPA.paths import DATA_DIR, PROJECT_DIR
from raw_data.datasets import sciplex

# Load and concatenate chunks
adatas_sciplex = []
logging.info("Starting to load in sciplex data")

# Get paths to all sciplex chunks
chunk_paths = sciplex()

# Load chunks with progress bar
for chunk_path in tqdm(chunk_paths, desc="Loading sciplex chunks"):
    tqdm.write(f"Loading {os.path.basename(chunk_path)}")
    adatas_sciplex.append(sc.read(chunk_path))
    
adata_sciplex = adatas_sciplex[0].concatenate(adatas_sciplex[1:])
logging.info("Sciplex data loaded")
# -

# Add gene_id to sciplex

adata_sciplex.var['gene_id'] = adata_sciplex.var.id.str.split('.').str[0]
adata_sciplex.var['gene_id'].head()

# ### Get gene ids from symbols via sfaira

# Load genome container with sfaira

try: 
    # load json file with symbol to id mapping
    import json
    with open(DATA_DIR/ 'symbols_dict.json') as json_file:
        symbols_dict = json.load(json_file)
except: 
    logging.info("No symbols_dict.json found, falling back to sfaira")
    genome_container = sfaira.versions.genomes.GenomeContainer(organism="homo_sapiens", release="82")
    symbols_dict = genome_container.symbol_to_id_dict
    # Extend symbols dict with unknown symbol
    symbols_dict.update({'PLSCR3':'ENSG00000187838'})

# Identify genes that are shared between lincs and trapnell

# For lincs
adata_lincs.var['gene_id'] = adata_lincs.var_names.map(symbols_dict)
adata_lincs.var['in_sciplex'] = adata_lincs.var.gene_id.isin(adata_sciplex.var.gene_id)

# For trapnell
adata_sciplex.var['in_lincs'] = adata_sciplex.var.gene_id.isin(adata_lincs.var.gene_id)

# ## Preprocess sciplex dataset

# See `sciplex3.ipynb`

# The original CPA implementation required to subset the data due to scaling limitations.   
# In this version we expect to be able to handle the full sciplex dataset.

# +
SUBSET = False

if SUBSET: 
    sc.pp.subsample(adata_sciplex, fraction=0.5, random_state=42)
# -

sc.pp.normalize_per_cell(adata_sciplex)

sc.pp.log1p(adata_sciplex)

sc.pp.highly_variable_genes(adata_sciplex, n_top_genes=1032, subset=False)

# ### Combine HVG with lincs genes
#
# Union of genes that are considered highly variable and those that are shared with lincs

((adata_sciplex.var.in_lincs) | (adata_sciplex.var.highly_variable)).sum()

# Subset to that union of genes

adata_sciplex = adata_sciplex[:, (adata_sciplex.var.in_lincs) | (adata_sciplex.var.highly_variable)].copy()

# ### Create additional meta data 

# Normalise dose values

adata_sciplex.obs['dose_val'] = adata_sciplex.obs.dose.astype(float) / np.max(adata_sciplex.obs.dose.astype(float))
adata_sciplex.obs.loc[adata_sciplex.obs['product_name'].str.contains('Vehicle'), 'dose_val'] = 1.0

adata_sciplex.obs['dose_val'].value_counts()

# Change `product_name`

adata_sciplex.obs['product_name'] = [x.split(' ')[0] for x in adata_sciplex.obs['product_name']]
adata_sciplex.obs.loc[adata_sciplex.obs['product_name'].str.contains('Vehicle'), 'product_name'] = 'control'

# Create copy of `product_name` with column name `control`

adata_sciplex.obs['condition'] = adata_sciplex.obs.product_name.copy()

# Add combinations of drug (`condition`), dose (`dose_val`), and cell_type (`cell_type`)

# make column of dataframe to categorical 
adata_sciplex.obs["condition"] = adata_sciplex.obs["condition"].astype('category').cat.rename_categories({"(+)-JQ1": "JQ1"})
adata_sciplex.obs['drug_dose_name'] = adata_sciplex.obs.condition.astype(str) + '_' + adata_sciplex.obs.dose_val.astype(str)
adata_sciplex.obs['cov_drug_dose_name'] = adata_sciplex.obs.cell_type.astype(str) + '_' + adata_sciplex.obs.drug_dose_name.astype(str)
adata_sciplex.obs['cov_drug'] = adata_sciplex.obs.cell_type.astype(str) + '_' + adata_sciplex.obs.condition.astype(str)

# Add `control` columns with vale `1` where only the vehicle was used

adata_sciplex.obs['control'] = [1 if x == 'control_1.0' else 0 for x in adata_sciplex.obs.drug_dose_name.values]

# ## Compute DE genes

# +
from chemCPA.helper import rank_genes_groups_by_cov

rank_genes_groups_by_cov(adata_sciplex, groupby='cov_drug', covariate='cell_type', control_group='control', key_added='all_DEGs')
# -

adata_subset = adata_sciplex[:, adata_sciplex.var.in_lincs].copy()
rank_genes_groups_by_cov(adata_subset, groupby='cov_drug', covariate='cell_type', control_group='control', key_added='lincs_DEGs')
adata_sciplex.uns['lincs_DEGs'] = adata_subset.uns['lincs_DEGs']

# ### Map all unique `cov_drug_dose_name` to the computed DEGs, independent of the dose value
#
# Create mapping between names with dose and without dose

cov_drug_dose_unique = adata_sciplex.obs.cov_drug_dose_name.unique()

remove_dose = lambda s: '_'.join(s.split('_')[:-1])
cov_drug = pd.Series(cov_drug_dose_unique).apply(remove_dose)
dose_no_dose_dict = dict(zip(cov_drug_dose_unique, cov_drug))

# ### Compute new dicts for DEGs

uns_keys = ['all_DEGs', 'lincs_DEGs']

for uns_key in uns_keys:
    new_DEGs_dict = {}

    df_DEGs = pd.Series(adata_sciplex.uns[uns_key])

    for key, value in dose_no_dose_dict.items():
        if 'control' in key:
            continue
        new_DEGs_dict[key] = df_DEGs.loc[value]
    adata_sciplex.uns[uns_key] = new_DEGs_dict

adata_sciplex

# ## Create sciplex splits
#
# This is not the right configuration fot the experiments we want but for the moment this is okay

# ### OOD in Pathways

# +
adata_sciplex.obs['split_ho_pathway'] = 'train'  # reset

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

ho_drug_pathway = adata_sciplex.obs['condition'].isin(ho_drugs)
adata_sciplex.obs.loc[ho_drug_pathway, 'pathway_level_1'].value_counts()
# -

ho_drug_pathway.sum()

# +
adata_sciplex.obs.loc[ho_drug_pathway & (adata_sciplex.obs['dose_val'] == 1.0), 'split_ho_pathway'] = 'ood'

test_idx = sc.pp.subsample(adata_sciplex[adata_sciplex.obs['split_ho_pathway'] != 'ood'], .15, copy=True).obs.index
adata_sciplex.obs.loc[test_idx, 'split_ho_pathway'] = 'test'
# -

pd.crosstab(adata_sciplex.obs.pathway_level_1, adata_sciplex.obs['condition'][adata_sciplex.obs.condition.isin(ho_drugs)])

adata_sciplex.obs['split_ho_pathway'].value_counts()

adata_sciplex[adata_sciplex.obs.split_ho_pathway == 'ood'].obs.condition.value_counts()

adata_sciplex[adata_sciplex.obs.split_ho_pathway == 'test'].obs.condition.value_counts()

# ### OOD drugs in epigenetic regulation, Tyrosine kinase signaling, cell cycle regulation

adata_sciplex.obs['pathway_level_1'].value_counts()

# ___
#
# #### Tyrosine signaling

adata_sciplex.obs.loc[adata_sciplex.obs.pathway_level_1.isin(["Tyrosine kinase signaling"]),'condition'].value_counts()

tyrosine_drugs = adata_sciplex.obs.loc[adata_sciplex.obs.pathway_level_1.isin(["Tyrosine kinase signaling"]),'condition'].unique()

# +
adata_sciplex.obs['split_tyrosine_ood'] = 'train'  

test_idx = sc.pp.subsample(adata_sciplex[adata_sciplex.obs.pathway_level_1.isin(["Tyrosine kinase signaling"])], .20, copy=True).obs.index
adata_sciplex.obs.loc[test_idx, 'split_tyrosine_ood'] = 'test'

adata_sciplex.obs.loc[adata_sciplex.obs.condition.isin(["Cediranib", "Crizotinib", "Motesanib", "BMS-754807", "Nintedanib"]), 'split_tyrosine_ood'] = 'ood'  
# -

adata_sciplex.obs.split_tyrosine_ood.value_counts()

pd.crosstab(adata_sciplex.obs.split_tyrosine_ood, adata_sciplex.obs['condition'][adata_sciplex.obs.condition.isin(tyrosine_drugs)])

pd.crosstab(adata_sciplex.obs.split_tyrosine_ood, adata_sciplex.obs.dose_val)

# ____
#
# #### Epigenetic regulation

adata_sciplex.obs.loc[adata_sciplex.obs.pathway_level_1.isin(["Epigenetic regulation"]),'condition'].value_counts()

epigenetic_drugs = adata_sciplex.obs.loc[adata_sciplex.obs.pathway_level_1.isin(["Epigenetic regulation"]),'condition'].unique()

# +
adata_sciplex.obs['split_epigenetic_ood'] = 'train'  

test_idx = sc.pp.subsample(adata_sciplex[adata_sciplex.obs.pathway_level_1.isin(["Epigenetic regulation"])], .20, copy=True).obs.index
adata_sciplex.obs.loc[test_idx, 'split_epigenetic_ood'] = 'test'

adata_sciplex.obs.loc[adata_sciplex.obs.condition.isin(["Azacitidine", "Pracinostat", "Trichostatin", "Quisinostat", "Tazemetostat"]), 'split_epigenetic_ood'] = 'ood'  
# -

adata_sciplex.obs.split_epigenetic_ood.value_counts()

pd.crosstab(adata_sciplex.obs.split_epigenetic_ood, adata_sciplex.obs['condition'][adata_sciplex.obs.condition.isin(epigenetic_drugs)])

pd.crosstab(adata_sciplex.obs.split_tyrosine_ood, adata_sciplex.obs.dose_val)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# __________
#
# #### Cell cycle regulation
# -

adata_sciplex.obs.loc[adata_sciplex.obs.pathway_level_1.isin(["Cell cycle regulation"]),'condition'].value_counts()

cell_cycle_drugs = adata_sciplex.obs.loc[adata_sciplex.obs.pathway_level_1.isin(["Cell cycle regulation"]),'condition'].unique()

# +
adata_sciplex.obs['split_cellcycle_ood'] = 'train'  

test_idx = sc.pp.subsample(adata_sciplex[adata_sciplex.obs.pathway_level_1.isin(["Cell cycle regulation"])], .20, copy=True).obs.index
adata_sciplex.obs.loc[test_idx, 'split_cellcycle_ood'] = 'test'

adata_sciplex.obs.loc[adata_sciplex.obs.condition.isin(["SNS-314", "Flavopiridol", "Roscovitine"]), 'split_cellcycle_ood'] = 'ood'  
# -

adata_sciplex.obs.split_cellcycle_ood.value_counts()

pd.crosstab(adata_sciplex.obs.split_cellcycle_ood, adata_sciplex.obs['condition'][adata_sciplex.obs.condition.isin(cell_cycle_drugs)])

pd.crosstab(adata_sciplex.obs.split_cellcycle_ood, adata_sciplex.obs.dose_val)

[c for c in adata_sciplex.obs.columns if 'split' in c]

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Further splits
#
# **We omit these split as we design our own splits - for referece this is commented out for the moment**
#
# Also a split which sees all data:

# +
# adata.obs['split_all'] = 'train'
# test_idx = sc.pp.subsample(adata, .10, copy=True).obs.index
# adata.obs.loc[test_idx, 'split_all'] = 'test'

# +
# adata.obs['ct_dose'] = adata.obs.cell_type.astype('str') + '_' + adata.obs.dose_val.astype('str')
# -

# Round robin splits: dose and cell line combinations will be held out in turn.

# +
# i = 0
# split_dict = {}

# +
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

# +
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

# +
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

# +
# adata.uns['all_DEGs']
# -

# ## Save adata

# Reindex the lincs dataset

# +
sciplex_ids = pd.Index(adata_sciplex.var.gene_id)

lincs_idx = [sciplex_ids.get_loc(_id) for _id in adata_lincs.var.gene_id[adata_lincs.var.in_sciplex]]

# +
non_lincs_idx = [sciplex_ids.get_loc(_id) for _id in adata_sciplex.var.gene_id if not adata_lincs.var.gene_id.isin([_id]).any()]

lincs_idx.extend(non_lincs_idx)
# -

adata_sciplex = adata_sciplex[:, lincs_idx].copy()

# +
fname = PROJECT_DIR/'datasets'/'sciplex3_matched_genes_lincs.h5ad'

sc.write(fname, adata_sciplex)
# -

# Check that it worked

sc.read(fname)

# ## Subselect to shared only shared genes

# Subset to shared genes

adata_lincs = adata_lincs[:, adata_lincs.var.in_sciplex].copy() 

adata_sciplex = adata_sciplex[:, adata_sciplex.var.in_lincs].copy()

adata_lincs.var_names

adata_sciplex.var_names

# ## Save adata objects with shared genes only
# Index of lincs has also been reordered accordingly

# +
fname = PROJECT_DIR/'datasets'/'sciplex3_lincs_genes.h5ad'

sc.write(fname, adata_sciplex)
# -

# ____

# +
fname_lincs = PROJECT_DIR/'datasets'/'lincs_full_smiles_sciplex_genes.h5ad'

sc.write(fname_lincs, adata_lincs)
# -


