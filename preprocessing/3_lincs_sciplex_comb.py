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

# # 3 LINCS SCIPLEX COMBINATORIAL GENE MATCHING
#
# **Requires**
# * `'lincs_full_smiles.h5ad'`
# * `'combo_sciplex_prep_hvg_filtered.h5ad'`
#
# **Output**
# * `'combo_sciplex_matched_genes_lincs.h5ad'`
# * `lincs`: `'combo_sciplex_lincs_genes.h5ad'`
# * `sciplex`: `'lincs_full_smiles_combo_sciplex_genes.h5ad'`
#
# ## Description 
#
# The goal of this notebook is to match and merge genes between the LINCS and combinatorial SciPlex datasets, resulting in the creation of three new datasets:
#
# ### Created datasets
#
# - **`combo_sciplex_matched_genes_lincs.h5ad`**: Contains **combinatorial SciPlex observations**. **Genes are limited to the intersection** of the genes found in both LINCS and SciPlex datasets.
#
# - **`combo_sciplex_lincs_genes.h5ad`**: Contains **combinatorial SciPlex data**, but filtered to include **only the genes that are shared with the LINCS dataset**.
#
# - **`lincs_full_smiles_combo_sciplex_genes.h5ad`**: Contains **LINCS data**, but filtered to include **only the genes that are shared with the combinatorial SciPlex dataset**.
#
# To create these datasets, we need to match the genes between the two datasets, which is done as follows:
#
# ### Gene Matching
#
# 1. **Gene ID Assignment**: SciPlex gene names are standardized to Ensembl gene IDs by extracting the primary identifier and using either **sfaira** or a predefined mapping (`symbols_dict.json`). The LINCS dataset is already standardized.
#
# 2. **Identifying Shared Genes**: We then compute the intersection of the gene IDs (`gene_id`) inside LINCS and SciPlex. Both datasets are then filtered to retain only these shared genes.
#
# 3. **Reindexing**: The LINCS dataset is reindexed to match the order of genes in the SciPlex dataset.

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
import logging

logging.basicConfig(level=logging.INFO)
from notebook_utils import suppress_output

import scanpy as sc
with suppress_output():
    sc.set_figure_params(dpi=80, frameon=False)
    sc.logging.print_header()
    warnings.filterwarnings('ignore')

%load_ext autoreload
%autoreload 2

# ## Load data

# Load lincs
adata_lincs = sc.read(DATA_DIR/'lincs_full_smiles.h5ad')

# Load combinatorial sciplex
logging.info("Loading combinatorial sciplex data")
adata_sciplex = sc.read(DATA_DIR/'combo_sciplex_prep_hvg_filtered.h5ad')
logging.info("Combinatorial sciplex data loaded")

# Add gene_id to sciplex if not already present
if 'gene_id' not in adata_sciplex.var:
    adata_sciplex.var['gene_id'] = adata_sciplex.var.index.str.split('.').str[0]

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

# Identify genes that are shared between lincs and combinatorial sciplex

# For lincs
adata_lincs.var['gene_id'] = adata_lincs.var_names.map(symbols_dict)
adata_lincs.var['in_sciplex'] = adata_lincs.var.gene_id.isin(adata_sciplex.var.gene_id)

# For combinatorial sciplex
adata_sciplex.var['in_lincs'] = adata_sciplex.var.gene_id.isin(adata_lincs.var.gene_id)

# ## Preprocess sciplex dataset if needed
# Note: The combinatorial dataset is already preprocessed, so we skip preprocessing steps

# Reindex the datasets to match gene order
sciplex_ids = pd.Index(adata_sciplex.var.gene_id)
lincs_idx = [sciplex_ids.get_loc(_id) for _id in adata_lincs.var.gene_id[adata_lincs.var.in_sciplex]]
non_lincs_idx = [sciplex_ids.get_loc(_id) for _id in adata_sciplex.var.gene_id if not adata_lincs.var.gene_id.isin([_id]).any()]
lincs_idx.extend(non_lincs_idx)

adata_sciplex = adata_sciplex[:, lincs_idx].copy()

# Save the matched genes dataset
fname = PROJECT_DIR/'datasets'/'combo_sciplex_matched_genes_lincs.h5ad'
sc.write(fname, adata_sciplex)

# Check that it worked
sc.read(fname)

# ## Subselect to shared only shared genes
adata_lincs = adata_lincs[:, adata_lincs.var.in_sciplex].copy() 
adata_sciplex = adata_sciplex[:, adata_sciplex.var.in_lincs].copy()

# Print some stats about the gene matching
print("\nGene matching statistics:")
print(f"Number of genes in LINCS: {adata_lincs.shape[1]}")
print(f"Number of genes in combinatorial sciplex: {adata_sciplex.shape[1]}")
print(f"Number of shared genes: {sum(adata_sciplex.var.in_lincs)}")

# ## Save adata objects with shared genes only
# Save combinatorial sciplex with LINCS genes
fname = PROJECT_DIR/'datasets'/'combo_sciplex_lincs_genes.h5ad'
sc.write(fname, adata_sciplex)

# Save LINCS with combinatorial sciplex genes
fname_lincs = PROJECT_DIR/'datasets'/'lincs_full_smiles_combo_sciplex_genes.h5ad'
sc.write(fname_lincs, adata_lincs) 