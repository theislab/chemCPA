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
# * `'sciplex_combinatorial.h5ad'` (previously `'combo_sciplex_prep_hvg_filtered.h5ad'`)
#
# **Output**
# * `'combo_sciplex_matched_genes_lincs.h5ad'`
# * `'combo_sciplex_lincs_genes.h5ad'`
# * `'lincs_full_smiles_combo_sciplex_genes.h5ad'`
#
# ## Description 
#
# The goal of this notebook is to match and merge genes between the LINCS and combinatorial SciPlex datasets, resulting in the creation of three new datasets:
#
# - **`combo_sciplex_matched_genes_lincs.h5ad`**: Contains **combinatorial SciPlex observations**. Genes limited to the intersection of genes in both LINCS and SciPlex.
# - **`combo_sciplex_lincs_genes.h5ad`**: Same combinatorial SciPlex data, but only the shared genes.
# - **`lincs_full_smiles_combo_sciplex_genes.h5ad`**: LINCS data filtered to those shared genes as well.
#
# ### Gene Matching
#
# 1. **Gene ID Assignment**: SciPlex gene names are standardized to Ensembl gene IDs by extracting the primary identifier and using either **sfaira** or a predefined mapping (`symbols_dict.json`). The LINCS dataset is already standardized.
# 2. **Identifying Shared Genes**: We then compute the intersection of the gene IDs (`gene_id`) inside LINCS and SciPlex.
# 3. **Reindexing**: The LINCS dataset is reindexed to match the order of genes in the SciPlex dataset.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sfaira
import warnings
import logging

from chemCPA.paths import DATA_DIR, PROJECT_DIR

# Ensure we can import from the project directory
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

pd.set_option('display.max_columns', 100)

logging.basicConfig(level=logging.INFO)
from notebook_utils import suppress_output

import scanpy as sc
with suppress_output():
    sc.set_figure_params(dpi=80, frameon=False)
    sc.logging.print_header()
    warnings.filterwarnings('ignore')

# If using within a Jupyter environment:
# %load_ext autoreload
# %autoreload 2

############################################################
# 1) Load LINCS
############################################################
lincs_path = DATA_DIR / 'lincs_full_smiles.h5ad'
logging.info(f"Loading LINCS from {lincs_path}")
adata_lincs = sc.read(lincs_path)
logging.info("LINCS data loaded")

############################################################
# 2) Load combinatorial sciplex dataset
#    from raw_data.datasets import sciplex_combinatorial
#    ensuring it downloads if missing
############################################################
from raw_data.datasets import sciplex_combinatorial

sciplex_path = sciplex_combinatorial()  # This triggers a download if not present
logging.info(f"Loading combinatorial sciplex data from {sciplex_path}")
adata_sciplex = sc.read(sciplex_path)
logging.info("Combinatorial sciplex data loaded")

# Add gene_id if missing
if 'gene_id' not in adata_sciplex.var:
    adata_sciplex.var['gene_id'] = adata_sciplex.var.index.str.split('.').str[0]

############################################################
# 3) Map gene symbols to Ensembl IDs (if needed)
############################################################
try:
    import json
    with open(DATA_DIR / 'symbols_dict.json') as json_file:
        symbols_dict = json.load(json_file)
except FileNotFoundError:
    logging.info("No symbols_dict.json found, falling back to sfaira.")
    genome_container = sfaira.versions.genomes.GenomeContainer(
        organism="homo_sapiens", release="82"
    )
    symbols_dict = genome_container.symbol_to_id_dict
    symbols_dict.update({'PLSCR3':'ENSG00000187838'})  # Example addition

# Convert var_names in LINCS to gene_id if not already done
adata_lincs.var['gene_id'] = adata_lincs.var_names.map(symbols_dict)

############################################################
# 4) Identify shared genes
############################################################
adata_lincs.var['in_sciplex'] = adata_lincs.var['gene_id'].isin(adata_sciplex.var['gene_id'])
adata_sciplex.var['in_lincs'] = adata_sciplex.var['gene_id'].isin(adata_lincs.var['gene_id'])

############################################################
# 5) Reindex sciplex to match gene order
############################################################
sciplex_ids = pd.Index(adata_sciplex.var['gene_id'])
lincs_idx = [sciplex_ids.get_loc(_id) for _id in adata_lincs.var['gene_id'][adata_lincs.var['in_sciplex']]]
non_lincs_idx = [
    sciplex_ids.get_loc(_id)
    for _id in adata_sciplex.var['gene_id']
    if not adata_lincs.var['gene_id'].isin([_id]).any()
]
lincs_idx.extend(non_lincs_idx)
adata_sciplex = adata_sciplex[:, lincs_idx].copy()

############################################################
# 6) Save combined, matched dataset
############################################################
fname_matched = PROJECT_DIR / 'datasets' / 'combo_sciplex_matched_genes_lincs.h5ad'
sc.write(fname_matched, adata_sciplex)
logging.info(f"Matched genes dataset saved to {fname_matched}")

############################################################
# 7) Filter each dataset to *only* the shared genes
############################################################
adata_lincs_filtered = adata_lincs[:, adata_lincs.var['in_sciplex']].copy()
adata_sciplex_filtered = adata_sciplex[:, adata_sciplex.var['in_lincs']].copy()

logging.info("Filtering each dataset to only the shared genes.")

############################################################
# 8) Print gene matching stats
############################################################
shared_count = adata_sciplex_filtered.var['in_lincs'].sum()
print("\nGene matching statistics:")
print(f"Number of genes in LINCS: {adata_lincs_filtered.shape[1]}")
print(f"Number of genes in combinatorial sciplex: {adata_sciplex_filtered.shape[1]}")
print(f"Number of shared genes: {shared_count}")

############################################################
# 9) Save final output files
############################################################
fname_sciplex = PROJECT_DIR / 'datasets' / 'combo_sciplex_lincs_genes.h5ad'
sc.write(fname_sciplex, adata_sciplex_filtered)
logging.info(f"Combinatorial SciPlex (shared genes) saved to {fname_sciplex}")

fname_lincs = PROJECT_DIR / 'datasets' / 'lincs_full_smiles_combo_sciplex_genes.h5ad'
sc.write(fname_lincs, adata_lincs_filtered)
logging.info(f"LINCS (shared genes) saved to {fname_lincs}")

############################################################
# 10) Log summary info for all three outputs
############################################################
logging.info("----- Final Dataset Summaries -----")

# 1) combo_sciplex_matched_genes_lincs.h5ad
logging.info("Summary of combo_sciplex_matched_genes_lincs.h5ad:")
logging.info(f"Path: {fname_matched}")
logging.info(f"Shape: {adata_sciplex.shape}")
logging.info(f"Observations (cells): {adata_sciplex.n_obs}")
logging.info(f"Genes (columns): {adata_sciplex.n_vars}")
if 'gene_id' in adata_sciplex.var:
    logging.info(f"Unique gene_ids: {adata_sciplex.var['gene_id'].nunique()}")

# 2) combo_sciplex_lincs_genes.h5ad
logging.info("Summary of combo_sciplex_lincs_genes.h5ad:")
logging.info(f"Path: {fname_sciplex}")
logging.info(f"Shape: {adata_sciplex_filtered.shape}")
logging.info(f"Observations (cells): {adata_sciplex_filtered.n_obs}")
logging.info(f"Genes (columns): {adata_sciplex_filtered.n_vars}")
if 'gene_id' in adata_sciplex_filtered.var:
    logging.info(f"Unique gene_ids: {adata_sciplex_filtered.var['gene_id'].nunique()}")

# 3) lincs_full_smiles_combo_sciplex_genes.h5ad
logging.info("Summary of lincs_full_smiles_combo_sciplex_genes.h5ad:")
logging.info(f"Path: {fname_lincs}")
logging.info(f"Shape: {adata_lincs_filtered.shape}")
logging.info(f"Observations (cells): {adata_lincs_filtered.n_obs}")
logging.info(f"Genes (columns): {adata_lincs_filtered.n_vars}")
if 'gene_id' in adata_lincs_filtered.var:
    logging.info(f"Unique gene_ids: {adata_lincs_filtered.var['gene_id'].nunique()}")

logging.info("----- End of Summaries -----")

