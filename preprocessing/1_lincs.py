# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 1 LINCS
#
# **Requires**
# lincs_full.h5ad / lincs.h5ad
#
# **Outputs**
# lincs_full_pp.h5ad / lincs_pp.h5ad
#
# ## Description
#
# This notebook processes gene expression data from the LINCS dataset:
#
# 1. **Data Cleaning**: Loads LINCS data, cleans columns, and renames key fields.
# 2. **Filtering Insufficient Conditions**: Filters out conditions with fewer than 5 samples.
# 3. **Calculating Differentially Expressed Genes (DEGs)**: Identifies the top 50 genes most differentially expressed for each condition compared to the control (`DMSO`).
# 4. **Creating Data Splits**: Defines `'train'`, `'ood'`, and `'test'` splits for model training and evaluation:
#    - **OOD**: A random 10% selection from the samples with top occurring conditions, assigned to `'ood'`.
#    - **Test**: 16% of the remaining observations assigned to `'test'`.
#    - **Train**: The rest of the observations assigned to `'train'`.
#
#
#
#
#
#

# +
import os
import warnings

import numpy as np
import pandas as pd

from scipy import sparse
from tqdm.auto import tqdm

from chemCPA.helper import rank_genes_groups_by_cov
from chemCPA.paths import DATA_DIR
from pathlib import Path
import sys
import logging
from notebook_utils import suppress_output
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import raw_data.datasets as datasets


import scanpy as sc
with suppress_output():
    sc.set_figure_params(dpi=100, frameon=False)
    sc.logging.print_header()
    warnings.filterwarnings('ignore')

# logging.info is visible when running as python script 
if not any('ipykernel' in arg for arg in sys.argv):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
# -

# ## Load data
#

# Get the absolute path to the chemCPA root directory
full = True 
load_adata = True
# Ensure adata_path is a Path object
adata_path = Path(datasets.lincs_full()) if full else DATA_DIR / datasets.lincs()
logging.info(f"Starting to load in data from {adata_path}")
adata = sc.read(adata_path) if load_adata else None
logging.info(f"Data loaded from {adata_path}")




# # Rename columns & clean up columns

# +
logging.info("Renaming and cleaning up columns")
import re

def remove_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)

adata.obs['condition'] = adata.obs['pert_iname'].apply(remove_non_alphanumeric)
adata.obs['cell_type'] = adata.obs['cell_id']
adata.obs['dose_val'] = adata.obs['pert_dose'].astype(float) / np.max(adata.obs['pert_dose'].astype(float))
adata.obs['cov_drug_dose_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.condition.astype(str) + '_' + adata.obs.dose_val.astype(str)
adata.obs['cov_drug_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.condition.astype(str)
adata.obs['eval_category'] = adata.obs['cov_drug_name']
adata.obs['control'] = (adata.obs['condition'] == 'DMSO').astype(int)

# adata.obs['cov_drug_dose_name'] = adata.obs['cov_drug_dose_name'].str.replace('/','|')
# -

pd.crosstab(adata.obs.condition, adata.obs.cell_type)

drug_abundance = adata.obs.condition.value_counts()
suff_drug_abundance = drug_abundance.index[drug_abundance>5]

# Delete conditions isufficient # of observations
adata = adata[adata.obs.condition.isin(suff_drug_abundance)].copy()
adata 
logging.info("Finished cleaning up columns")

# Calculate differential genes manually, such that the genes are the same per condition.

# +
logging.info("Processing DEGs")
# %%time

de_genes = {}
de_genes_quick = {}

adata_df = adata.to_df()
adata_df = adata_df.join(adata.obs['condition'])  # Ensures correct alignment
dmso = adata_df[adata_df.condition == "DMSO"].mean(numeric_only=True)


for cond, df in tqdm(adata_df.groupby('condition')): 
    if cond != 'DMSO':
        drug_mean = df.mean(numeric_only=True)
        de_50_idx = np.argsort(abs(drug_mean - dmso))[-50:]
        de_genes_quick[cond] = drug_mean.index[de_50_idx].values

if full: 
    de_genes = de_genes_quick
else:
    sc.tl.rank_genes_groups(
        adata,
        groupby='condition', 
        reference='DMSO',
        rankby_abs=True,
        n_genes=50
    )
    for cond in tqdm(np.unique(adata.obs['condition'])):
        if cond != 'DMSO':
            df = sc.get.rank_genes_groups_df(adata, group=cond)
            de_genes[cond] = df['names'][:50].values

logging.info("Completed processing DEGs")


# -

# Mapping from `rank_genes_groups_cov` might cause problems when drug contains '_'

# +
def extract_drug(cond): 
    split = cond.split('_')
    if len(split) == 2: 
        return split[-1]
    return '_'.join(split[1:-1])

adata.obs['cov_drug_dose_name'].apply(lambda s: len(s.split('_'))).value_counts()
adata.obs['eval_category'].apply(lambda s: len(s.split('_'))).value_counts()
# -

adata.uns['rank_genes_groups_cov'] = {cat: de_genes_quick[extract_drug(cat)] for cat in adata.obs.eval_category.unique() if extract_drug(cat) != 'DMSO'}

adata.uns['rank_genes_groups_cov']

# +
adata.obs['split'] = 'train'

# take ood from top occurring perturbations to avoid losing data on low occ ones
ood_idx = sc.pp.subsample(
    adata[adata.obs.condition.isin(list(adata.obs.condition.value_counts().index[1:50]))],
    .1,
    copy=True
).obs.index
adata.obs['split'].loc[ood_idx] = 'ood'

# take test from a random subsampling of the rest
test_idx = sc.pp.subsample(
    adata[adata.obs.split != 'ood'],
    .16,
    copy=True
).obs.index
adata.obs['split'].loc[test_idx] = 'test'
# -

pd.crosstab(adata.obs['split'], adata.obs['condition'])

try: 
    del(adata.uns['rank_genes_groups'])  # too large
except: 
    print('All good.')

logging.info("Converting to sparse matrix")
# code compatibility
adata.X = sparse.csr_matrix(adata.X)
logging.info("Finished converting to sparse matrix")

output_path = adata_path.with_name(adata_path.stem + "_pp.h5ad")
logging.info(f"Writing file to disk at {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
sc.write(output_path, adata)
logging.info(f"File was written successfully at {output_path}.")

# ### Check that `adata.uns[rank_genes_groups_cov]` has all entries in `adata.obs.cov_drug_name` as keys

for i, k in enumerate(adata.obs.eval_category.unique()):
    try: 
        adata.uns['rank_genes_groups_cov'][k]
    except: 
        print(f"{i}: {k}") if 'DMSO' not in k else None

# ### Checking the same for the stored adata object

adata_2 = sc.read(output_path)

for i, k in enumerate(adata_2.obs.eval_category.unique()):
    try: 
        adata_2.uns['rank_genes_groups_cov'][k]
    except: 
        print(f"{i}: {k}") if 'DMSO' not in k else None

set(list(adata.uns['rank_genes_groups_cov'])) - set((list(adata_2.uns['rank_genes_groups_cov'])))

set((list(adata_2.uns['rank_genes_groups_cov']))) - set(list(adata.uns['rank_genes_groups_cov']))

len(list(adata_2.uns["rank_genes_groups_cov"].keys()))

adata.obs["dose_val"].value_counts()


