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

# # LINCS SMILES Integration Notebook
#
# **Requires:**
# - `lincs_full_pp.h5ad` or `lincs_pp.h5ad`
#
# **Outputs:**
# - `lincs_full_smiles.h5ad` or `lincs_smiles.h5ad`
#
# ## Description
#
# The aim of this notebook is to integrate SMILES data into the LINCS dataset from a previous notebook.
#
# ### Loading LINCS and Reference Data
#
# The notebook begins by loading two primary datasets:
#
# 1. **LINCS Dataset (`adata_in`)**: Contains perturbation IDs (`pert_id`) representing different drugs or compounds.
#
# 2. **Reference Dataset (`reference_df`)**: Loaded from a TSV file (`GSE92742_Broad_LINCS_pert_info.txt`), which provides `pert_id` and the corresponding `canonical_smiles`.
#
# Both datasets contain `pert_id` columns, which are used for merging.
#
# ### Left Merge Between AnnData and SMILES
#
# - The reference dataset is restricted to include only drugs present in the LINCS dataset (`adata.obs.pert_id`).
# - A left merge is performed on `adata.obs` with `reference_df` using `pert_id` as the key, adding the `canonical_smiles` column to `adata.obs`.
#
# ### Cleaning and Additional Validation
#
# 1. **Removing Invalid SMILES**:
#    - The cleaning process involves removing invalid or restricted SMILES strings such as `-666`, `'restricted'`, or `NaN`.
# 2. **Validation with RDKit**:
#    - RDKit is used to validate chemical structures, ensuring that only valid SMILES are retained.
# 3. **Filtering Perturbations**:
#    - Perturbations (`pert_id`) with insufficient replicates or invalid dose values (e.g., `pert_dose` of `-666`) are removed to ensure a robust dataset.
#
#
#
#
#

# +
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
import pandas as pd
import scanpy as sc
from rdkit import Chem
from chemCPA.paths import DATA_DIR, PROJECT_DIR
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import sys
sys.path.append(root_dir)
import raw_data.datasets as datasets
import logging

logging.basicConfig(level=logging.INFO)
from notebook_utils import suppress_output

with suppress_output():
    sc.set_figure_params(dpi=80, frameon=False)
    sc.logging.print_header()
    warnings.filterwarnings('ignore')
# -

# %load_ext autoreload
# %autoreload 2

# ### Loading LINCS and reference data

# +
full = True
load_adata = True 

if full:
    adata_in = DATA_DIR / 'lincs_full_pp.h5ad'
    adata_out =  PROJECT_DIR / 'datasets' / 'lincs_full_smiles.h5ad' 
else: 
    adata_in = DATA_DIR / 'lincs_pp.h5ad'
    adata_out = PROJECT_DIR / 'datasets' / 'lincs_smiles.h5ad'  

    
logging.info(f"Starting to load in data from {adata_in}")
adata = sc.read(adata_in) if load_adata else None
logging.info(f"Data loaded from {adata_in}")
# -

# Checking number of drugs for LINCS

pert_id_unique = pd.Series(np.unique(adata.obs.pert_id))
print(f"# of unique perturbations: {len(pert_id_unique)}")

# Loading reference dataframe that contains SMILES 
# restricting to `'pert_id'` and `'canonical_smiles'`

reference_df = pd.read_csv(datasets.lincs_pert_info(), delimiter = "\t")
reference_df = reference_df.loc[reference_df.pert_id.isin(pert_id_unique), ['pert_id', 'canonical_smiles']]
reference_df.canonical_smiles.value_counts()

cond = ~pert_id_unique.isin(reference_df.pert_id)
print(f"From {len(pert_id_unique)} total drugs, {cond.sum()} were not part of the reference dataframe.")

# Adding `'canoncical_smiles'` column to `adata.obs` via `pd.merge`

adata.obs = adata.obs.reset_index().merge(reference_df, how="left").set_index('index')

# Removing invalid SMILES strings 

adata.obs.pert_id

reference_df

adata.obs.loc[:, 'canonical_smiles'] = adata.obs.canonical_smiles.astype('str')
invalid_smiles = adata.obs.canonical_smiles.isin(['-666', 'restricted', 'nan'])
print(f'Among {len(adata)} observations, {100*invalid_smiles.sum()/len(adata):.2f}% ({invalid_smiles.sum()}) have an invalid SMILES string')
adata = adata[~invalid_smiles]

# Remove invalid `'pert_dose'` value: `-666`

cond = adata.obs.pert_dose.isin([-666])
adata = adata[~cond]
print(f"A total of {cond.sum()} observations have invalid dose values")

drugs_validation = adata.obs.canonical_smiles.value_counts() < 6
valid_drugs = drugs_validation.index[~drugs_validation]
cond = adata.obs.canonical_smiles.isin(valid_drugs)
print(f"A total of {(~cond).sum()} observation belong to drugs which do not have enough replicates")
adata = adata[cond]

# Checking that SMILES are valid according to `rdkit` 

# +


def check_smiles(smiles):
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    if m is None:
        print('invalid SMILES')
        return False
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            print('invalid chemistry')
            return False
    return True

def remove_invalid_smiles(dataframe, smiles_key: str = 'SMILES', return_condition: bool = False):
    unique_drugs = pd.Series(np.unique(dataframe[smiles_key]))
    valid_drugs = unique_drugs.apply(check_smiles)
    print(f"A total of {(~valid_drugs).sum()} have invalid SMILES strings")
    _validation_map = dict(zip(unique_drugs, valid_drugs))
    cond = dataframe[smiles_key].apply(lambda x: _validation_map[x])
    if return_condition: 
        return cond
    dataframe = dataframe[cond].copy()
    return dataframe

adata
# -

cond = remove_invalid_smiles(adata.obs, smiles_key='canonical_smiles', return_condition=True)
adata = adata[cond]

# ### Add additional drugbank info to `adata.obs`

# +
drugbank_path = Path(datasets.drugbank_all())

if drugbank_path.exists(): 
    drugbank_df = pd.read_csv(drugbank_path)
else: 
    print(f'Invalid path: {drugbank_path}')

# +
from rdkit.Chem import CanonSmiles

drugs_canonical = pd.Series(np.unique(adata.obs.canonical_smiles)).apply(CanonSmiles)
db_canonical_smiles = drugbank_df.SMILES.apply(CanonSmiles)
n_overlap = drugs_canonical.isin(db_canonical_smiles).sum()
print(f'From a total of {len(drugs_canonical)}, {100*n_overlap/len(drugs_canonical):.2f}% ({n_overlap}) is also available in drugbank.')
# -

cond = db_canonical_smiles.isin(drugs_canonical)
drugbank_df.loc[cond, ['ATC_level_1']].value_counts()

# ### Add `train`, `test`, `ood` split for full lincs dataset (if not already part in `adata.obs`)

# +
from sklearn.model_selection import train_test_split

if 'split' not in list(adata.obs):
    print("Addig 'split' to 'adata.obs'.")
    unique_drugs = np.unique(adata.obs.canonical_smiles)
    drugs_train, drugs_tmp = train_test_split(unique_drugs, test_size=0.2, random_state=42)
    drugs_val, drugs_test = train_test_split(drugs_tmp, test_size=0.5, random_state=42)

    adata.obs['split'] = 'train'
    adata.obs.loc[adata.obs.canonical_smiles.isin(drugs_val), 'split'] = 'test'
    adata.obs.loc[adata.obs.canonical_smiles.isin(drugs_test), 'split'] = 'ood'
# -

# ### Check that `.obs.split=='test'` has sufficient samples for `pert_id` and `cell_id`

adata.obs.split.value_counts()

cond_test = adata.obs.split.isin(['test'])
adata.obs.loc[cond_test, 'cell_id'].value_counts()

adata.obs.loc[cond_test, 'pert_id'].value_counts()

# +
pert_count_treshold = 5
cov_count_treshold = 20

pert_id_neg = adata.obs.loc[cond_test, 'pert_id'].value_counts() < pert_count_treshold
print(f"pert_id: {pert_id_neg.sum()}/{len(pert_id_neg)} converted back to 'train' due to insufficient # of samples.")

cov_id_neg = adata.obs.loc[cond_test, 'cell_id'].value_counts() < cov_count_treshold
print(f"cell_id: {cov_id_neg.sum()}/{len(cov_id_neg)} converted back to 'train' due to insufficient # of samples.")

cond = cond_test & adata.obs.pert_id.isin(pert_id_neg.index[pert_id_neg])
cond |= cond_test & adata.obs.cell_id.isin(cov_id_neg.index[cov_id_neg])
# -

adata.obs['split1'] = adata.obs.split.copy()
adata.obs.loc[cond, 'split1'] = 'train'
print(f"split['test']: {cond.sum()}/{len(cond)} samples are converted back to 'train'.")

adata.obs.split1.value_counts()

# ### Add random split

adata.obs_names

train_obs, val_test_obs = train_test_split(adata.obs_names, test_size=0.15, random_state=42)
val_obs, test_obs = train_test_split(val_test_obs, test_size=0.5, random_state=42)

# +
adata.obs['random_split'] = ''
adata.obs.loc[train_obs, 'random_split'] = 'train'
adata.obs.loc[val_obs, 'random_split'] = 'test'
adata.obs.loc[test_obs, 'random_split'] = 'ood'


adata.obs['random_split'].value_counts() 
# -

# Check that perturbations occur in train split (no explicit ood!)

len(adata.obs.loc[adata.obs.random_split == 'train', 'pert_id'].unique()) 

len(adata.obs.pert_id.unique())

# ## Safe adata

logging.info(f"Writing file to disk at {adata_out}")
adata.write(adata_out)
logging.info(f"File was written successfully at {adata_out}.")
adata

# ### Loading the result for `adata_out`

adata = sc.read(adata_out)

# **Additional**: Check that `adata.uns[rank_genes_groups_cov]` has all entries in `adata.obs.cov_drug_name` as keys

for i, k in enumerate(adata.obs.cov_drug_name.unique()):
    try: 
        adata.uns['rank_genes_groups_cov'][k]
    except: 
        print(f"{i}: {k}") if 'DMSO' not in k else None


