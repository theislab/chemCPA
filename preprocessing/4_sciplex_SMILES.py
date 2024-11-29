# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: chemical_CPA
#     language: python
#     name: python3
# ---

# # 4 SCIPLEX SMILES
#
# This is an updated version of `sciplex_SMILES.ipynb` which relies on a `drug_dict` to assign SMILES strings.  
# The `sciplex_SMILES.ipynb` notebook is not applicable to the full sciplex data as it relies on the `.obs_names`.  
# Hence, the second half of the dataset (left out in the original CPA publication) would be left without SMILES entries. 
#
# **Requires**
# * `'sciplex3_matched_genes_lincs.h5ad'`
# * `'sciplex3_lincs_genes.h5ad'`
# * `'trapnell_final_V7.h5ad'`
#
# **Output**
# * `'trapnell_cpa(_lincs_genes).h5ad'`
# * `'trapnell_cpa_subset(_lincs_genes).h5ad'`
#
#
# ## Description
# This script assigns SMILES strings to drug conditions in the sciplex dataset, serving as a counterpart to `2_lincs_SMILES.py` but handling sciplex data. Below is a summary of its key steps:
#
# 1. **Load Data**: The script uses either `sciplex3_lincs_genes.h5ad` or `sciplex3_matched_genes_lincs.h5ad` as the target dataset to which SMILES strings are added. The choice depends on the `LINCS_GENES` flag: if `LINCS_GENES` is `True`, the dataset with the LINCS gene subset (`sciplex3_lincs_genes.h5ad`) is used; if `False`, the matched genes dataset (`sciplex3_matched_genes_lincs.h5ad`) is used.
#
# 2. **Create and Assign SMILES**: A dictionary (`drug_dict`) is created by zipping the `condition` and `SMILES` columns from `trapnell_final_V7.h5ad`. The script assigns SMILES to the target dataset by applying `drug_dict` to the `condition` column.
#
# 3. **Canonicalization and Validation**: The SMILES strings are validated using `rdkit` to make them canonical. The notebook also checks that each drug condition is assigned a unique SMILES string.
#
# 4. **Subset Creation**: A subset of the target dataset is created by sampling up to 50 observations per drug condition to reduce data size. The subsets are concatenated into `adata_cpa_subset`.
#
# 5. **Output**: Depending on the `LINCS_GENES` flag, the script creates two output files:
#
#    - If `LINCS_GENES` is `True`, the produced files are the whole set `trapnell_cpa_lincs_genes.h5ad` and the subset as `trapnell_cpa_subset_lincs_genes.h5ad`.
#    - Analogously if`LINCS_GENES` is `False`, the produced files are `trapnell_cpa.h5ad` and `trapnell_cpa_subset.h5ad`.
#
#
#

#
# ## Imports

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import scanpy as sc
from rdkit import Chem

import warnings
from chemCPA.paths import DATA_DIR, PROJECT_DIR

import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ## Load data
# Note: Run notebook for both adata objects (LINCS_GENES)

# +
# Switch between 977 (True) and 2000 (False) gene set 
LINCS_GENES = True 

adata_cpa = sc.read(DATA_DIR/f"sciplex3_{'matched_genes_lincs' if not LINCS_GENES else 'lincs_genes'}.h5ad") 
adata_cpi = sc.read(datasets.trapnell_final_v7())
# -

adata_cpa

# Determine output directory

adata_out = DATA_DIR / f"trapnell_cpa{'_lincs_genes' if LINCS_GENES else ''}.h5ad"
adata_out_subset = DATA_DIR / f"trapnell_cpa_subset{'_lincs_genes' if LINCS_GENES else ''}.h5ad"

# Overview over adata files

# +
# adata_cpa

# +
# adata_cpi
# -

# __________
# ### Drug is combined with acid

# In the `adata_cpi` we distinguish between `'ENMD-2076'` and `'ENMD-2076 L-(+)-Tartaric acid'`.  
# They have different also different SMILES strings in `.obs.SMILES`. 
# Since we do not keep this different in the `.obs.condition` columns,  
# which is a copy of `.obs.product_name` for `adata_cpa`, see `'lincs_sciplex_gene_matching.ipynb'`,  
# I am ignoring this. As result we only have 188 drugs in the sciplex dataset.

adata_cpi.obs.product_name[adata_cpi.obs.SMILES == 'O[C@H]([C@@H](O)C(O)=O)C(O)=O.CN1CCN(CC1)C1=NC(\\C=C\\C2=CC=CC=C2)=NC(NC2=NNC(C)=C2)=C1 |r,c:24,26,28,36,38,t:17,22,32|']

# +
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

# Test in a kinase inhibitor
mol = Chem.MolFromSmiles("CN1CCN(CC1)C1=CC(NC2=NNC(C)=C2)=NC(\\C=C\\C2=CC=CC=C2)=N1")
# Default
mol
# -

# Test in a kinase inhibitor
mol = Chem.MolFromSmiles("O[C@H]([C@@H](O)C(O)=O)C(O)=O.CN1CCN(CC1)C1=NC(\\C=C\\C2=CC=CC=C2)=NC(NC2=NNC(C)=C2)=C1")
# Default
mol

# ___________

# ## Create drug SMILES dict

drug_dict = dict(zip(adata_cpi.obs.condition, adata_cpi.obs.SMILES))

# The dict has 188 different entries

len(drug_dict)

# Checking that the `'ENMD-2076'` entry does not include the adid:

Chem.MolFromSmiles(drug_dict['ENMD-2076'])

# This is a good wat to check the unique `(drug, smiles)` combinations that exist in the `adata_cpi`

# +
# np.unique([f'{condition}_{smiles}' for condition, smiles in list(zip(adata_cpi.obs.condition, adata_cpi.obs.SMILES))])
# -

# ## Rename drug `(+)-JQ1`
# This had a different name in the old Sciplex dataset, where it was called `JQ1`. We rename it for consistency.

adata_cpa.obs["condition"] = adata_cpa.obs["condition"].cat.rename_categories({"(+)-JQ1": "JQ1"})

# ## Add SMILES to `adata_cpa`

adata_cpa.obs['SMILES'] = adata_cpa.obs.condition.map(drug_dict)

adata_cpa[adata_cpa.obs["condition"] == "JQ1"].obs["SMILES"].unique()

# ## Check that SMILES match `obs.condition` data
#
# Print some stats on the `condition` columns

print(f'We have {len(list(adata_cpa.obs.condition.value_counts().index))} drug names in adata_cpa: \n\n\t{list(adata_cpa.obs.condition.value_counts().index)}\n\n')
print(f'We have {len(list(adata_cpi.obs.condition.value_counts().index))} drug names in adata_cpi: \n\n\t{list(adata_cpi.obs.condition.value_counts().index)}')

# Check that assigned SMILES match the condition,  
# it should be just one smiles string per condition

(adata_cpa.obs.condition=='nan').sum()

# ### Check for nans

(adata_cpa.obs.condition=='nan').sum()

# ### Take care of `control` SMILES

counts = adata_cpa[adata_cpa.obs.condition=='control'].obs.SMILES.value_counts()
list(counts.index[counts>0])

# Add DMSO SMILES:`CS(C)=O`

adata_cpa.obs["SMILES"] = adata_cpa.obs["SMILES"].astype("category").cat.rename_categories({"": "CS(C)=O"})

adata_cpa.obs.loc[adata_cpa.obs.condition=='control', 'SMILES'].value_counts()

# ### Check double assigned condition

for pert, df in adata_cpa.obs.groupby('condition'):
    n_smiles = (df.SMILES.value_counts()!=0).sum()
    print(f"{pert}: {n_smiles}") if n_smiles > 1 else None

# Check that condition align with SMILES
#
# If everything is correct there should be no output

for pert, df in adata_cpa.obs.groupby('condition'):
    n_smiles = (df.SMILES.value_counts()!=0).sum()
    print(f"{pert}: {n_smiles}") if n_smiles > 1 else None

# ## Make SMILES canonical

# +
print(f'rdkit version: {rdkit.__version__}\n')

adata_cpa.obs.SMILES = adata_cpa.obs.SMILES.apply(Chem.CanonSmiles)
# -

# ## Add a random split to adata_cpa

# +
# This does not make sense

# from sklearn.model_selection import train_test_split

# if 'split' not in list(adata_cpa.obs):
#     print("Addig 'split' to 'adata_cpa.obs'.")
#     unique_drugs = np.unique(adata_cpa.obs.SMILES)
#     drugs_train, drugs_tmp = train_test_split(unique_drugs, test_size=0.2)
#     drugs_val, drugs_test = train_test_split(drugs_tmp, test_size=0.5)

#     adata_cpa.obs['split'] = 'train'
#     adata_cpa.obs.loc[adata_cpa.obs.SMILES.isin(drugs_val), 'split'] = 'test'
#     adata_cpa.obs.loc[adata_cpa.obs.SMILES.isin(drugs_test), 'split'] = 'ood'
# -

# ## Create subset `adata_cpa_subset` from `adata_cpa`

# +
adatas = []

for drug in np.unique(adata_cpa.obs.condition): 
    tmp = adata_cpa[adata_cpa.obs.condition == drug].copy()
    tmp = sc.pp.subsample(tmp, n_obs=50, copy=True, random_state=42)
    adatas.append(tmp)

adata_cpa_subset = adatas[0].concatenate(adatas[1:])
adata_cpa_subset.uns = adata_cpa.uns.copy()

adata_cpa_subset
# -

# ## Safe both adata objects

adata_cpa.write(adata_out)
adata_cpa_subset.write(adata_out_subset)

# ### Loading the result for `adata_out`

adata = sc.read(adata_out_subset)
adata.obs.dose.value_counts()

adata_cpa.uns["log1p"]



