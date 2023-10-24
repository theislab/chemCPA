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
# **Requires**
#
# **Output**
#
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from rdkit import Chem

from chemCPA.paths import DATA_DIR, PROJECT_DIR

sc.set_figure_params(dpi=80, frameon=False)
sc.logging.print_header()

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Loading LINCS and reference data

# %%
full = True
load_adata = True

if full:
    adata_in = DATA_DIR / "lincs_full_pp.h5ad"
    adata_out = PROJECT_DIR / "datasets" / "lincs_full_smiles.h5ad"
else:
    adata_in = DATA_DIR / "lincs_pp.h5ad"
    adata_out = PROJECT_DIR / "datasets" / "lincs_smiles.h5ad"


adata = sc.read(adata_in) if load_adata else None

# %% [markdown]
# Checking number of drugs for LINCS

# %%
pert_id_unique = pd.Series(np.unique(adata.obs.pert_id))
print(f"# of unique perturbations: {len(pert_id_unique)}")

# %% [markdown]
# Loading reference dataframe that contains SMILES
# restricting to `'pert_id'` and `'canonical_smiles'`

# %%
reference_df = pd.read_csv(
    DATA_DIR / "GSE92742_Broad_LINCS_pert_info.txt", delimiter="\t"
)
reference_df = reference_df.loc[
    reference_df.pert_id.isin(pert_id_unique), ["pert_id", "canonical_smiles"]
]
reference_df.canonical_smiles.value_counts()

# %%
cond = ~pert_id_unique.isin(reference_df.pert_id)
print(
    f"From {len(pert_id_unique)} total drugs, {cond.sum()} were not part of the reference dataframe."
)

# %% [markdown]
# Adding `'canoncical_smiles'` column to `adata.obs` via `pd.merge`

# %%
adata.obs = adata.obs.reset_index().merge(reference_df, how="left").set_index("index")

# %% [markdown]
# Removing invalid SMILES strings

# %%
adata.obs.pert_id

# %%
reference_df

# %%
adata.obs.loc[:, "canonical_smiles"] = adata.obs.canonical_smiles.astype("str")
invalid_smiles = adata.obs.canonical_smiles.isin(["-666", "restricted", "nan"])
print(
    f"Among {len(adata)} observations, {100*invalid_smiles.sum()/len(adata):.2f}% ({invalid_smiles.sum()}) have an invalid SMILES string"
)
adata = adata[~invalid_smiles]

# %% [markdown]
# Remove invalid `'pert_dose'` value: `-666`

# %%
cond = adata.obs.pert_dose.isin([-666])
adata = adata[~cond]
print(f"A total of {cond.sum()} observations have invalid dose values")

# %%
drugs_validation = adata.obs.canonical_smiles.value_counts() < 6
valid_drugs = drugs_validation.index[~drugs_validation]
cond = adata.obs.canonical_smiles.isin(valid_drugs)
print(
    f"A total of {(~cond).sum()} observation belong to drugs which do not have enough replicates"
)
adata = adata[cond]

# %% [markdown]
# Checking that SMILES are valid according to `rdkit`

# %%


def check_smiles(smiles):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        print("invalid SMILES")
        return False
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            print("invalid chemistry")
            return False
    return True


def remove_invalid_smiles(
    dataframe, smiles_key: str = "SMILES", return_condition: bool = False
):
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

# %%
cond = remove_invalid_smiles(
    adata.obs, smiles_key="canonical_smiles", return_condition=True
)
adata = adata[cond]

# %% [markdown]
# ### Add additional drugbank info to `adata.obs`

# %%
drugbank_path = DATA_DIR / "drug_bank" / "drugbank_all.csv"

if drugbank_path.exists():
    drugbank_df = pd.read_csv(drugbank_path)
else:
    print(f"Invalid path: {drugbank_path}")

# %%
from rdkit.Chem import CanonSmiles

drugs_canonical = pd.Series(np.unique(adata.obs.canonical_smiles)).apply(CanonSmiles)
db_canonical_smiles = drugbank_df.SMILES.apply(CanonSmiles)
n_overlap = drugs_canonical.isin(db_canonical_smiles).sum()
print(
    f"From a total of {len(drugs_canonical)}, {100*n_overlap/len(drugs_canonical):.2f}% ({n_overlap}) is also available in drugbank."
)

# %%
cond = db_canonical_smiles.isin(drugs_canonical)
drugbank_df.loc[cond, ["ATC_level_1"]].value_counts()

# %% [markdown]
# ### Add `train`, `test`, `ood` split for full lincs dataset (if not already part in `adata.obs`)

# %%
from sklearn.model_selection import train_test_split

if "split" not in list(adata.obs):
    print("Addig 'split' to 'adata.obs'.")
    unique_drugs = np.unique(adata.obs.canonical_smiles)
    drugs_train, drugs_tmp = train_test_split(
        unique_drugs, test_size=0.2, random_state=42
    )
    drugs_val, drugs_test = train_test_split(drugs_tmp, test_size=0.5, random_state=42)

    adata.obs["split"] = "train"
    adata.obs.loc[adata.obs.canonical_smiles.isin(drugs_val), "split"] = "test"
    adata.obs.loc[adata.obs.canonical_smiles.isin(drugs_test), "split"] = "ood"

# %% [markdown]
# ### Check that `.obs.split=='test'` has sufficient samples for `pert_id` and `cell_id`

# %%
adata.obs.split.value_counts()

# %%
cond_test = adata.obs.split.isin(["test"])
adata.obs.loc[cond_test, "cell_id"].value_counts()

# %%
adata.obs.loc[cond_test, "pert_id"].value_counts()

# %%
pert_count_treshold = 5
cov_count_treshold = 20

pert_id_neg = adata.obs.loc[cond_test, "pert_id"].value_counts() < pert_count_treshold
print(
    f"pert_id: {pert_id_neg.sum()}/{len(pert_id_neg)} converted back to 'train' due to insufficient # of samples."
)

cov_id_neg = adata.obs.loc[cond_test, "cell_id"].value_counts() < cov_count_treshold
print(
    f"cell_id: {cov_id_neg.sum()}/{len(cov_id_neg)} converted back to 'train' due to insufficient # of samples."
)

cond = cond_test & adata.obs.pert_id.isin(pert_id_neg.index[pert_id_neg])
cond |= cond_test & adata.obs.cell_id.isin(cov_id_neg.index[cov_id_neg])

# %%
adata.obs["split1"] = adata.obs.split.copy()
adata.obs.loc[cond, "split1"] = "train"
print(f"split['test']: {cond.sum()}/{len(cond)} samples are converted back to 'train'.")

# %%
adata.obs.split1.value_counts()

# %% [markdown]
# ### Add random split

# %%
adata.obs_names

# %%
train_obs, val_test_obs = train_test_split(
    adata.obs_names, test_size=0.15, random_state=42
)
val_obs, test_obs = train_test_split(val_test_obs, test_size=0.5, random_state=42)

# %%
adata.obs["random_split"] = ""
adata.obs.loc[train_obs, "random_split"] = "train"
adata.obs.loc[val_obs, "random_split"] = "test"
adata.obs.loc[test_obs, "random_split"] = "ood"


adata.obs["random_split"].value_counts()

# %% [markdown]
# Check that perturbations occur in train split (no explicit ood!)

# %%
len(adata.obs.loc[adata.obs.random_split == "train", "pert_id"].unique())

# %%
len(adata.obs.pert_id.unique())

# %% [markdown]
# ## Safe adata

# %%
adata.write(adata_out)
adata

# %% [markdown]
# ### Loading the result for `adata_out`

# %%
adata = sc.read(adata_out)

# %% [markdown]
# **Additional**: Check that `adata.uns[rank_genes_groups_cov]` has all entries in `adata.obs.cov_drug_name` as keys

# %%
for i, k in enumerate(adata.obs.cov_drug_name.unique()):
    try:
        adata.uns["rank_genes_groups_cov"][k]
    except:
        print(f"{i}: {k}") if "DMSO" not in k else None

# %%
