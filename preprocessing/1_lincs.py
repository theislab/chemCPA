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

# %%
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.notebook import tqdm

sc.set_figure_params(dpi=100, frameon=False)
sc.logging.print_header()

# %%
import os

os.chdir("./../")
# %%
import warnings

from chemCPA.helper import rank_genes_groups_by_cov

warnings.filterwarnings("ignore")

# %%
full = False
load_adata = True
adata_in = "datasets/lincs_full.h5ad" if full else "datasets/lincs.h5ad"
adata = sc.read(adata_in) if load_adata else None

adata_out = "".join(adata_in.split(".")[:-1]) + "_pp.h5ad"
adata_out

# %%
adata.obs["condition"] = adata.obs["pert_iname"]
adata.obs["condition"] = adata.obs["condition"].str.replace("/", "|")

adata.obs["cell_type"] = adata.obs["cell_id"]
adata.obs["dose_val"] = adata.obs["pert_dose"]
adata.obs["cov_drug_dose_name"] = (
    adata.obs.cell_type.astype(str)
    + "_"
    + adata.obs.condition.astype(str)
    + "_"
    + adata.obs.dose_val.astype(str)
)
adata.obs["cov_drug_name"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)
adata.obs["eval_category"] = adata.obs["cov_drug_name"]
adata.obs["control"] = (adata.obs["condition"] == "DMSO").astype(int)

# adata.obs['cov_drug_dose_name'] = adata.obs['cov_drug_dose_name'].str.replace('/','|')

# %%
pd.crosstab(adata.obs.condition, adata.obs.cell_type)

# %%
drug_abundance = adata.obs.condition.value_counts()
suff_drug_abundance = drug_abundance.index[drug_abundance > 5]

# %%
# Delete conditions isufficient # of observations
adata = adata[adata.obs.condition.isin(suff_drug_abundance)].copy()
adata

# %% [markdown]
# Calculate differential genes manually, such that the genes are the same per condition.

# %%
# %%time

de_genes = {}
de_genes_quick = {}

adata_df = adata.to_df()
adata_df["condition"] = adata.obs.condition
dmso = adata_df[adata_df.condition == "DMSO"].mean()

for cond, df in tqdm(adata_df.groupby("condition")):
    if cond != "DMSO":
        drug_mean = df.mean()
        de_50_idx = np.argsort(abs(drug_mean - dmso))[-50:]
        de_genes_quick[cond] = drug_mean.index[de_50_idx].values

if full:
    de_genes = de_genes_quick

else:
    sc.tl.rank_genes_groups(
        adata, groupby="condition", reference="DMSO", rankby_abs=True, n_genes=50
    )
    for cond in tqdm(np.unique(adata.obs["condition"])):
        if cond != "DMSO":
            df = sc.get.rank_genes_groups_df(adata, group=cond)  # this takes a while
            de_genes[cond] = df["names"][:50].values


# %% [markdown]
# Mapping from `rank_genes_groups_cov` might cause problems when drug contains '_'


# %%
def extract_drug(cond):
    split = cond.split("_")
    if len(split) == 2:
        return split[-1]
    return "_".join(split[1:])


adata.obs["cov_drug_dose_name"].apply(lambda s: len(s.split("_"))).value_counts()
adata.obs["eval_category"].apply(lambda s: len(s.split("_"))).value_counts()

# %%
adata.uns["rank_genes_groups_cov"] = {
    cat: de_genes_quick[extract_drug(cat)]
    for cat in adata.obs.eval_category.unique()
    if extract_drug(cat) != "DMSO"
}

# %%
adata.uns["rank_genes_groups_cov"]

# %%
adata.obs["split"] = "train"

# take ood from top occurring perturbations to avoid losing data on low occ ones
ood_idx = sc.pp.subsample(
    adata[
        adata.obs.condition.isin(list(adata.obs.condition.value_counts().index[1:50]))
    ],
    0.1,
    copy=True,
).obs.index
adata.obs["split"].loc[ood_idx] = "ood"

# take test from a random subsampling of the rest
test_idx = sc.pp.subsample(adata[adata.obs.split != "ood"], 0.16, copy=True).obs.index
adata.obs["split"].loc[test_idx] = "test"

# %%
pd.crosstab(adata.obs["split"], adata.obs["condition"])

# %%
try:
    del adata.uns["rank_genes_groups"]  # too large
except:
    print("All good.")

# %%
# code compatibility
from scipy import sparse

adata.X = sparse.csr_matrix(adata.X)

# %%
sc.write(adata_out, adata)

# %%
print("all done.")

# %% [markdown]
# ### Check that `adata.uns[rank_genes_groups_cov]` has all entries in `adata.obs.cov_drug_dose_name` as keys

# %%
for i, k in enumerate(adata.obs.eval_category.unique()):
    try:
        adata.uns["rank_genes_groups_cov"][k]
    except:
        print(f"{i}: {k}") if "DMSO" not in k else None

# %% [markdown]
# ### Checking the same for the stored adata object

# %%
adata_2 = sc.read(adata_out)

# %%
for i, k in enumerate(adata_2.obs.eval_category.unique()):
    try:
        adata_2.uns["rank_genes_groups_cov"][k]
    except:
        print(f"{i}: {k}") if "DMSO" not in k else None

# %%
set(list(adata.uns["rank_genes_groups_cov"])) - set(
    (list(adata_2.uns["rank_genes_groups_cov"]))
)

# %%
set((list(adata_2.uns["rank_genes_groups_cov"]))) - set(
    list(adata.uns["rank_genes_groups_cov"])
)

# %%
