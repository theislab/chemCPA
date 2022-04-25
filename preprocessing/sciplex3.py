# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
# ---

# %%
import os

import numpy as np
import pandas as pd

# %%
import scanpy as sc

os.chdir("./../")
from chemCPA.helper import rank_genes_groups_by_cov

# %%
adatas = []
for i in range(5):
    adatas.append(sc.read(f"./datasets/sciplex_raw_chunk_{i}.h5ad"))
adata = adatas[0].concatenate(adatas[1:])

# %%
sc.pp.subsample(adata, fraction=0.5)
sc.pp.normalize_per_cell(adata)

# %%
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)

# %%
adata.obs["dose_val"] = adata.obs.dose.astype(float) / np.max(
    adata.obs.dose.astype(float)
)
adata.obs["dose_val"][adata.obs["product_name"].str.contains("Vehicle")] = 1.0
adata.obs["product_name"] = [x.split(" ")[0] for x in adata.obs["product_name"]]
adata.obs["product_name"][adata.obs["product_name"].str.contains("Vehicle")] = "control"
adata.obs["drug_dose_name"] = (
    adata.obs.product_name.astype(str) + "_" + adata.obs.dose_val.astype(str)
)
adata.obs["cov_drug_dose_name"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.drug_dose_name.astype(str)
)
adata.obs["condition"] = adata.obs.product_name.copy()
adata.obs["control"] = [
    1 if x == "Vehicle_1.0" else 0 for x in adata.obs.drug_dose_name.values
]
adata.obs["cov_drug"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)

# %%
from chemCPA.helper import rank_genes_groups_by_cov

rank_genes_groups_by_cov(
    adata, groupby="cov_drug", covariate="cell_type", control_group="control"
)

# %%
new_genes_dict = {}
for cat in adata.obs.cov_drug_dose_name.unique():
    if "control" not in cat:
        rank_keys = np.array(list(adata.uns["rank_genes_groups_cov"].keys()))
        bool_idx = [x in cat for x in rank_keys]
        genes = adata.uns["rank_genes_groups_cov"][rank_keys[bool_idx][0]]
        new_genes_dict[cat] = genes

# %%
adata.uns["rank_genes_groups_cov"] = new_genes_dict

# %% [markdown]
# # Split

# %% tags=[]
adata.obs["split"] = "train"  # reset
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
ood = adata.obs["condition"].isin(ho_drugs)
len(ho_drugs)

# %%
adata.obs["split"][ood & (adata.obs["dose_val"] == 1.0)] = "ood"
test_idx = sc.pp.subsample(
    adata[adata.obs["split"] != "ood"], 0.10, copy=True
).obs.index
adata.obs["split"].loc[test_idx] = "test"

# %%
pd.crosstab(adata.obs["split"], adata.obs["condition"])

# %%
adata.obs["split"].value_counts()

# %%
adata[adata.obs.split == "ood"].obs.condition.value_counts()

# %%
adata[adata.obs.split == "test"].obs.condition.value_counts()

# %% [markdown]
# Also a split which sees all data:

# %%
adata.obs["split_all"] = "train"
test_idx = sc.pp.subsample(adata, 0.10, copy=True).obs.index
adata.obs["split_all"].loc[test_idx] = "test"

# %%
adata.obs["ct_dose"] = (
    adata.obs.cell_type.astype("str") + "_" + adata.obs.dose_val.astype("str")
)

# %% [markdown]
# Round robin splits: dose and cell line combinations will be held out in turn.

# %%
i = 0
split_dict = {}

# %%
# single ct holdout
for ct in adata.obs.cell_type.unique():
    for dose in adata.obs.dose_val.unique():
        i += 1
        split_name = f"split{i}"
        split_dict[split_name] = f"{ct}_{dose}"

        adata.obs[split_name] = "train"
        adata.obs[split_name][adata.obs.ct_dose == f"{ct}_{dose}"] = "ood"

        test_idx = sc.pp.subsample(
            adata[adata.obs[split_name] != "ood"], 0.16, copy=True
        ).obs.index
        adata.obs[split_name].loc[test_idx] = "test"

        display(adata.obs[split_name].value_counts())

# %%
# double ct holdout
for cts in [("A549", "MCF7"), ("A549", "K562"), ("MCF7", "K562")]:
    for dose in adata.obs.dose_val.unique():
        i += 1
        split_name = f"split{i}"
        split_dict[split_name] = f"{cts[0]}+{cts[1]}_{dose}"

        adata.obs[split_name] = "train"
        adata.obs[split_name][adata.obs.ct_dose == f"{cts[0]}_{dose}"] = "ood"
        adata.obs[split_name][adata.obs.ct_dose == f"{cts[1]}_{dose}"] = "ood"

        test_idx = sc.pp.subsample(
            adata[adata.obs[split_name] != "ood"], 0.16, copy=True
        ).obs.index
        adata.obs[split_name].loc[test_idx] = "test"

        display(adata.obs[split_name].value_counts())

# %%
# triple ct holdout
for dose in adata.obs.dose_val.unique():
    i += 1
    split_name = f"split{i}"

    split_dict[split_name] = f"all_{dose}"
    adata.obs[split_name] = "train"
    adata.obs[split_name][adata.obs.dose_val == dose] = "ood"

    test_idx = sc.pp.subsample(
        adata[adata.obs[split_name] != "ood"], 0.16, copy=True
    ).obs.index
    adata.obs[split_name].loc[test_idx] = "test"

    display(adata.obs[split_name].value_counts())

# %%
adata.uns["splits"] = split_dict

# %%
sc.write("./datasets/sciplex3_new.h5ad", adata)
