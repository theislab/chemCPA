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
# If not done before: combine adatas!

from pathlib import Path

import pandas as pd
import scanpy as sc

from chemCPA.paths import PROJECT_DIR

adatas = [
    sc.read(PROJECT_DIR / "datasets" / f"adata_{split}_biolord_split_30.h5ad") for split in ["train", "test", "ood"]
]
# adatas = [sc.read(Path("project_folder") / "datasets" / f"adata_{split}_biolord_split_30.h5ad") for split in ["train", "test"]]

for adata in adatas:
    df = adata.obs
    for col in df.select_dtypes(["category"]).columns:
        _type = type(df[col].cat.categories[0])
        print(f"{col}: {_type}")
        try:
            df[col] = df[col].astype(_type)
        except:
            print(col)
            df[col] = df[col].astype(str)


adata = sc.concat(adatas)


key_check = (
    ~pd.Series(adatas[0].uns["rank_genes_groups_cov_all"].keys()).isin(
        list(adatas[1].uns["rank_genes_groups_cov_all"].keys())
    )
).sum()
print(f"Key check: {key_check} should be 0.")

adata.uns["rank_genes_groups_cov_all"] = adatas[0].uns["rank_genes_groups_cov_all"]

# Add proper DMSO
adata.obs["smiles"].replace("nan", "CS(C)=O", inplace=True)

# %%
sc.write(Path("project_folder") / "datasets" / "adata_biolord_split_30.h5ad", adata)
