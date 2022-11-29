# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %%
import pandas as pd
import scanpy as sc

from chemCPA.paths import DATA_DIR

pd.set_option("display.max_columns", 200)

# %%
list(DATA_DIR.iterdir())

# %%
adata_sciplex = sc.read(DATA_DIR / "sciplex_complete_middle_subset_lincs_genes.h5ad")

# %%
adata_sciplex.obs.columns

# %%
adata_sciplex.obs.loc[
    adata_sciplex.obs.split_ood_multi_task == "ood", "condition"
].unique()

# %%
# Subset to second largest dose

print(adata_sciplex.obs.dose.unique())
adata_sciplex = adata_sciplex[adata_sciplex.obs.dose.isin([0.0, 1e4])].copy()

# %%
# Add new splits for dose=1000 and cell_type (A549, MCF7, K562) being unseen for ood drugs

for cell_type in adata_sciplex.obs.cell_type.unique():
    print(cell_type)
    adata_sciplex.obs[f"split_baseline_{cell_type}"] = adata_sciplex.obs[
        "split_ood_multi_task"
    ]
    sub_df = adata_sciplex.obs.loc[
        (adata_sciplex.obs[f"split_baseline_{cell_type}"] == "ood")
        * (adata_sciplex.obs.cell_type != cell_type)
    ]

    train_test = sub_df.index
    test = sub_df.sample(frac=0.5).index

    adata_sciplex.obs.loc[train_test, f"split_baseline_{cell_type}"] = "train"
    adata_sciplex.obs.loc[test, f"split_baseline_{cell_type}"] = "test"

# %%
adata_sciplex.obs["split_baseline_A549"].value_counts()

# %%
pd.crosstab(adata_sciplex.obs["split_ood_multi_task"], adata_sciplex.obs["condition"])

# %%
# Quick check that everything is correct

cell_type = "K562"

# pd.crosstab(adata_sciplex.obs[f'split_baseline_{cell_type}'], adata_sciplex.obs['condition'])
pd.crosstab(
    adata_sciplex.obs[f"split_baseline_{cell_type}"], adata_sciplex.obs["cell_type"]
)

# %%
# write adata

adata_sciplex.write(DATA_DIR / "adata_baseline_high_dose.h5ad", compression="gzip")

# %%
