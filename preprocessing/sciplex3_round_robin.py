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

import matplotlib.pyplot as plt

# %%
import pandas as pd
import scanpy as sc
import seaborn as sns

sc.set_figure_params(dpi=100, frameon=False)
sc.logging.print_header()

# %%
adata = sc.read("../datasets/sciplex3_old_reproduced.h5ad")

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
        adata.obs[split_name][
            (adata.obs.ct_dose == f"{ct}_{dose}") & (adata.obs.control == 0)
        ] = "ood"

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
        adata.obs[split_name][
            (adata.obs.ct_dose == f"{cts[0]}_{dose}") & (adata.obs.control == 0)
        ] = "ood"
        adata.obs[split_name][
            (adata.obs.ct_dose == f"{cts[1]}_{dose}") & (adata.obs.control == 0)
        ] = "ood"

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
    adata.obs[split_name][
        (adata.obs.dose_val == dose) & (adata.obs.control == 0)
    ] = "ood"

    test_idx = sc.pp.subsample(
        adata[adata.obs[split_name] != "ood"], 0.16, copy=True
    ).obs.index
    adata.obs[split_name].loc[test_idx] = "test"

    display(adata.obs[split_name].value_counts())

# %%
split_dict

# %%
adata.uns["splits"] = split_dict

# %%
sc.write("../datasets/sciplex3_old_reproduced.h5ad", adata)

# %%
import numpy as np

for split in adata.obs.keys():
    if "split" in split:
        print(split, np.sum(adata[adata.obs[split].values == "ood"].obs["control"]))

# %%
for k in adata.uns["splits"].keys():
    if "1.0" in adata.uns["splits"][k]:
        print(k, adata.uns["splits"][k])

# %%
