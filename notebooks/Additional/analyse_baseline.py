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
import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from utils import compute_pred_ctrl, load_dataset, load_smiles

from chemCPA.data import load_dataset_splits
from chemCPA.paths import DATA_DIR, FIGURE_DIR, PROJECT_DIR, ROOT

# %%
pd.set_option("display.max_columns", 200)

# %%
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %%
seml_collection = "baseline_comparison"

# %%
seml_collection = "baseline_comparison"

base_hash_dict = dict(
    baseline_A549="044c4dba0c8719985c3622834f2cbd58",
    baseline_K562="5ea85d5dd7abd5962d1d3eeff1b8c1ff",
    baseline_MCF7="4d98e7d857f497d870e19c6d12175aaa",
)


# %%
def load_config(seml_collection, model_hash):
    file_path = PROJECT_DIR / f"{seml_collection}.json"  # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)

    for _config in tqdm(file_data):
        if _config["config_hash"] == model_hash:
            # print(config)
            config = _config["config"]
            config["config_hash"] = _config["config_hash"]
    return config


# %%
config = load_config(seml_collection, base_hash_dict["baseline_A549"])

config["dataset"]["data_params"]["dataset_path"] = (
    DATA_DIR / "sciplex_complete_middle_subset_lincs_genes.h5ad"
)
config["model"]["embedding"]["directory"] = (
    ROOT / config["model"]["embedding"]["directory"]
)

dataset, key_dict = load_dataset(config)

config["dataset"]["n_vars"] = dataset.n_vars

canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict, True
)
# model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)

# %%
data_params = config["dataset"]["data_params"]
data_params["split_key"] = "split_random"
datasets, dataset_all = load_dataset_splits(**data_params, return_dataset=True)

# %%
data_params

# %%
dosages = [1e1, 1e2, 1e3, 1e4]
cell_lines = ["A549", "K562", "MCF7"]
use_DEGs = True

# %%
len(pd.Series(dataset_all.pert_categories).unique())


# %%
def get_baseline_predictions(
    hash,
    seml_collection="baseline_comparison",
    smiles=None,
    dosages=[1e1, 1e2, 1e3, 1e4],
    cell_lines=["A549", "K562", "MCF7"],
    use_DEGs=False,
    verbose=False,
    name_tag=None,
):
    if smiles is None:
        smiles = canon_smiles_unique_sorted

    config = load_config(seml_collection, hash)
    config["dataset"]["n_vars"] = dataset.n_vars
    config["dataset"]["data_params"]["dataset_path"] = (
        ROOT / config["dataset"]["data_params"]["dataset_path"]
    )
    config["model"]["embedding"]["directory"] = (
        ROOT / config["model"]["embedding"]["directory"]
    )
    data_params = config["dataset"]["data_params"]
    datasets = load_dataset_splits(**data_params, return_dataset=False)

    predictions, _ = compute_pred_ctrl(
        dataset_all,
        dataset_ctrl=datasets["test_control"],
        # dataset_ctrl=datasets["training_control"],
        dosages=dosages,
        cell_lines=cell_lines,
        use_DEGs=use_DEGs,
        verbose=verbose,
    )

    predictions = pd.DataFrame.from_dict(predictions, orient="index", columns=["R2"])
    if name_tag:
        predictions["model"] = name_tag
    predictions["genes"] = "degs" if use_DEGs else "all"
    return predictions


# %%
# drug_r2_baseline_degs, _ = compute_pred_ctrl(
#     dataset=datasets["ood"],
#     dataset_ctrl=datasets["test_control"],
#     dosages=dosages,
#     cell_lines=cell_lines,
#     use_DEGs=True,
#     verbose=False,
# )

# drug_r2_baseline_all, _ = compute_pred_ctrl(
#     dataset=datasets["ood"],
#     dataset_ctrl=datasets["test_control"],
#     dosages=dosages,
#     cell_lines=cell_lines,
#     use_DEGs=False,
#     verbose=False,
# )
predictions = []

predictions.extend(
    [
        get_baseline_predictions(_hash, name_tag=name_tag, use_DEGs=True)
        for name_tag, _hash in base_hash_dict.items()
    ]
)

predictions.extend(
    [
        get_baseline_predictions(_hash, name_tag=name_tag, use_DEGs=False)
        for name_tag, _hash in base_hash_dict.items()
    ]
)

# %%
predictions


# %%
def rename_model(str):
    str_list = str.split("_")
    if len(str_list) == 2:
        return str_list[0]
    else:
        assert len(str_list) == 3
        return "_".join([str_list[0], str_list[2]])


# %%
predictions = pd.concat(predictions)
predictions.reset_index(inplace=True)
predictions["cell_type"] = predictions["index"].apply(lambda s: s.split("_")[0])
predictions["condition"] = predictions["index"].apply(lambda s: s.split("_")[1])
predictions["dose"] = predictions["index"].apply(lambda s: s.split("_")[2])
predictions["model_ct"] = predictions["model"]
predictions["model"] = predictions["model"].apply(rename_model)

# %%
predictions["dose"].unique()

# %%
cond = (predictions["genes"] == "degs") & (predictions["dose"] == "1.0")
predictions[cond].groupby(["condition"]).mean().sort_values("R2")

# %%
mean_df = predictions[cond].groupby(["condition"]).mean().sort_values("R2")
mol_set = set(mean_df.index[:50])

# %%
std_df = predictions[cond].groupby(["condition"]).std().sort_values("R2")
std_df.rename(columns={"R2": "R2_std"}, inplace=True)

# %%
std_df

# %%
mean_df

# %%
df = pd.concat([mean_df, std_df], axis=1)

# %%
df_subset = df[(df["R2"] < 0.784) & (df["R2_std"] < 0.3)]
len(df_subset)

# %%
ood_drugs = df_subset.sample(frac=1).index.to_list()

# %%
import scanpy as sc

adata_sciplex = sc.read(DATA_DIR / "sciplex_complete_middle_subset_lincs_genes.h5ad")

# %%
pd.crosstab(
    adata_sciplex.obs["split_ood_multi_task"],
    adata_sciplex.obs["condition"].isin(["control"]),
)

# %%

print(adata_sciplex.obs.dose.unique())
adata_sciplex = adata_sciplex[adata_sciplex.obs.dose.isin([0.0, 1e3])].copy()

# %%
i = 0
n_drugs = 5
drug_sets = []
for j in range(n_drugs, len(ood_drugs), n_drugs):
    drug_set = ood_drugs[i : j + n_drugs]
    if (j + n_drugs) > len(ood_drugs):
        drug_set += ood_drugs[: 2 * n_drugs - len(drug_set)]
    i = j
    drug_sets.append(drug_set)

# %%
[print(s) for s in drug_sets]

# %%
for i, drug_set in enumerate(drug_sets):
    for cell_type in adata_sciplex.obs.cell_type.unique():
        split = f"split_fold{i}_{cell_type}"
        print(split)
        adata_sciplex.obs[split] = adata_sciplex.obs["split_ood_multi_task"]
        adata_sciplex.obs.loc[adata_sciplex.obs[split] == "ood", split] = "train"
        adata_sciplex.obs.loc[
            adata_sciplex.obs["condition"].isin(drug_set), split
        ] = "ood"

        sub_df = adata_sciplex.obs.loc[
            adata_sciplex.obs[split].isin(["ood"])
            * (adata_sciplex.obs.cell_type != cell_type)
        ]
        train_test = sub_df.index
        test = sub_df.sample(frac=0.5).index

        sub_df2 = adata_sciplex.obs.loc[adata_sciplex.obs[split].isin(["train"])]
        train_test2 = sub_df2.index
        test2 = sub_df.sample(frac=0.05).index

        adata_sciplex.obs.loc[train_test, split] = "train"
        adata_sciplex.obs.loc[test, split] = "test"
        adata_sciplex.obs.loc[train_test2, split] = "train"
        adata_sciplex.obs.loc[test2, split] = "test"


# %%
# split_fold0_A549
# split_fold0_MCF7
# split_fold0_K562
# split_fold1_A549
# split_fold1_MCF7
# split_fold1_K562
# split_fold2_A549
# split_fold2_MCF7
# split_fold2_K562
# split_fold3_A549
# split_fold3_MCF7
# split_fold3_K562
# split_fold4_A549
# split_fold4_MCF7
# split_fold4_K562
# split_fold5_A549
# split_fold5_MCF7
# split_fold5_K562
# split_fold6_A549
# split_fold6_MCF7
# split_fold6_K562
# split_fold7_A549
# split_fold7_MCF7
# split_fold7_K562

# %%
pd.crosstab(adata_sciplex.obs["split_fold7_A549"], adata_sciplex.obs["condition"])

# %%
pd.crosstab(adata_sciplex.obs["split_fold7_A549"], adata_sciplex.obs["cell_type"])

# %%
pd.crosstab(
    adata_sciplex.obs["split_fold7_A549"],
    adata_sciplex.obs["condition"].isin(["control"]),
)

# %%
adata_sciplex.write(DATA_DIR / "adata_fold.h5ad", compression="gzip")

# %%
