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

# %% [markdown]
# **Requirements:**
# * Trained models
# * GROVER:
#      * fine-tuned:      `'a2e83773f445adf813284155efbede9e'`
#      * non-pretrained: `'5cacac24918054861104eacff97fcf5c'`
# * RDKit:
#      * fine-tuned:      `'d2686f53a55468497195941fac1d7e5e'`
#      * non-pretrained: `'28c172ee2884c3204fa0df4b7223ff93'`
# * JT-VAE:
#      * fine-tuned:      `'a15a363b77060383b397a81861615864'`
#      * non-pretrained: `'cbf9e956049fce00dbcebdfc1aeb67fe'`
#
# Here everything is in setting 2 (extended gene set, 977 L1000 + 1023 HVGs)
#
# **Outputs:**
# * **Table 3**
# * Supplement Table 10
# ___
# # Imports

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import umap.plot
from utils import (
    compute_drug_embeddings,
    compute_pred,
    compute_pred_ctrl,
    load_config,
    load_dataset,
    load_model,
    load_smiles,
)

from chemCPA.data import load_dataset_splits
from chemCPA.paths import FIGURE_DIR

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
sns.set_style("whitegrid")
sns.set_context("poster")


# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Load model configs and dataset
# * Define `seml_collection` and `model_hash` to load data and model
#
#
# **Info**
# * split:            `split_ood_finetuning`
# * append_ae_layer:  `True`

# %%
seml_collection = "finetuning_num_genes"


model_hash_pretrained_rdkit = "d2686f53a55468497195941fac1d7e5e"  # Fine-tuned
model_hash_scratch_rdkit = "28c172ee2884c3204fa0df4b7223ff93"  # Non-pretrained

model_hash_pretrained_grover = "a2e83773f445adf813284155efbede9e"  # Fine-tuned
model_hash_scratch_grover = "5cacac24918054861104eacff97fcf5c"  # Non-pretrained

model_hash_pretrained_jtvae = "a15a363b77060383b397a81861615864"  # Fine-tuned
model_hash_scratch_jtvae = "cbf9e956049fce00dbcebdfc1aeb67fe"  # Non-pretrained


# %% [markdown]
# ## Load config and SMILES

# %%
config = load_config(seml_collection, model_hash_pretrained_rdkit)
dataset, key_dict = load_dataset(config)
config["dataset"]["n_vars"] = dataset.n_vars

# %%
canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(
    config, dataset, key_dict, True
)

# %% [markdown]
# Get list of drugs that are ood in `ood_drugs`

# %%
ood_drugs = (
    dataset.obs.condition[
        dataset.obs[config["dataset"]["data_params"]["split_key"]].isin(["ood"])
    ]
    .unique()
    .to_list()
)

# %% [markdown]
# ## Load dataset splits

# %%
config["dataset"]["data_params"]

# %%
data_params = config["dataset"]["data_params"]
datasets = load_dataset_splits(**data_params, return_dataset=False)

# %% [markdown]
# ____
# # Run models
# ## Baseline model

# %%
dosages = [1e1, 1e2, 1e3, 1e4]
cell_lines = ["A549", "K562", "MCF7"]
use_DEGs = True

# %%
drug_r2_baseline_degs, _ = compute_pred_ctrl(
    dataset=datasets["ood"],
    dataset_ctrl=datasets["test_control"],
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)

drug_r2_baseline_all, _ = compute_pred_ctrl(
    dataset=datasets["ood"],
    dataset_ctrl=datasets["test_control"],
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)

# %% [markdown]
# ## RDKit

# %%
ood_drugs

# %% [markdown]
# ### Pretrained

# %%
config = load_config(seml_collection, model_hash_pretrained_rdkit)
config["dataset"]["n_vars"] = dataset.n_vars
model_pretrained_rdkit, embedding_pretrained_rdkit = load_model(
    config, canon_smiles_unique_sorted
)

# %%
drug_r2_pretrained_degs_rdkit, _ = compute_pred(
    model_pretrained_rdkit,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)

drug_r2_pretrained_all_rdkit, _ = compute_pred(
    model_pretrained_rdkit,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)

# %% [markdown]
# ### Non-pretrained model

# %%
config = load_config(seml_collection, model_hash_scratch_rdkit)
config["dataset"]["n_vars"] = dataset.n_vars
model_scratch_rdkit, embedding_scratch_rdkit = load_model(
    config, canon_smiles_unique_sorted
)

# %%
drug_r2_scratch_degs_rdkit, _ = compute_pred(
    model_scratch_rdkit,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)  # non-pretrained

drug_r2_scratch_all_rdkit, _ = compute_pred(
    model_scratch_rdkit,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)  # non-pretrained

# %% [markdown]
# ## GROVER

# %% [markdown]
# ### Pretrained

# %%
config = load_config(seml_collection, model_hash_pretrained_grover)
config["dataset"]["n_vars"] = dataset.n_vars
model_pretrained_grover, embedding_pretrained_grover = load_model(
    config, canon_smiles_unique_sorted
)

# %%
drug_r2_pretrained_degs_grover, _ = compute_pred(
    model_pretrained_grover,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)

drug_r2_pretrained_all_grover, _ = compute_pred(
    model_pretrained_grover,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)

# %% [markdown]
# ### Non-pretrained model

# %%
config = load_config(seml_collection, model_hash_scratch_grover)
config["dataset"]["n_vars"] = dataset.n_vars
model_scratch_grover, embedding_scratch_grover = load_model(
    config, canon_smiles_unique_sorted
)

# %%
drug_r2_scratch_degs_grover, _ = compute_pred(
    model_scratch_grover,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)  # non-pretrained

drug_r2_scratch_all_grover, _ = compute_pred(
    model_scratch_grover,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)  # non-pretrained

# %% [markdown]
# ## JT-VAE model

# %% [markdown]
# ### Pretrained

# %%
config = load_config(seml_collection, model_hash_pretrained_jtvae)
config["dataset"]["n_vars"] = dataset.n_vars
model_pretrained_jtvae, embedding_pretrained_jtvae = load_model(
    config, canon_smiles_unique_sorted
)

# %%
drug_r2_pretrained_degs_jtvae, _ = compute_pred(
    model_pretrained_jtvae,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)

drug_r2_pretrained_all_jtvae, _ = compute_pred(
    model_pretrained_jtvae,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)

# %% [markdown]
# ### Non-pretrained model

# %%
config = load_config(seml_collection, model_hash_scratch_jtvae)
config["dataset"]["n_vars"] = dataset.n_vars
model_scratch_jtvae, embedding_scratch_jtvae = load_model(
    config, canon_smiles_unique_sorted
)

# %%
drug_r2_scratch_degs_jtvae, _ = compute_pred(
    model_scratch_jtvae,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=True,
    verbose=False,
)  # non-pretrained

drug_r2_scratch_all_jtvae, _ = compute_pred(
    model_scratch_jtvae,
    datasets["ood"],
    genes_control=datasets["test_control"].genes,
    dosages=dosages,
    cell_lines=cell_lines,
    use_DEGs=False,
    verbose=False,
)  # non-pretrained


# %% [markdown]
# # Combine results and create dataframe

# %%
def create_df(
    drug_r2_baseline,
    drug_r2_pretrained_rdkit,
    drug_r2_scratch_rdkit,
    drug_r2_pretrained_grover,
    drug_r2_scratch_grover,
    drug_r2_pretrained_jtvae,
    drug_r2_scratch_jtvae,
):
    df_baseline = pd.DataFrame.from_dict(
        drug_r2_baseline, orient="index", columns=["r2_de"]
    )
    df_baseline["type"] = "baseline"
    df_baseline["model"] = "baseline"

    df_pretrained_rdkit = pd.DataFrame.from_dict(
        drug_r2_pretrained_rdkit, orient="index", columns=["r2_de"]
    )
    df_pretrained_rdkit["type"] = "pretrained"
    df_pretrained_rdkit["model"] = "rdkit"
    df_scratch_rdkit = pd.DataFrame.from_dict(
        drug_r2_scratch_rdkit, orient="index", columns=["r2_de"]
    )
    df_scratch_rdkit["type"] = "non-pretrained"
    df_scratch_rdkit["model"] = "rdkit"

    df_pretrained_grover = pd.DataFrame.from_dict(
        drug_r2_pretrained_grover, orient="index", columns=["r2_de"]
    )
    df_pretrained_grover["type"] = "pretrained"
    df_pretrained_grover["model"] = "grover"
    df_scratch_grover = pd.DataFrame.from_dict(
        drug_r2_scratch_grover, orient="index", columns=["r2_de"]
    )
    df_scratch_grover["type"] = "non-pretrained"
    df_scratch_grover["model"] = "grover"

    df_pretrained_jtvae = pd.DataFrame.from_dict(
        drug_r2_pretrained_jtvae, orient="index", columns=["r2_de"]
    )
    df_pretrained_jtvae["type"] = "pretrained"
    df_pretrained_jtvae["model"] = "jtvae"
    df_scratch_jtvae = pd.DataFrame.from_dict(
        drug_r2_scratch_jtvae, orient="index", columns=["r2_de"]
    )
    df_scratch_jtvae["type"] = "non-pretrained"
    df_scratch_jtvae["model"] = "jtvae"

    df = pd.concat(
        [
            df_baseline,
            df_pretrained_rdkit,
            df_scratch_rdkit,
            df_pretrained_grover,
            df_scratch_grover,
            df_pretrained_jtvae,
            df_scratch_jtvae,
        ]
    )

    df["r2_de"] = df["r2_de"].apply(lambda x: max(x, 0))
    # df['delta'] = df['pretrained'] - df['scratch']
    df["cell_line"] = pd.Series(df.index.values).apply(lambda x: x.split("_")[0]).values
    df["drug"] = pd.Series(df.index.values).apply(lambda x: x.split("_")[1]).values
    df["dose"] = pd.Series(df.index.values).apply(lambda x: x.split("_")[2]).values
    df["dose"] = df["dose"].astype(float)

    #     df['combination'] = df.index.values
    #     assert (df[df.type=='pretrained'].combination == df[df.type=='non-pretrained'].combination).all()

    #     delta = (df[df.type=='pretrained'].r2_de - df[df.type=='non-pretrained'].r2_de).values
    #     df['delta'] = list(delta) + list(-delta) + [0]*len(delta)

    df = df.reset_index()
    return df


# %%
df_degs = create_df(
    drug_r2_baseline_degs,
    drug_r2_pretrained_degs_rdkit,
    drug_r2_scratch_degs_rdkit,
    drug_r2_pretrained_degs_grover,
    drug_r2_scratch_degs_grover,
    drug_r2_pretrained_degs_jtvae,
    drug_r2_scratch_degs_jtvae,
)
df_all = create_df(
    drug_r2_baseline_all,
    drug_r2_pretrained_all_rdkit,
    drug_r2_scratch_all_rdkit,
    drug_r2_pretrained_all_grover,
    drug_r2_scratch_all_grover,
    drug_r2_pretrained_all_jtvae,
    drug_r2_scratch_all_jtvae,
)

# %% [markdown]
# ## Compute mean and median across DEGs and all genes

# %%
r2_degs_mean = []
for model, _df in df_degs.groupby(["model", "type", "dose"]):
    dose = model[2]
    if dose == 1.0:
        print(f"Model: {model}, R2 mean: {_df.r2_de.mean()}")
        r2_degs_mean.append(_df.r2_de.mean())

# %%
r2_all_mean = []
for model, _df in df_all.groupby(["model", "type", "dose"]):
    dose = model[2]
    if dose == 1.0:
        print(f"Model: {model}, R2 mean: {_df.r2_de.mean()}")
        r2_all_mean.append(_df.r2_de.mean())

# %%
r2_degs_median = []
for model, _df in df_degs.groupby(["model", "type", "dose"]):
    dose = model[2]
    if dose == 1.0:
        print(f"Model: {model}, R2 median: {_df.r2_de.median()}")
        r2_degs_median.append(_df.r2_de.median())

# %%
r2_all_median = []
model = []
model_type = []
for _model, _df in df_all.groupby(["model", "type", "dose"]):
    dose = _model[2]
    if dose == 1.0:
        print(f"Model: {_model}, R2 median: {_df.r2_de.median()}")
        r2_all_median.append(_df.r2_de.median())
        model.append(_model[0])
        model_type.append(_model[1])

# %% [markdown]
# # Compute Table 3

# %%
df_dict = {
    "Model": model,
    "Type": model_type,
    "Mean $r^2$ all": r2_all_mean,
    "Mean $r^2$ DEGs": r2_degs_mean,
    "Median $r^2$ all": r2_all_median,
    "Median $r^2$ DEGs": r2_degs_median,
}

df = pd.DataFrame.from_dict(df_dict)
df = df.set_index("Model")

# %%
print(df.to_latex(float_format="%.2f"))

# %% [markdown]
# ____
# # Compute Supplement Table 10

# %% [markdown]
# Calculations

# %%
dose = 1.0
vs_model = "baseline"

models = []
gene_set = []
p_values = []
vs_models = []


for model in ["rdkit", "grover", "jtvae"]:
    for vs_model in ["baseline", "non-pretrained"]:
        _df = df_all[df_all.model.isin([vs_model, model])]
        _df = _df[_df.type.isin(["pretrained", vs_model]) & (_df.dose == dose)]
        #     display(_df)
        stat, pvalue = scipy.stats.ttest_rel(
            _df[(_df.type == "pretrained") & (_df.dose == dose)].r2_de,
            _df[(_df.type == vs_model) & (_df.dose == dose)].r2_de,
        )
        #     print(f"Model: {model}, p-value: {pvalue}")
        models.append(model)
        gene_set.append("all genes")
        p_values.append(pvalue)
        vs_models.append(vs_model)

        _df = df_degs[df_degs.model.isin(["baseline", model])]
        _df = _df[_df.type.isin(["pretrained", vs_model]) & (_df.dose == dose)]
        #     display(_df)
        stat, pvalue = scipy.stats.ttest_rel(
            _df[(_df.type == "pretrained") & (_df.dose == dose)].r2_de,
            _df[(_df.type == vs_model) & (_df.dose == dose)].r2_de,
        )
        #     print(f"Model: {model}, p-value: {pvalue}")
        models.append(model)
        gene_set.append("DEGs")
        p_values.append(pvalue)
        vs_models.append(vs_model)

# %%
df_dict = {
    "Model $G$": models,
    "Against": vs_models,
    "Gene set": gene_set,
    "p-value": p_values,
}

df = pd.DataFrame.from_dict(df_dict)
df = df.set_index("Model $G$")

# %% [markdown]
# Print table

# %%
print(df.to_latex(float_format="%.4f"))

# %% [markdown]
# ____

# %%
