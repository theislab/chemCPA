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

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Analyzing the results for `finetuning_num_genes`
#
# This is part 1, the results of sweeping all hyperparameter for rdkit

# %% pycharm={"name": "#%%\n"}
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import seml
from matplotlib import pyplot as plt

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
plt.rcParams["savefig.facecolor"] = "white"
sns.set_context("poster")
pd.set_option("display.max_columns", 100)

# %% pycharm={"name": "#%%\n"}
results = seml.get_results(
    "finetuning_num_genes",
    to_data_frame=True,
    fields=["config", "result", "seml", "config_hash"],
    states=["COMPLETED"],
    filter_dict={
        "batch_id": 3,
        "config.dataset.data_params.split_key": "split_ood_finetuning",  # split_ood_finetuning, split_random, split_ho_pathway, split_ho_epigenetic, split_ho_epigenetic_all
        "config.model.append_ae_layer": True,
    },
)

# %% pycharm={"name": "#%%\n"}
# Look at number of experiments per model
results["config.model.embedding.model"].value_counts()

# %% pycharm={"name": "#%%\n"}
pd.crosstab(
    results["config.model.embedding.model"],
    results["result.perturbation disentanglement"].isnull(),
)

# %%
[c for c in results.columns if "ae" in c]

# %%
pd.crosstab(
    results["config.dataset.data_params.split_key"],
    results["config.model.load_pretrained"],
)

# %%
pd.crosstab(
    results["config.dataset.data_params.split_key"],
    results["result.loss_reconstruction"].isnull(),
)

# %%
# columns
results.isnull().any()[results.isnull().any()]

# %%
# rows without nans
clean_id = results.loc[~results["result.training"].isnull(), "_id"]
# clean_id

# %% [markdown]
# ## Preprocessing the results dataframe

# %%
# percentage of training runs that resulted in NaNs or totally failed

results_clean = results[results._id.isin(clean_id)].copy()
print(f"Percentage of invalid (nan) runs: {1 - len(clean_id) / len(results)}")

# Remove runs with r2 < 0.6 on the training set
# results_clean = results_clean[results_clean['result.training'].apply(lambda x: x[0][0])>0.6]

# %%
results_clean["config.model.embedding.model"].value_counts()

# %%
results_clean[
    [
        "config.model.load_pretrained",
        "result.test_sc",
        "config.model.append_ae_layer",
        "result.ood",
    ]
]

# %%
# calculate some stats
get_mean = lambda x: np.array(x)[-1, 0]
get_mean_de = lambda x: np.array(x)[-1, 1]

results_clean["result.training_mean"] = results_clean["result.training"].apply(get_mean)
results_clean["result.training_mean_de"] = results_clean["result.training"].apply(
    get_mean_de
)
results_clean["result.val_mean"] = results_clean["result.test"].apply(get_mean)
results_clean["result.val_mean_de"] = results_clean["result.test"].apply(get_mean_de)
results_clean["result.test_mean"] = results_clean["result.ood"].apply(get_mean)
results_clean["result.test_mean_de"] = results_clean["result.ood"].apply(get_mean_de)
results_clean["result.perturbation disentanglement"] = results_clean[
    "result.perturbation disentanglement"
].apply(lambda x: x[0])
results_clean["result.final_reconstruction"] = results_clean[
    "result.loss_reconstruction"
].apply(lambda x: x[-1])

results_clean.head(3)

# %%
results_clean["result.training_sc_mean"] = results_clean["result.training_sc"].apply(
    get_mean
)
results_clean["result.training_sc_mean_de"] = results_clean["result.training_sc"].apply(
    get_mean_de
)
results_clean["result.val_sc_mean"] = results_clean["result.test_sc"].apply(get_mean)
results_clean["result.val_sc_mean_de"] = results_clean["result.test_sc"].apply(
    get_mean_de
)
results_clean["result.test_sc_mean"] = results_clean["result.ood_sc"].apply(get_mean)
results_clean["result.test_sc_mean_de"] = results_clean["result.ood_sc"].apply(
    get_mean_de
)

results_clean = results_clean[results_clean["result.val_sc_mean"] > 0.01]
results_clean = results_clean[results_clean["result.val_mean_de"] > 0.4]
# results_clean = results_clean[results_clean["config.model.append_ae_layer"] == True]

# %%
# results_clean["config.model.load_pretrained"]

# %%
# results_clean.plot.box(by="config.model.load_pretrained", y="result.final_reconstruction")
sns.violinplot(
    data=results_clean,
    x="config.model.load_pretrained",
    y="result.final_reconstruction",
    inner="point",
    scale="width",
)

# %% [markdown]
# ## Look at early stopping

# %%
fig, ax = plt.subplots(2, 1)
sns.histplot(
    data=results_clean[results_clean["config.model.load_pretrained"] == True][
        "result.epoch"
    ].apply(max),
    ax=ax[0],
)
ax[0].set_title("Total epochs before final stopping (min 125), pretrained")

ax[1] = sns.histplot(
    data=results_clean[results_clean["config.model.load_pretrained"] == False][
        "result.epoch"
    ].apply(max),
    ax=ax[1],
)
ax[1].set_title("Total epochs before final stopping (min 125), non pretrained")

plt.tight_layout()

# %% [markdown]
# ## Look at $r^2$ reconstruction

# %%
[c for c in results_clean.columns if "pretrain" in c]

results_clean[
    [
        "config.model.embedding.model",
        "config.model.load_pretrained",
        "config.model.append_ae_layer",
    ]
]

# %% [markdown]
# ### DE genes

# %%
# DE genes
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    ("result.training_mean_de", "result.val_mean_de", "result.test_mean_de")
):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        hue="config.model.load_pretrained",
        inner="points",
        ax=ax[i],
        scale="width",
    )
    # ax[i].set_ylim([0.3,1.01])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    ax[i].legend(title="Pretrained", loc="lower right", fontsize=18, title_fontsize=24)

ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %%
# DE genes
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    ("result.training_sc_mean_de", "result.val_sc_mean_de", "result.test_sc_mean_de")
):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        hue="config.model.load_pretrained",
        inner="points",
        ax=ax[i],
        scale="width",
    )
    ax[i].set_ylim([0.0, 0.5])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    ax[i].legend(title="Pretrained", loc="lower right", fontsize=18, title_fontsize=24)

ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %% [markdown]
# ### All genes

# %%
# DE genes
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(("result.training_mean", "result.val_mean", "result.test_mean")):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        hue="config.model.load_pretrained",
        inner="points",
        ax=ax[i],
        scale="width",
    )
    # ax[i].set_ylim([0.3,1.01])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    ax[i].legend(title="Pretrained", loc="lower right", fontsize=18, title_fontsize=24)

ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %% [markdown]
# ## Look at disentanglement scores

# %%
rows = 1
cols = 1
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows), sharex=True)

for y in ["result.perturbation disentanglement"]:
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        hue="config.model.load_pretrained",
        inner="point",
        ax=ax,
        scale="width",
    )
    # ax[i].set_ylim([0,1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right")
    ax.axhline(0.18, color="orange")
    ax.set_xlabel("")
    ax.set_ylabel(y.split(".")[-1])
plt.tight_layout()

# %% [markdown]
# ## Subselect to disentangled models

# %%
n_top = 2
performance_condition = (
    lambda emb, pretrained, max_entangle: (
        results_clean["config.model.embedding.model"] == emb
    )
    & (results_clean["result.perturbation disentanglement"] < max_entangle)
    & (results_clean["config.model.load_pretrained"] == pretrained)
)

best = []
for embedding in list(results_clean["config.model.embedding.model"].unique()):
    for pretrained in [True, False]:
        df = results_clean[performance_condition(embedding, pretrained, 0.8)]
        print(embedding, pretrained, len(df))
        best.append(
            df.sort_values(by="result.val_mean_de", ascending=False).head(n_top)
        )

best = pd.concat(best)

# %%
# All genes, DE genes, disentanglement
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    ["result.test_mean", "result.test_mean_de", "result.perturbation disentanglement"]
):
    sns.violinplot(
        data=best,
        x="config.model.embedding.model",
        y=y,
        hue="config.model.load_pretrained",
        inner="points",
        ax=ax[i],
        scale="width",
    )
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    ax[i].legend(title="Pretrained", loc="lower right", fontsize=18, title_fontsize=24)
ax[0].get_legend().remove()
ax[0].set_ylim([0.4, 1.01])
ax[1].get_legend().remove()
ax[1].set_ylim([0.4, 1.01])
ax[2].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()


# %%
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    [
        "result.training_mean",
        "result.training_mean_de",
        "result.perturbation disentanglement",
    ]
):
    sns.violinplot(
        data=best,
        x="config.model.embedding.model",
        y=y,
        hue="config.model.load_pretrained",
        inner="points",
        ax=ax[i],
        scale="width",
    )
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
ax[0].get_legend().remove()
ax[0].set_ylim([0.4, 1.01])
ax[1].get_legend().remove()
ax[1].set_ylim([0.4, 1.01])
ax[2].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %% [markdown]
# ## Take a deeper look in the `.config` of the best performing models

# %%
[c for c in best.columns if "hash" in c]

# %%
best[["config.model.load_pretrained", "config_hash", "result.test_mean_de"]]

# %%
