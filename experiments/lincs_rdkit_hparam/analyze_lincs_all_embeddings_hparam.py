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
# # Analyzing the results for `lincs_rdkit_hparam`
#
# This is part 2, the results of sweeping the drug-embedding related hyperparameters for all other embeddings

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
    "lincs_rdkit_hparam",
    to_data_frame=True,
    fields=["config", "result", "seml", "config_hash"],
    states=["COMPLETED"],
    # filter_dict={"batch_id": 6}
)

# %% pycharm={"name": "#%%\n"}
# filter out the non-relevant rdkit runs
results = results[(results["config.model.hparams.dim"] == 32)]
results["config.model.embedding.model"].value_counts()

# %% pycharm={"name": "#%%\n"}
results.loc[:, [c for c in results.columns if "disentanglement" in c]]

# %% pycharm={"name": "#%%\n"}
good_disentanglement = (
    results["result.perturbation disentanglement"].apply(lambda x: x[0]) < 0.2
)

# %%
results.loc[good_disentanglement, [c for c in results.columns if "result" in c]]

# %% [markdown]
# ## Preprocessing the results dataframe

# %%
sweeped_params = [
    "model.hparams.dim",
    "model.hparams.dropout",
    "model.hparams.dosers_width",
    "model.hparams.dosers_depth",
    "model.hparams.dosers_lr",
    "model.hparams.dosers_wd",
    "model.hparams.autoencoder_width",
    "model.hparams.autoencoder_depth",
    "model.hparams.autoencoder_lr",
    "model.hparams.autoencoder_wd",
    "model.hparams.adversary_width",
    "model.hparams.adversary_depth",
    "model.hparams.adversary_lr",
    "model.hparams.adversary_wd",
    "model.hparams.adversary_steps",
    "model.hparams.reg_adversary",
    "model.hparams.penalty_adversary",
    "model.hparams.batch_size",
    "model.hparams.step_size_lr",
    "model.hparams.embedding_encoder_width",
    "model.hparams.embedding_encoder_depth",
]

# %%
# percentage of training runs that resulted in NaNs
import math

nan_results = results[
    results["result.loss_reconstruction"].apply(lambda x: math.isnan(sum(x)))
]
results_clean = results[
    ~results["result.loss_reconstruction"].apply(lambda x: math.isnan(sum(x)))
].copy()
print(len(nan_results) / len(results))

# %%
results_clean["config.model.embedding.model"].value_counts()

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


results_clean

# %% [markdown]
# ## Look at early stopping

# %%
ax = sns.histplot(data=results_clean["result.epoch"].apply(max), bins=15)
ax.set_title("Total epochs before final stopping (min 125)")

# %% [markdown]
# ## Look at $r^2$ reconstruction

# %% [markdown]
# ### DE genes

# %%
rows = 1
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), sharex=True)

for i, y in enumerate(
    ("result.training_mean_de", "result.val_mean_de", "result.test_mean_de")
):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        inner="point",
        ax=ax[i],
    )
    ax[i].set_ylim([0.39, 1])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    plt.tight_layout()

# %% [markdown]
# ### All genes

# %%
rows = 1
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), sharex=True)

for i, y in enumerate(("result.training_mean", "result.val_mean", "result.test_mean")):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        inner="point",
        ax=ax[i],
    )
    ax[i].set_ylim([0.82, 1])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    plt.tight_layout()

# %% [markdown]
# ## Look at disentanglement scores

# %%
rows = 1
cols = 1
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), sharex=True)

for y in ["result.perturbation disentanglement"]:
    sns.violinplot(
        data=results_clean, x="config.model.embedding.model", y=y, inner="point", ax=ax
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
n_top = 5

performance_condition = lambda emb, max_entangle: (
    results_clean["config.model.embedding.model"] == emb
) & (results_clean["result.perturbation disentanglement"] < max_entangle)

best = []
top_one = []
best_disentangled = []
for embedding in list(results_clean["config.model.embedding.model"].unique()):
    df = results_clean[performance_condition(embedding, 0.15)]
    print(embedding, len(df))
    best.append(df.sort_values(by="result.val_mean_de", ascending=False).head(n_top))
    top_one.append(df.sort_values(by="result.val_mean_de", ascending=False).head(1))
    best_disentangled.append(
        df.sort_values(by="result.perturbation disentanglement", ascending=True).head(
            n_top
        )
    )

best = pd.concat(best)
top_one = pd.concat(top_one)
best_disentangled = pd.concat(best_disentangled)

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
        inner="points",
        ax=ax[i],
        scale="width",
    )
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])

plt.tight_layout()


# %% [markdown]
# Top 3 best disentangled models per embedding type

# %%
# All genes, DE genes, disentanglement
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    ["result.test_mean", "result.test_mean_de", "result.perturbation disentanglement"]
):
    sns.violinplot(
        data=best_disentangled,
        x="config.model.embedding.model",
        y=y,
        inner="points",
        ax=ax[i],
        scale="width",
    )
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])

plt.tight_layout()

# %% [markdown]
# Top one performing models

# %%
# All genes, DE genes, disentanglement
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    ["result.test_mean", "result.test_mean_de", "result.perturbation disentanglement"]
):
    sns.violinplot(
        data=top_one,
        x="config.model.embedding.model",
        y=y,
        inner="points",
        ax=ax[i],
        scale="width",
    )
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])

plt.tight_layout()

# %% [markdown]
# ## Take a deeper look in the `.config` of the best performing models

# %%
top_one["config.model.embedding.model"]

# %%
top_one[
    ["config." + col for col in sweeped_params]
    + ["result.perturbation disentanglement", "result.test_mean", "result.test_mean_de"]
]

# %%
sweeped_cols = np.array(["config." + col for col in sweeped_params])
top_one[
    ["config.model.embedding.model"]
    + list(sweeped_cols[best[sweeped_cols].std() > 1e-5])
]

# %%
# -> Middle sized doser width
results_clean["config.model.hparams.dosers_width"].value_counts()

# %%
# Check dim
results_clean["config.model.hparams.dim"].value_counts()

# %%
# Only GCN was able to improve in {batch_id: 6}
top_one[[c for c in results_clean.columns if ("hash" in c) | ("embedding.model" in c)]]

# %%
best[[c for c in results_clean.columns if ("hash" in c) | ("embedding.model" in c)]]

# %%
best[best["config.model.embedding.model"] == "GCN"]

# %% [markdown]
# ## Look at correlation between disentanglement and reconstruction

# %%
fig, ax = plt.subplots(figsize=(10, 8))

# Regression without weave
sns.regplot(
    data=results_clean[results_clean["config.model.embedding.model"] != "weave"],
    x="result.perturbation disentanglement",
    y="result.test_mean_de",
    ax=ax,
    scatter=False,
)

sns.scatterplot(
    data=results_clean,
    x="result.perturbation disentanglement",
    y="result.test_mean_de",
    ax=ax,
    style="config.model.embedding.model",
    legend=None,
    color="grey",
    alpha=0.6,
)
sns.scatterplot(
    data=best,
    x="result.perturbation disentanglement",
    y="result.test_mean_de",
    hue="config.model.embedding.model",
    ax=ax,
    style="config.model.embedding.model",
)
ax.set_xlim([0, 0.44])
ax.set_ylim([0.44, 0.93])
ax.legend(loc="best")

# %% [markdown]
# ## Look at epochs vs. performance

# %%
[c for c in results_clean.columns if "epochs" in c]

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    data=results_clean,
    x="result.total_epochs",
    y="result.test_mean_de",
    ax=ax,
    style="config.model.embedding.model",
    color="grey",
    alpha=0.7,
    legend=None,
)
sns.scatterplot(
    data=best,
    x="result.total_epochs",
    y="result.test_mean_de",
    ax=ax,
    style="config.model.embedding.model",
    hue="config.model.embedding.model",
)

# %%
