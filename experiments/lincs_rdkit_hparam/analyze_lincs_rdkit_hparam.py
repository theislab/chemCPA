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

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Analyzing the results for `lincs_rdkit_hparam`
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
    "lincs_rdkit_hparam",
    to_data_frame=True,
    fields=["config", "result", "seml", "config_hash"],
    states=["COMPLETED"],
    filter_dict={"batch_id": 1},  # just the first batch (rdkit only)
)

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
ax = sns.histplot(data=results_clean["result.epoch"].apply(max))
ax.set_title("Total epochs before final stopping (min 125)")

# %% [markdown]
# ## Look at $r^2$ reconstruction

# %% [markdown]
# ### DE genes

# %%
rows = 1
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 7 * rows), sharex=True)

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
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    plt.tight_layout()

# %% [markdown]
# ### All genes

# %%
rows = 1
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 7 * rows), sharex=True)

for i, y in enumerate(("result.training_mean", "result.val_mean", "result.test_mean")):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        inner="point",
        ax=ax[i],
    )
    ax[i].set_ylim([0.82, 1])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    plt.tight_layout()

# %% [markdown]
# ## Look at disentanglement scores

# %%
rows = 1
cols = 1
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 7 * rows), sharex=True)

for y in ["result.perturbation disentanglement"]:
    sns.violinplot(
        data=results_clean, x="config.model.embedding.model", y=y, inner="point", ax=ax
    )
    # ax[i].set_ylim([0,1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.axhline(0.18, color="orange")
    ax.set_xlabel("")
    ax.set_ylabel(y.split(".")[-1])
    plt.tight_layout()

# %% [markdown]
# ## Subselect to disentangled models

# %%


performance_condition = lambda emb, max_entangle: (
    results_clean["config.model.embedding.model"] == emb
) & (results_clean["result.perturbation disentanglement"] < max_entangle)

best = []
for embedding in list(results_clean["config.model.embedding.model"].unique()):
    df = results_clean[performance_condition(embedding, 0.18)]
    print(embedding, len(df))
    best.append(df.sort_values(by="result.val_mean_de", ascending=False).head(3))

best = pd.concat(best)

# %%
# All genes, DE genes, disentanglement
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

for i, y in enumerate(
    ["result.test_mean", "result.test_mean_de", "result.perturbation disentanglement"]
):
    sns.violinplot(
        data=best, x="config.model.embedding.model", y=y, inner="points", ax=ax[i]
    )
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])

plt.tight_layout()


# %% [markdown]
# ## Take a deeper look in the `.config` of the best performing models

# %%
best[
    ["config." + col for col in sweeped_params]
    + ["result.perturbation disentanglement", "result.test_mean", "result.test_mean_de"]
]

# %%
# -> Middle sized autoencoder width
results_clean["config.model.hparams.autoencoder_width"].value_counts()

# %%
# -> rather high regularisation
results_clean["config.model.hparams.reg_adversary"].sort_values(ascending=False)[:10]

# %%
