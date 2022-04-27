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
# # Analyzing the results for `sciplex_hparam`
#
# This is preliminary to the `fintuning_num_genes` experiments. We look at the results of sweeping the optimisation related hyperparameters for fine-tuning on the sciplex dataset for all other embeddings.

# %% pycharm={"name": "#%%\n"}
import math
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
    "sciplex_hparam",
    to_data_frame=True,
    fields=["config", "result", "seml", "config_hash"],
    states=["COMPLETED"],
)

# %% pycharm={"name": "#%%\n"}
# Look at number of experiments per model
results["config.model.embedding.model"].value_counts()

# %% pycharm={"name": "#%%\n"}
results.loc[:, [c for c in results.columns if "disentanglement" in c]]

# %% [markdown]
# ## Preprocessing the results dataframe

# %%
sweeped_params = [
    # "model.hparams.dim",
    # "model.hparams.dropout",
    # "model.hparams.dosers_width",
    # "model.hparams.dosers_depth",
    "model.hparams.dosers_lr",
    "model.hparams.dosers_wd",
    # "model.hparams.autoencoder_width",
    # "model.hparams.autoencoder_depth",
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
    # "model.hparams.embedding_encoder_width",
    # "model.hparams.embedding_encoder_depth",
]

# %%
# percentage of training runs that resulted in NaNs or totally failed
nan_results = results[
    results["result.loss_reconstruction"].apply(lambda x: math.isnan(sum(x)))
]
results_clean = results[
    ~results["result.loss_reconstruction"].apply(lambda x: math.isnan(sum(x)))
].copy()
print(len(nan_results) / len(results))

# Remove runs with r2 < 0.6 on the training set
results_clean = results_clean[
    results_clean["result.training"].apply(lambda x: x[0][0]) > 0.6
]

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


results_clean.head(3)

# %% [markdown]
# ## Look at early stopping

# %%
ax = sns.histplot(data=results_clean["result.epoch"].apply(max))
ax.set_title("Total epochs before final stopping (min 125)")

# %% [markdown]
# ## Look at $r^2$ reconstruction

# %%
[c for c in results_clean.columns if "pretrain" in c]

results_clean[["config.model.embedding.model", "config.model.load_pretrained"]]

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
        scale="width",
    )
    ax[i].set_ylim([0.0, 1])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
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
    # sns.violinplot(data=results_clean[results_clean['config.model.load_pretrained']==True], x="config.model.embedding.model", y=y, inner='point' ,ax=ax[i], scale='width')
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        # hue='config.model.load_pretrained',
        inner="point",
        ax=ax[i],
        scale="width",
    )
    # ax[i].set_ylim([0.82,1])
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
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
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
n_top = 5

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
        df = results_clean[performance_condition(embedding, pretrained, 0.18)]
        print(embedding, pretrained, len(df))
        best.append(
            df.sort_values(by="result.val_mean_de", ascending=False).head(n_top)
        )

best = pd.concat(best)

# %% [markdown]
# 1. Check the disentanglement computation
# 2. Plot the UMAP on the drug latens and compare from scratch vs. pre-trained
#     * This would be a major selling point
# 3. Other metics but R2? CPA Review: Wasserstein distance?
# 4. Better variance r2 for pre-trained models? Variance of the gaussian output (3rd and 4th output)

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
    # ax[i].set_ylim([0.75, 1.01])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
    ax[i].legend(title="Pretrained", loc="lower right", fontsize=18, title_fontsize=24)
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
    ax[i].legend(title="Pretrained", loc="best", fontsize=18, title_fontsize=24)
plt.tight_layout()

# %% [markdown]
# ## Take a deeper look in the `.config` of the best performing models

# %%
best[
    ["config." + col for col in sweeped_params]
    + ["result.perturbation disentanglement", "result.test_mean", "result.test_mean_de"]
]

# %%
