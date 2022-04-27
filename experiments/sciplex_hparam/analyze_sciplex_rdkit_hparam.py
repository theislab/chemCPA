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
# # Analyzing the results for `sciplex_hparam` with `grover` and `rdkit` sweeps
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

from chemCPA.paths import FIGURE_DIR

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
    filter_dict={
        # 'batch_id': 3,
        "config.dataset.data_params.split_key": "split_ho_pathway"
    },
)

# %% pycharm={"name": "#%%\n"}
# Look at number of experiments per model
results["config.model.embedding.model"].value_counts()

# %% pycharm={"name": "#%%\n"}
results.loc[:, [c for c in results.columns if "pretrain" in c]]

# %%
pd.crosstab(
    results["config.model.embedding.model"],
    results["result.perturbation disentanglement"].isnull(),
)

# %%
[c for c in results.columns if "split" in c]

# %%
pd.crosstab(
    results["config.dataset.data_params.split_key"],
    results["result.perturbation disentanglement"].isnull(),
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

results_clean = results[results._id.isin(clean_id)].copy()
print(f"Percentage of invalid (nan) runs: {1 - len(clean_id) / len(results)}")

# Remove runs with r2 < 0.6 on the training set
# results_clean = results_clean[results_clean['result.training'].apply(lambda x: x[0][0])>0.6]

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
results_clean["result.covariate disentanglement"] = results_clean[
    "result.covariate disentanglement"
].apply(lambda x: x[0][0])
results_clean["result.final_reconstruction"] = results_clean[
    "result.loss_reconstruction"
].apply(lambda x: x[-1])

results_clean.head(3)

# %%
# results_clean["config.model.load_pretrained"]

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
    ax[i].set_ylim([0.3, 1.01])
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
    ax[i].set_ylim([0.3, 1.01])
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
rows = 2
cols = 1
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), sharex=True)

max_entangle = [0.07, 0.65]
for i, y in enumerate(
    ["result.perturbation disentanglement", "result.covariate disentanglement"]
):
    sns.violinplot(
        data=results_clean,
        x="config.model.embedding.model",
        y=y,
        inner="point",
        ax=ax[i],
        hue="config.model.load_pretrained",
    )
    # ax[i].set_ylim([0,1])
    x_ticks = ax[i].get_xticklabels()
    [x_tick.set_text(x_tick.get_text().split("_")[0]) for x_tick in x_ticks]
    ax[i].set_xticklabels(x_ticks, rotation=25, ha="center")
    ax[i].axhline(max_entangle[i], ls=":", color="black")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(y.split(".")[-1])
ax[1].get_legend().remove()
ax[0].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %% [markdown]
# ## Subselect to disentangled models

# %%
n_top = 2


def performance_condition(emb, pretrained, max_entangle, max_entangle_cov):
    cond = results_clean["config.model.embedding.model"] == emb
    cond = cond & (results_clean["result.perturbation disentanglement"] < max_entangle)
    cond = cond & (results_clean["config.model.load_pretrained"] == pretrained)
    cond = cond & (results_clean["result.covariate disentanglement"] < max_entangle_cov)
    return cond


best = []
for embedding in list(results_clean["config.model.embedding.model"].unique()):
    for pretrained in [True, False]:
        df = results_clean[
            performance_condition(
                embedding, pretrained, max_entangle[0], max_entangle[1]
            )
        ]
        print(embedding, pretrained, len(df))
        # if len(df) == 0:
        #     df = results_clean[performance_condition(embedding, pretrained, max_entangle[0], max_entangle[1]+0.05)]
        #     if len(df) == 0:
        #         df = results_clean[performance_condition(embedding, pretrained, max_entangle[0], max_entangle[1]+0.2)]
        #         if len(df) == 0:
        #             df = results_clean[performance_condition(embedding, pretrained, max_entangle[0], max_entangle[1]+0.3)]
        if not pretrained and len(df) == 0:
            best = best[:-1]  # delete previous run
        best.append(
            df.sort_values(by="result.val_mean_de", ascending=False).head(n_top)
        )

best = pd.concat(best)

# %%
# All genes, DE genes, disentanglement
rows, cols = 2, 2
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

for i, y in enumerate(
    [
        "result.test_mean",
        "result.test_mean_de",
        "result.perturbation disentanglement",
        "result.covariate disentanglement",
    ]
):
    sns.violinplot(
        data=best,
        x="config.model.embedding.model",
        y=y,
        inner="points",
        ax=ax[i // cols, i % cols],
        scale="area",
        hue="config.model.load_pretrained",
    )
    x_ticks = ax[i // cols, i % cols].get_xticklabels()
    [x_tick.set_text(x_tick.get_text().split("_")[0]) for x_tick in x_ticks]
    ax[i // cols, i % cols].set_xticklabels(x_ticks, rotation=25, ha="center")
    ax[i // cols, i % cols].set_xlabel("")
    ax[i // cols, i % cols].set_ylabel(y.split(".")[-1])
ax[0, 0].set_ylabel("$\mathbb{E}\,[R^2]$ on all genes")
# ax[0,0].set_ylim([0.89, 0.96])
ax[0, 1].set_ylabel("$\mathbb{E}\,[R^2]$ on DE genes")
ax[0, 1].set_ylim([0.59, 0.91])

ax[1, 0].set_ylabel("Drug entanglement")
ax[1, 0].axhline(max_entangle[0], ls=":", color="black")
ax[1, 0].text(
    3.0, max_entangle[0] + 0.003, "max entangled", fontsize=15, va="center", ha="center"
)
ax[1, 0].set_ylim([-0.01, 0.089])
ax[1, 1].set_ylabel("Covariate entanglement")
ax[1, 1].text(
    3.0, max_entangle[1] + 0.015, "max entangled", fontsize=15, va="center", ha="center"
)
ax[1, 1].axhline(max_entangle[1], ls=":", color="black")

ax[0, 0].get_legend().remove()
ax[1, 0].get_legend().remove()
ax[1, 1].get_legend().remove()
ax[0, 1].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.6),
)
plt.tight_layout()

split_keys = results_clean["config.dataset.data_params.split_key"].unique()
assert len(split_keys) == 1
split_key = split_keys[0]

plt.savefig(
    FIGURE_DIR / f"sciplex_{split_key}_lincs_genes.eps",
    format="eps",
    bbox_inches="tight",
)


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
best[
    ["config." + col for col in sweeped_params]
    + ["result.perturbation disentanglement", "result.test_mean", "result.test_mean_de"]
]

# %%

# %%
