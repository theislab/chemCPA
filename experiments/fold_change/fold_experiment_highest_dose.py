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

# %%
# %load_ext lab_black

# %%
results = seml.get_results(
    "fold_comparison",
    to_data_frame=True,
    fields=["config", "result", "seml", "config_hash"],
    states=["COMPLETED"],
    filter_dict={
        "batch_id": 2,
        # "config.dataset.data_params.split_key": "split_ood_finetuning",  # split_ood_finetuning, split_random, split_ho_pathway, split_ho_epigenetic, split_ho_epigenetic_all
        # "config.model.append_ae_layer": False,
    },
)

# %%
results.head(5)

# %%
results["config.model.embedding.model"].value_counts()

# %%
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
results.isnull().any()[results.isnull().any()]

# %%
clean_id = results.loc[~results["result.training"].isnull(), "_id"]

# %%
results_clean = results[results._id.isin(clean_id)].copy()
print(f"Percentage of invalid (nan) runs: {1 - len(clean_id) / len(results)}")

# %%
results_clean["config.model.embedding.model"].value_counts()

# %%
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
[c for c in results_clean.columns if "pretrain" in c]

results_clean[
    [
        "config.model.embedding.model",
        "config.model.load_pretrained",
        "config.dataset.data_params.split_key",
    ]
].drop_duplicates()

# %%
splits = results["config.dataset.data_params.split_key"].unique()

splits

# %%
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))

# for i, y in enumerate(
#     ("result.training_mean_de", "result.val_mean_de", "result.test_mean_de")
# ):
for i, y in enumerate(splits[:3]):
    sns.violinplot(
        data=results_clean[results_clean["config.dataset.data_params.split_key"] == y],
        x="config.model.embedding.model",
        y="result.test_mean_de",
        hue="config.model.load_pretrained",
        inner="points",
        ax=ax[i],
        scale="width",
    )
    # ax[i].set_ylim([0.3,1.01])
    ax[i].set_xticklabels(["CPA", "chemCPA"])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=75, ha="right")
    ax[i].set_xlabel(y.split("_")[-1])
    ax[i].set_ylabel("test_mean_de")
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

# %%
rows = 2
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), sharex=True)

max_entangle = [0.1, 0.8]
for i, y in enumerate(
    ["result.perturbation disentanglement", "result.covariate disentanglement"]
):
    for j, (ct, df) in enumerate(
        results_clean.groupby("config.dataset.data_params.split_key")
    ):
        if j > 2:
            print(f"Igoring splits {ct}")
            continue
        sns.boxplot(
            data=df,
            x="config.model.embedding.model",
            y=y,
            # inner="point",
            # kind='violin',
            ax=ax[i, j],
            hue="config.model.load_pretrained",
        )
        axis = ax[i, j]
        # ax[i].set_ylim([0,1])
        axis.set_xticklabels(["CPA", "chemCPA"])
        axis.set_xticklabels(axis.get_xticklabels(), rotation=75, ha="right")
        axis.axhline(max_entangle[i], ls=":", color="black")
        if i == 1:
            axis.set_xlabel(ct.split("_")[-1])
        else:
            axis.set_xlabel("")

        axis.set_ylabel(y.split(".")[-1])
        axis.get_legend().remove()
ax[rows, cols].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %%
n_top = 1


def performance_condition(emb, pretrained, max_entangle, max_entangle_cov):
    cond = results_clean["config.model.embedding.model"] == emb
    cond = cond & (results_clean["result.perturbation disentanglement"] < max_entangle)
    cond = cond & (results_clean["result.covariate disentanglement"] < max_entangle_cov)
    cond = cond & (results_clean["config.model.load_pretrained"] == pretrained)
    return cond


best = []
for ct, df_ct in results_clean.groupby("config.dataset.data_params.split_key"):
    for embedding in list(results_clean["config.model.embedding.model"].unique()):
        for pretrained in [True, False]:
            df = df_ct[performance_condition(embedding, pretrained, 0.13, 0.69)]
            if len(df) == 0:
                print(
                    f"Combination {embedding} {'pretrained' if pretrained else ''} did not meet disentanglement condition."
                )
                df = df_ct[performance_condition(embedding, pretrained, 0.13, 1)]
                df = df.sort_values(
                    by="result.covariate disentanglement", ascending=True
                ).head(1)
            print(embedding, pretrained, len(df))
            best.append(
                df.sort_values(by="result.val_mean_de", ascending=False).head(n_top)
            )

best = pd.concat(best)

# %%
pd.crosstab(
    best["config.dataset.data_params.split_key"], best["config.model.embedding.model"]
)

# %%
splits = results["config.dataset.data_params.split_key"].unique()
folds = pd.Series(splits).apply(lambda s: s.split("_")[1]).unique()
cell_types = pd.Series(splits).apply(lambda s: s.split("_")[2]).unique()

print(folds)
print(cell_types)

# %%
pd.Index(cell_types).get_loc("A549")

# %%
rows = 2 * len(folds)
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), sharex=True)

max_entangle = [0.1, 0.8]
for ii, fold in enumerate(folds):
    for i, y in enumerate(
        [
            "result.test_mean",
            "result.test_mean_de",
        ]
    ):
        i += 2 * ii
        for _, (ct, df) in enumerate(
            best.groupby("config.dataset.data_params.split_key")
        ):
            if ct.split("_")[1] != fold:
                continue
            j = pd.Index(cell_types).get_loc(ct.split("_")[2])
            sns.boxplot(
                data=df,
                x="config.model.embedding.model",
                y=y,
                # inner="point",
                # kind='violin',
                ax=ax[i, j],
                hue="config.model.load_pretrained",
            )
            axis = ax[i, j]
            # ax[i].set_ylim([0,1])
            axis.set_xticklabels(["CPA", "chemCPA"])
            axis.set_xticklabels(axis.get_xticklabels(), rotation=75, ha="right")
            # axis.axhline(max_entangle[i], ls=":", color="black")
            if i == 1:
                axis.set_xlabel(ct.split("_")[-1])
            else:
                axis.set_xlabel("")

            axis.set_ylabel(y.split(".")[-1])
            axis.get_legend().remove()
    ax[i, j].legend(
        title="Pretrained",
        fontsize=18,
        title_fontsize=24,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.tight_layout()

# %%
rows, cols = 1, 4
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
# ax[0].set_ylim([0.4, 1.01])
ax[1].get_legend().remove()
# ax[1].set_ylim([0.4, 1.01])
ax[2].get_legend().remove()
ax[3].legend(
    title="Pretrained",
    fontsize=18,
    title_fontsize=24,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.tight_layout()

# %%
cols = [
    "config.model.embedding.model",
    "config.model.load_pretrained",
    "config.dataset.data_params.split_key",
    "result.val_mean_de",
    "result.test_mean",
    "result.test_mean_de",
    "result.perturbation disentanglement",
    "result.covariate disentanglement",
    "config_hash",
]

best.loc[:, cols]

# %%
best.loc[:, cols].groupby(
    ["config.model.embedding.model", "config.model.load_pretrained"]
).mean()

# %%
print(best.loc[:, cols].to_markdown())

# %%
