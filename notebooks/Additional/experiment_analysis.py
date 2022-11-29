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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %%
BLACK = False

if BLACK:
    plt.style.use("dark_background")
else:
    matplotlib.style.use("fivethirtyeight")
    matplotlib.style.use("seaborn-talk")
    matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
    sns.set_style("whitegrid")

matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 120
sns.set_context("poster")

# %%
df = pd.concat(
    [
        pd.read_parquet("cpa_predictions.parquet"),
        pd.read_parquet("scgen_predictions.parquet"),
    ]
)
# df = pd.concat(
#     [
#         pd.read_parquet("cpa_predictions_high_dose.parquet"),
#         pd.read_parquet("scgen_predictions_high_dose.parquet"),
#     ]
# )

# %%
df

# %%
fig, ax = plt.subplots(figsize=(20, 9))

sns.boxplot(
    data=df[df["genes"] == "all"],
    x="condition",
    y="R2",
    hue="model",
    palette="tab10",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right")
ax.set_xlabel("")
ax.set_ylabel("$E[r^2]$ on all genes")
ax.legend(
    title="Model type",
    #     fontsize=18,
    #     title_fontsize=24,
    loc="lower left",
    bbox_to_anchor=(1, 0.2),
)
ax.grid(".", color="darkgrey")
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(20, 9))

sns.boxplot(
    data=df[df["genes"] == "degs"],
    x="condition",
    y="R2",
    hue="model",
    palette="tab10",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right")
ax.set_xlabel("")
ax.set_ylabel("$E[r^2]$ on all genes")
ax.legend(
    title="Model type",
    #     fontsize=18,
    #     title_fontsize=24,
    loc="lower left",
    bbox_to_anchor=(1, 0.2),
)
ax.grid(".", color="darkgrey")
plt.tight_layout()

# %%
df.groupby(["model", "genes"]).std()

# %%
df.groupby(["model", "genes"]).mean()

# %%
DELTA = False

if DELTA:
    df["delta"] = 0

    for cond, _df in df.groupby(["cell_type", "condition", "genes"]):
        df.loc[
            df[["cell_type", "condition", "genes"]].isin(cond).prod(1).astype(bool),
            "delta",
        ] = (
            _df["R2"].values - _df.loc[_df["model"] == "baseline", "R2"].values[0]
        )

# %%
df

# %%
df1 = df[df.genes == "all"].groupby(["model"]).mean().round(2)
df2 = df[df.genes == "degs"].groupby(["model"]).mean().round(2)
df3 = df[df.genes == "all"].groupby(["model"]).median().round(2)
df4 = df[df.genes == "degs"].groupby(["model"]).median().round(2)

# %%
result_df = (
    pd.concat(
        [df1, df2, df3, df4],
        axis=1,
        keys=["Mean all genes", "Mean DEGs", "Median all genes", "Median DEGs"],
    )
    .reindex(["baseline", "scGen", "cpa", "chemCPA", "chemCPA_pretrained"])
    .rename(
        index={
            "baseline": "Baseline",
            "cpa": "CPA",
            "chemCPA_pretrained": "chemCPA pretrained",
        }
    )
)

result_df

# %%
print(result_df.to_markdown())

# %%
print(result_df.to_latex())

# %%
