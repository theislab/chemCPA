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

# %%
from pathlib import Path

import matplotlib
import seaborn as sn

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
sn.set_context("poster")

# %% pycharm={"name": "#%%\n"}
zinc_dgl = Path.home() / ".dgl" / "jtvae" / "train.txt"
lincs_trapnell = Path.cwd().parent / "lincs_trapnell.smiles"
outfile = Path.cwd().parent / "lincs_trapnell.smiles.short"
assert zinc_dgl.exists() and lincs_trapnell.exists()

# %% pycharm={"name": "#%%\n"}
for p in [zinc_dgl, lincs_trapnell]:
    with open(p) as f:
        max_length = 0
        for smile in f:
            if len(smile.strip()) > max_length:
                max_length = len(smile.strip())
print(f"Max length of {p} is {max_length}")

# %% pycharm={"name": "#%%\n"}
with open(lincs_trapnell) as f:
    count = 0
    for smile in f:
        smile = smile.strip()
        if len(smile) >= 200:
            count += 1
print(f"There are {count} SMILES >= 200")

# %% pycharm={"name": "#%%\n"}
with open(lincs_trapnell) as f:
    h = []
    for smile in f:
        h.append(len(smile.strip()))

# %% pycharm={"name": "#%%\n"}
ax = sn.histplot(h)
ax.set_title("SMILES-length in LINCS")

# %% [markdown]
# ## Generate a new smiles list
# We generate a new list of SMILES that are pruned to length <= 200

# %% pycharm={"name": "#%%\n"}
with open(outfile, "w") as outfile, open(lincs_trapnell) as infile:
    for line in infile:
        line = line.strip()
        if len(line) < 200:
            outfile.write(line + "\n")

# %% pycharm={"name": "#%%\n"}
with open(Path.cwd().parent / "lincs_trapnell.smiles.mini", "w") as outfile, open(
    lincs_trapnell
) as infile:
    for line in infile:
        line = line.strip()
        if len(line) <= 120:
            outfile.write(line + "\n")
