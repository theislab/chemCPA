# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
# ---

# %% [markdown]
# **Requires**
# * `'lincs_full_smiles_sciplex_genes.h5ad'`
# * `'trapnell_cpa.h5ad'`
#
# **Output**
# * `'lincs_complete.h5ad'`
#
# ## Imports

# %%
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sfaira
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

from compert.paths import DATA_DIR, PROJECT_DIR

IPythonConsole.ipython_useSVG = False
matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
os.getcwd()
pd.set_option("display.max_columns", 100)
sc.set_figure_params(dpi=80, frameon=False)
sc.logging.print_header()
sns.set_context("poster")

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Load data

# %%
adata_lincs = sc.read(PROJECT_DIR / "datasets" / "lincs_full_smiles_sciplex_genes.h5ad")

# %%
adata_sciplex = sc.read(PROJECT_DIR / "datasets" / "trapnell_cpa.h5ad")

# %%
adata_lincs

# %%
adata_sciplex


# %% [markdown]
# ## Create the `drugs_df` that includes the trapnell ood drugs

# %%
def tanimoto_score(input_smiles, target_smiles):
    input_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(input_smiles))
    target_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(target_smiles))
    return DataStructs.TanimotoSimilarity(input_fp, target_fp)


# %%
ood_drugs = ["SNS-314", "Flavopiridol", "Roscovitine"]
ood_drugs.extend(
    ["Azacitidine", "Pracinostat", "Trichostatin", "Quisinostat", "Tazemetostat"]
)
ood_drugs.extend(["Cediranib", "Crizotinib", "Motesanib", "BMS-754807", "Nintedanib"])

# %%
drugs_df = adata_sciplex.obs.loc[
    adata_sciplex.obs.condition.isin(ood_drugs),
    ["condition", "SMILES", "pathway_level_1"],
]
drugs_df = drugs_df.drop_duplicates()

# %%
drugs_df

# %% [markdown]
# ## Create the `lincs_df` that contains all drugs from the lincs dataset

# %%
lincs_df = adata_lincs.obs.loc[:, ["condition", "canonical_smiles"]].drop_duplicates()
lincs_df.canonical_smiles

# %% [markdown]
# ## Compare the tanimoto similarity between the ood drugs and all lincs drugs
# And add them to the `lincs_df` as `tanimoto_sim_{condition}` columns

# %%
from tqdm.notebook import tqdm

smiles_trapnell = []
smiles_lincs = []
for i, (drug, smiles, pathway) in tqdm(drugs_df.iterrows()):
    tanimoto_sim_col = f"tanimoto_sim_{drug}"
    lincs_df[tanimoto_sim_col] = lincs_df.canonical_smiles.apply(
        lambda lincs_smiles: tanimoto_score(lincs_smiles, smiles)
    )
    most_similar = lincs_df.sort_values(tanimoto_sim_col, ascending=False).head(2)
    #     print(most_similar)
    smiles_trapnell.append(smiles)
    smiles_lincs.append(most_similar)
#     print(drug, any(lincs_df.canonical_smiles.isin([smiles])), most_similar[tanimoto_sim_col], most_similar["drug"])
#     print(lincs_df.sort_values(tanimoto_sim_col, ascending=False).head(5)[["drug", tanimoto_sim_col]])

# %% [markdown]
# ## Add the top two most siilar drugs to the ood set in lincs

# %%
ood_idx = []
n_ood = 2

for condition in ood_drugs:
    idx = (
        lincs_df[f"tanimoto_sim_{condition}"]
        .sort_values(ascending=False)[:4]
        .index.tolist()
    )
    ood_idx.extend(idx[:n_ood])

lincs_df.loc[
    ood_idx, ["condition"] + [c for c in lincs_df.columns if "tanimoto_sim" in c]
]

# %% [markdown]
# ### Compare the ood_drugs with the second most similar drug in lincs

# %% tags=[]
smiles_lincs = lincs_df.loc[ood_idx[1::2], "canonical_smiles"].values.tolist()

# %%
smiles_sciplex = []
for ood_drug in ood_drugs:
    smiles_sciplex.extend(
        drugs_df.loc[drugs_df.condition == ood_drug, "SMILES"].values.tolist()
    )

# %%
for orig, lincs in zip(smiles_sciplex, smiles_lincs):
    im = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(orig), Chem.MolFromSmiles(lincs)],
        subImgSize=(600, 400),
        legends=[orig, lincs],
    )
    print(tanimoto_score(orig, lincs))
    plt.tight_layout()
    display(im)

# %% [markdown]
# ## Create the ood split for `adata_lincs`

# %%
adata_lincs.obs["split_ood_drugs"][adata_lincs.obs.condition.isin(ood_drugs)]

# %%
ood_conditions = adata_lincs.obs.condition[ood_idx].unique()

adata_lincs.obs.loc[
    adata_lincs.obs.condition.isin(ood_conditions), "condition"
].value_counts()[: len(ood_conditions)]

# %%
adata_lincs.obs["split_ood_drugs"] = adata_lincs.obs["random_split"]

adata_lincs.obs["split_ood_drugs"].value_counts()

# %%
adata_lincs.obs.loc[
    adata_lincs.obs.condition.isin(ood_conditions), "split_ood_drugs"
] = "ood"

adata_lincs.obs["split_ood_drugs"].value_counts()

# %% [markdown]
# ## Save `adata_lincs`

# %%
fname_lincs = PROJECT_DIR / "datasets" / "lincs_complete.h5ad"

sc.write(fname_lincs, adata_lincs)

# %%
