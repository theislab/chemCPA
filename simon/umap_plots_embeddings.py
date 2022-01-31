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
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sn
import torch
import umap.plot
from rdkit import Chem

from compert.embedding import get_chemical_representation

# To set some sane defaults
matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.facecolor"] = "white"
sn.set_context("poster")

embedding_dir = Path.cwd().parent / "embeddings"
data_dir = Path.cwd().parent / "datasets"
print("EMBEDDING DIR:", embedding_dir)
print("DATA DIR:", data_dir)

# %% pycharm={"name": "#%%\n"}
anndata = sc.read(data_dir / "trapnell_cpa.h5ad")

# %% pycharm={"name": "#%%\n"}
smiles_to_pathway_map = {
    Chem.CanonSmiles(smiles): pathway
    for smiles, pathway in anndata.obs.groupby(
        ["SMILES", "pathway_level_1"]
    ).groups.keys()
}

# %% pycharm={"name": "#%%\n"}
canon_smiles = list(smiles_to_pathway_map.keys())
print(canon_smiles[:10])

# %% pycharm={"name": "#%%\n"}
mappers = []
embedding_names = ("jtvae", "seq2seq", "rdkit", "grover_base")
for i, embedding_name in enumerate(embedding_names):
    print(embedding_name)
    embedding = get_chemical_representation(
        smiles=canon_smiles,
        embedding_model=embedding_name,
        device="cpu",
        data_dir=embedding_dir,
    ).weight
    assert len(embedding) == len(canon_smiles)
    df = pd.DataFrame(
        {
            "embedding": [t.numpy() for t in torch.unbind(embedding)],
            "smiles": canon_smiles,
            "pathway": [smiles_to_pathway_map[smiles] for smiles in canon_smiles],
        }
    )
    mappers.append(
        (
            umap.UMAP(n_neighbors=8, min_dist=0.15).fit_transform(embedding),
            df["pathway"],
        )
    )

# %% pycharm={"name": "#%%\n"}
fig, axs = plt.subplots(2, 2)
fig.set_figheight(15)
fig.set_figwidth(15)
emb_nice_title = {
    "jtvae": "JTVAE",
    "grover_base": "GROVER",
    "rdkit": "RDKit",
    "seq2seq": "seq2seq",
}
for i, ((mapper, pathway), embedding) in enumerate(zip(mappers, embedding_names)):
    ax = sn.scatterplot(
        x=mapper[:, 0],
        y=mapper[:, 1],
        ax=axs[i // 2][i % 2],
        hue=pathway,
        palette="tab20",
    )
    ax.set_title(emb_nice_title[embedding])
    if i == 3:
        # legend on the right
        # bbox = (1.05, 2.05)
        # legend below
        bbox = (0.5, -0.15)
        plt.legend(bbox_to_anchor=bbox)
    else:
        ax.get_legend().remove()
figure = plt.gcf()
figure.savefig(
    "../../Latex/figures/05_results/UMAP_latent_embeddings_raw.png", bbox_inches="tight"
)
plt.show()
