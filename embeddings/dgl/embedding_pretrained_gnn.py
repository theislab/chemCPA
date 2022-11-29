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

# %% [markdown]
# ## General imports

# %%
import sys

sys.path.insert(
    0, "/"
)  # this depends on the notebook depth and must be adapted per notebook
# %%
import numpy as np
from compert.paths import DATA_DIR, EMBEDDING_DIR
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)

# %% [markdown]
# ## Load Smiles list

# %%
dataset_name = "lincs_trapnell"

# %%
import pandas as pd

smiles_df = pd.read_csv(EMBEDDING_DIR / f"{dataset_name}.smiles")
smiles_list = smiles_df["smiles"].values

# %%
print(f"Number of smiles strings: {len(smiles_list)}")

# %% [markdown]
# ## Featurizer functions

# %%
node_feats = CanonicalAtomFeaturizer(atom_data_field="h")
edge_feats = CanonicalBondFeaturizer(bond_data_field="h", self_loop=True)

# %% [markdown]
# ## Create graphs from smiles and featurizers

# %%
mol_graphs = []

for smiles in smiles_list:
    mol_graphs.append(
        smiles_to_bigraph(
            smiles=smiles,
            add_self_loop=True,
            node_featurizer=node_feats,
            edge_featurizer=edge_feats,
        )
    )

# %%
print(f"Number of molecular graphs: {len(mol_graphs)}")

# %% [markdown]
# ## Batch graphs

# %%
import dgl

mol_batch = dgl.batch(mol_graphs)

# %%
mol_batch

# %% [markdown]
# ## Load pretrained model

# %% [markdown]
# Choose a model form [here](https://lifesci.dgl.ai/api/model.pretrain.html)

# %%
model_name = "GCN_canonical_PCBA"
# model_name = 'MPNN_canonical_PCBA'
# model_name = 'AttentiveFP_canonical_PCBA'
# model_name = 'Weave_canonical_PCBA'
# model_name = 'GCN_Tox21'

# %%
from dgllife.model import load_pretrained

model = load_pretrained(model_name)

verbose = True
if verbose:
    print(model)

# %% [markdown]
# ## Predict with pretrained model

# %% [markdown]
# ### Take readout, just before prediction

# %%
model.eval()
# no edge features
prediction = model(mol_batch, mol_batch.ndata["h"])
# # with edge features
# prediction = model(mol_batch, mol_batch.ndata['h'], mol_batch.edata['h'])
print(f"Prediction has shape: {prediction.shape}")
prediction

# %% [markdown]
# ## Save

# %%
import pandas as pd

df = pd.DataFrame(
    data=prediction.detach().numpy(),
    index=smiles_list,
    columns=[f"latent_{i+1}" for i in range(prediction.size()[1])],
)

# %%
import os

fname = f"{model_name}_embedding_{dataset_name}.parquet"

directory = EMBEDDING_DIR / "dgl" / "data" / "embeddings"
if not directory.exists():
    os.makedirs(directory)
    print(f"Created folder: {directory}")

df.to_parquet(directory / fname)

# %% [markdown]
# Check that it worked

# %%
df = pd.read_parquet(directory / fname)
df

# %%
df.std()

# %% [markdown]
# ## Drawing molecules

# %%
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw

# %%
mols = [Chem.MolFromSmiles(s) for s in smiles_list[:14]]
Draw.MolsToGridImage(mols, molsPerRow=7, subImgSize=(180, 150))

# %%
