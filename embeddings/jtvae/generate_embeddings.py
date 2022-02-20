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

# %% [markdown]
# # JTVAE embedding
# This is a molecule embedding using the JunctionTree VAE, as implemented in DGLLifeSci.
#
# It's pretrained on LINCS + Trapnell + half of ZINC (~220K molecules total).
# LINCS contains a `Cl.[Li]` molecule which fails during encoding, so it just gets a dummy encoding.

# %%
import pickle

import pandas as pd
import rdkit
import torch
from dgllife.data import JTVAECollator, JTVAEDataset
from dgllife.model import load_pretrained
from tqdm import tqdm

print(rdkit.__version__)
print(torch.__version__)
assert torch.cuda.is_available()

# %% pycharm={"name": "#%%\n"}
from dgllife.model import JTNNVAE

from_pretrained = False
if from_pretrained:
    model = load_pretrained("JTVAE_ZINC_no_kl")
else:
    trainfile = "data/train_077a9bedefe77f2a34187eb57be2d416.txt"
    modelfile = "data/model-vaetrain-final.pt"
    vocabfile = "data/vocab-final.pkl"

    with open(vocabfile, "rb") as f:
        vocab = pickle.load(f)

    model = JTNNVAE(vocab=vocab, hidden_size=450, latent_size=56, depth=3)
    model.load_state_dict(torch.load(modelfile, map_location="cpu"))


# %% pycharm={"name": "#%%\n"}
model = model.to("cuda")

# %% pycharm={"name": "#%%\n"}
smiles = pd.read_csv("../lincs_trapnell.smiles")
# need to remove the header, before passing it to JTVAE
smiles.to_csv("jtvae_dataset.smiles", index=False, header=None)

# %% pycharm={"name": "#%%\n"}
dataset = JTVAEDataset("jtvae_dataset.smiles", vocab=model.vocab, training=False)
collator = JTVAECollator(training=False)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, collate_fn=collator, drop_last=True
)

# %% [markdown]
# ## Reconstruction demo
# Reconstruct a couple of molecules to check reconstruction performance (it's not good).

# %% pycharm={"name": "#%%\n"}
acc = 0.0
device = "cuda"
for it, (tree, tree_graph, mol_graph) in enumerate(dataloader):
    if it > 10:
        break
    tot = it + 1
    smiles = tree.smiles
    tree_graph = tree_graph.to(device)
    mol_graph = mol_graph.to(device)
    dec_smiles = model.reconstruct(tree_graph, mol_graph)
    print(dec_smiles)
    print(smiles)
    print()
    if dec_smiles == smiles:
        acc += 1
print("Final acc: {:.4f}".format(acc / tot))

# %% [markdown]
# ## Generate embeddings for all LINCS + Trapnell molecules

# %% pycharm={"is_executing": true, "name": "#%%\n"}
get_data = lambda idx: collator([dataset[idx]])
errors = []
smiles = []
latents = []
for i in tqdm(range(len(dataset))):
    try:
        _, batch_tree_graphs, batch_mol_graphs = get_data(i)
        batch_tree_graphs = batch_tree_graphs.to("cuda")
        batch_mol_graphs = batch_mol_graphs.to("cuda")
        with torch.no_grad():
            _, tree_vec, mol_vec = model.encode(batch_tree_graphs, batch_mol_graphs)
        latent = torch.cat([model.T_mean(tree_vec), model.G_mean(mol_vec)], dim=1)
        latents.append(latent)
        smiles.append(dataset.data[i])
    except Exception as e:
        errors.append((dataset.data[i], e))

# %% pycharm={"is_executing": true, "name": "#%%\n"}
# There should only be one error, a Cl.[Li] molecule.
errors

# %% pycharm={"is_executing": true, "name": "#%%\n"}
# Add a dummy embedding for the Cl.[Li] molecule
dummy_emb = torch.mean(torch.concat(latents), dim=0).unsqueeze(dim=0)
assert dummy_emb.shape == latents[0].shape
smiles.append(errors[0][0])
latents.append(dummy_emb)
assert len(latents) == len(smiles)

# %% pycharm={"is_executing": true, "name": "#%%\n"}
np_latents = [latent.squeeze().cpu().detach().numpy() for latent in latents]
final_df = pd.DataFrame(
    np_latents,
    index=smiles,
    columns=[f"latent_{i + 1}" for i in range(np_latents[0].shape[0])],
)
final_df.to_parquet("data/jtvae_dgl.parquet")

# %% pycharm={"is_executing": true, "name": "#%%\n"}
final_df

# %% pycharm={"is_executing": true, "name": "#%%\n"}
smiles = pd.read_csv("../lincs_trapnell.smiles")
smiles2 = final_df.index

# %% pycharm={"is_executing": true, "name": "#%%\n"}
set(list(smiles["smiles"])) == set(list(smiles2))
