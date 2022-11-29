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

# %%
import moses
import numpy as np
import pandas as pd
import torch
from moses.vae import VAE
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.CRITICAL)
RDLogger.DisableLog("rdApp.*")

# %%
config_fpath = Path(
    "/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings/chemvae/config.txt"
)
state_dict_fpath = Path(
    "/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings/chemvae/vae_checkpoint_final.pt"
)

# %% pycharm={"name": "#%%\n"}
config = torch.load(config_fpath)
vocab = torch.load(config.vocab_save)
state = torch.load(state_dict_fpath)

# %%
model = VAE(vocab, config)
model.load_state_dict(state)
model.to("cuda")
model.eval()

# %%
all_smiles = list(
    pd.read_csv(
        "/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings/lincs_trapnell.smiles"
    )["smiles"].values
)

# %%
embeddings = []
for s in tqdm(all_smiles):
    with torch.no_grad():
        tensors = [model.string2tensor(s)]
        emb, _ = model.forward_encoder(tensors)
    embeddings.append(emb.cpu().numpy())

# %%
emb = np.concatenate(embeddings, axis=0)
final_df = pd.DataFrame(
    emb, index=all_smiles, columns=[f"latent_{i+1}" for i in range(emb.shape[1])]
)
final_df.to_parquet("chemvae.parquet")
final_df


# %% [markdown]
# ## Bit of testing
#
# Testing sampled SMILES for validitiy

# %%
def smiles_is_syntatically_valid(smiles):
    return Chem.MolFromSmiles(smiles, sanitize=False) is not None


def smiles_is_semantically_valid(smiles):
    valid = True
    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(smiles, sanitize=False))
    except:
        valid = False
    return valid


# %%
samples = model.sample(1000)

# %%
syn_valid = sum(smiles_is_syntatically_valid(s) for s in samples) / len(samples)
sem_valid = sum(smiles_is_syntatically_valid(s) for s in samples) / len(samples)
print(f"TOTAL: {len(samples)} SYN: {syn_valid} SEM: {sem_valid}")

# %%
samples

# %%
