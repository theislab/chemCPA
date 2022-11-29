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
# # GROVER
# Generate GROVER fingerprints for SMILES-drugs coming from LINCS + SciPlex3.
#
# Steps:
# 1. Load `lincs_trapnell.smiles` as the list of SMILES to be encoded
# 2. Generate fingerprints using GROVER
# 3. Save SMILES -> fingerprint mapping as a pandas df.
#
# from pathlib import Path
#
# import numpy as np
# import pandas as pd

# %%
import rdkit

# %%
import scanpy as sc
from rdkit import Chem

rdkit.__version__

# %%
# SET
datasets_fpath = Path("/home/icb/simon.boehm/Masters_thesis/MT_code/datasets")
all_smiles_fpath = Path.cwd().parent / "lincs_trapnell.smiles"

# %% [markdown]
# ## Step 1: Generate fingerprints
#
# - TODO: Right now we generate `rdkit_2d_normalized` features. Are these the correct ones?
# - TODO: There are pretrained & finetuned models also available, maybe that's useful for us:
#     - SIDER: Drug side effect prediction task
#     - ClinTox: Drug toxicity prediction task
#     - ChEMBL log P prediction task

# %% language="bash"
# set -euox pipefail
#
# # move csv of all smiles to be encoded into current workdir
# cp ../lincs_trapnell.smiles data/embeddings/lincs_trapnell.csv
# file="data/embeddings/lincs_trapnell.csv"
#
# # First we generate the feature embedding for the SMILES, which is an extra input
# # into GROVER
# echo "FILE: $file"
# features=$(echo $file | sed 's:.csv:.npz:')
# if [[ ! -f $features ]]; then
#     echo "Generating features: $features"
#     python scripts/save_features.py --data_path "$file" \
#                             --save_path "$features" \
#                             --features_generator rdkit_2d_normalized \
#                             --restart
# fi;
#
# # Second we input SMILES + Features into grover and get the fingerprint out
# # 'both' means we get a concatenated fingerprint of combined atoms + bonds features
# outfile=$(echo $file | sed 's:.csv:_grover_base_both.npz:')
# echo "EMB: $outfile"
# if [[ ! -f $outfile ]]; then
#     echo "Generating embedding: $outfile"
#     python main.py fingerprint --data_path "$file" \
#                        --features_path "$features" \
#                        --checkpoint_path data/model/grover_base.pt \
#                        --fingerprint_source both \
#                        --output "$outfile"
# fi;

# %%
lincs_trapnell_base = np.load("data/embeddings/lincs_trapnell_grover_base_both.npz")
print("Shape of GROVER_base embedding:", lincs_trapnell_base["fps"].shape)


# %% [markdown]
# ## Step 2: Generate DataFrame with SMILES -> Embedding mapping

# %%
def flatten(x: np.ndarray):
    assert len(x.shape) == 2 and x.shape[0] == 1
    return x[0]


embeddings_fpath = Path("data/embeddings")
smiles_file = embeddings_fpath / "lincs_trapnell.csv"
emb_file = embeddings_fpath / "lincs_trapnell_grover_base_both.npz"

# read list of smiles
smiles_df = pd.read_csv(smiles_file)
# read generated embedding (.npz has only one key, 'fps')
emb = np.load(emb_file)["fps"]
assert len(smiles_df) == emb.shape[0]

# generate a DataFrame with SMILES and Embedding in each row
final_df = pd.DataFrame(
    emb,
    index=smiles_df["smiles"].values,
    columns=[f"latent_{i+1}" for i in range(emb.shape[1])],
)
# remove duplicates indices (=SMILES) (This is probably useless)
final_df = final_df[~final_df.index.duplicated(keep="first")]
final_df.to_parquet(embeddings_fpath / "grover_base.parquet")

# %%
df = pd.read_parquet("data/embeddings/grover_base.parquet")

# %%
df

# %% [markdown]
# ## Step 3: Check
# Make extra sure the index of the generated dataframe is correct by loading our list of canonical SMILES again

# %%
all_smiles_fpath = Path.cwd().parent / "lincs_trapnell.smiles"
all_smiles = pd.read_csv(all_smiles_fpath)["smiles"].values
assert sorted(list(df.index)) == sorted(list(all_smiles))
