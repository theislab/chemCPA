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

# %% pycharm={"name": "#%%\n"}
import deepchem as dc
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from train_model import MAX_LENGTH, TOKENS

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# %% [markdown]
# ## Load the most recent checkpoint
# I stored all checkpoints in `embeddings/seq2seq/data`

# %% pycharm={"name": "#%%\n"}
model = dc.models.SeqToSeq(
    TOKENS,
    TOKENS,
    MAX_LENGTH,
    encoder_layers=2,
    decoder_layers=2,
    embedding_dimension=256,
    batch_size=100,
    model_dir="data",
)

# %% pycharm={"name": "#%%\n"}
model.get_checkpoints()

# %% pycharm={"name": "#%%\n"}
# loads the newest checkpoint
model.restore()

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Load all SMILES
# and predict their embedding

# %% pycharm={"name": "#%%\n"}
canonicalize = lambda smile: Chem.MolToSmiles(Chem.MolFromSmiles(smile))
all_smiles = list(pd.read_csv("../lincs_trapnell.smiles")["smiles"].values)

# %% pycharm={"name": "#%%\n"}
# quick check on subset of all embeddings
pred = model.predict_from_sequences(all_smiles[0:15])
for s_pred, s_real in zip(pred, all_smiles[0:15]):
    s_pred = "".join(s_pred)
    print(f"{s_pred == s_real}\n-- {s_real}\n-- {s_pred}")

# %% pycharm={"name": "#%%\n"}
# actually predict all embeddings
emb = model.predict_embeddings(all_smiles)

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Store the resulting embedding

# %% pycharm={"name": "#%%\n"}
final_df = pd.DataFrame(
    emb, index=all_smiles, columns=[f"latent_{i+1}" for i in range(emb.shape[1])]
)

# %% pycharm={"name": "#%%\n"}
final_df.to_parquet("data/seq2seq.parquet")
final_df

# %% pycharm={"name": "#%%\n"}
assert sorted(pd.read_csv("../lincs_trapnell.smiles")["smiles"].values) == sorted(
    final_df.index.values
)
