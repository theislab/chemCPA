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
# **Requirements**
# * According to this [paper](https://arxiv.org/pdf/1904.01561.pdf), features are computed with [descriptastorus](https://github.com/bp-kelley/descriptastorus) package
# * Install via: `pip install git+https://github.com/bp-kelley/descriptastorus`

# %% [markdown]
# ## General imports

# %%
import sys

sys.path.insert(
    0, "/"
)  # this depends on the notebook depth and must be adapted per notebook
# %%
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from chemCPA.paths import DATA_DIR, EMBEDDING_DIR

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

# %%
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

generator = MakeGenerator(("RDKit2D",))
for name, numpy_type in generator.GetColumns():
    print(f"{name}({numpy_type.__name__})")

# %%
n_jobs = 16
data = Parallel(n_jobs=n_jobs)(
    delayed(generator.process)(smiles)
    for smiles in tqdm(smiles_list, position=0, leave=True)
)

# %%
embedding = np.array(data)
embedding.shape

# %% [markdown]
# ## Check `nans` and `infs`

# %% [markdown]
# Check for `nans`

# %%
drug_idx, feature_idx = np.where(np.isnan(embedding))
print(f"drug_idx:\n {drug_idx}")
print(f"feature_idx:\n {feature_idx}")

# %% [markdown]
# Check for `infs` and add to idx lists

# %%
drug_idx_infs, feature_idx_infs = np.where(np.isinf(embedding))

drug_idx = np.concatenate((drug_idx, drug_idx_infs))
feature_idx = np.concatenate((feature_idx, feature_idx_infs))

# %% [markdown]
# Features that have these invalid values:

# %% tags=[]
np.array(generator.GetColumns())[np.unique(feature_idx)]

# %% [markdown]
# Set values to `0`

# %%
embedding[drug_idx, feature_idx]

# %%
embedding[drug_idx, feature_idx] = 0

# %% [markdown]
# ## Save

# %%
import pandas as pd

df = pd.DataFrame(
    data=embedding,
    index=smiles_list,
    columns=[f"latent_{i}" for i in range(embedding.shape[1])],
)

# Drop first feature from generator (RDKit2D_calculated)
df.drop(columns=["latent_0"], inplace=True)

# Drop columns with 0 standard deviation
threshold = 0.01
columns = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
print(f"Deleting columns with std<={threshold}: {columns}")
df.drop(
    columns=[f"latent_{idx+1}" for idx in np.where(df.std() <= 0.01)[0]], inplace=True
)

# %% [markdown]
# Check that correct columns were deleted:

# %%
np.where(df.std() <= threshold)

# %% [markdown]
# ### Normalise dataframe

# %%
normalized_df = (df - df.mean()) / df.std()

# %%
normalized_df.head()

# %% [markdown]
# Check destination folder

# %%
model_name = "rdkit2D"
fname = f"{model_name}_embedding_{dataset_name}.parquet"

directory = EMBEDDING_DIR / "rdkit" / "data" / "embeddings"
directory.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Save normalised version

# %%
normalized_df.to_parquet(directory / fname)

# %% [markdown]
# Check that it worked

# %%
df = pd.read_parquet(directory / fname)
df

# %%
