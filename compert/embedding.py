import pandas as pd
import numpy as np
import torch
from typing import List
from pathlib import Path


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_dir="embeddings/",
    device="cuda",
):
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    assert embedding_model in ("grover_base",)
    assert Path(data_dir).exists()

    if embedding_model == "grover_base":
        df = pd.read_parquet(
            Path(data_dir) / "grover" / "data" / "embeddings" / "grover_base.parquet"
        )

    # we append an all zero vector as the embedding for control
    df = df.append(
        pd.DataFrame(
            np.zeros((1, len(df.columns)), dtype=np.float32),
            columns=df.columns,
            index=["control_SMILES_0"],
        )
    )
    # control -> `smiles is None`. Replace with our magic value so it works with df.loc[]
    smiles = ["control_SMILES_0" if smile is None else smile for smile in smiles]
    emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
    assert emb.shape[0] == len(smiles)
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
