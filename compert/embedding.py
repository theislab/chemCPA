from pathlib import Path
from typing import List

import pandas as pd
import torch

from compert.paths import EMBEDDING_DIR


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_dir=None,
    device="cuda",
):
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    assert embedding_model in ("grover_base", "weave", "MPNN", "AttentiveFP", "GCN")

    if data_dir is None:
        data_dir = Path(EMBEDDING_DIR)
    else:
        data_dir = Path(data_dir)
    assert Path(data_dir).exists()

    if embedding_model == "grover_base":
        df = pd.read_parquet(
            data_dir / "grover" / "data" / "embeddings" / "grover_base.parquet"
        )
    elif embedding_model == "weave":
        df = pd.read_parquet(
            data_dir
            / "dgl"
            / "data"
            / "embeddings"
            / "Weave_canonical_PCBA_embedding_lincs_trapnell.parquet"
        )
    elif embedding_model == "MPNN":
        df = pd.read_parquet(
            data_dir
            / "dgl"
            / "data"
            / "embeddings"
            / "MPNN_canonical_PCBA_embedding_lincs_trapnell.parquet"
        )
    elif embedding_model == "GCN":
        df = pd.read_parquet(
            data_dir
            / "dgl"
            / "data"
            / "embeddings"
            / "GCN_canonical_PCBA_embedding_lincs_trapnell.parquet"
        )
    elif embedding_model == "AttentiveFP":
        df = pd.read_parquet(
            data_dir
            / "dgl"
            / "data"
            / "embeddings"
            / "AttentiveFP_canonical_PCBA_embedding_lincs_trapnell.parquet"
        )

    emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
    assert emb.shape[0] == len(smiles)
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
