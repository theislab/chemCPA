from pathlib import Path
from typing import List

import pandas as pd
import torch

from chemCPA.paths import EMBEDDING_DIR


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_path=None,
    device="cuda",
):
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    if data_path is None:
        assert embedding_model in (
            "grover_base",
            "weave",
            "MPNN",
            "AttentiveFP",
            "GCN",
            "seq2seq",
            "rdkit",
            "jtvae",
            "zeros",
            "chemvae",
        )
        data_path = Path(EMBEDDING_DIR)
        assert Path(data_path).exists(), f"{data_path} does not exist."

        df = None
        if embedding_model == "grover_base":
            df = pd.read_parquet(data_path / "grover" / "data" / "embeddings" / "grover_base.parquet")
        elif embedding_model == "weave":
            df = pd.read_parquet(
                data_path / "dgl" / "data" / "embeddings" / "Weave_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "MPNN":
            df = pd.read_parquet(
                data_path / "dgl" / "data" / "embeddings" / "MPNN_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "GCN":
            df = pd.read_parquet(
                data_path / "dgl" / "data" / "embeddings" / "GCN_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "AttentiveFP":
            df = pd.read_parquet(
                data_path
                / "dgl"
                / "data"
                / "embeddings"
                / "AttentiveFP_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "seq2seq":
            df = pd.read_parquet(Path(data_path) / "seq2seq" / "data" / "seq2seq.parquet")
        elif embedding_model == "rdkit":
            df = pd.read_parquet(
                data_path
                / "rdkit"
                / "data"
                / "embeddings"
                # / "rdkit2D_embedding_lincs_trapnell.parquet"
                / "rdkit2D_embedding_biolord.parquet"
            )
        elif embedding_model == "jtvae":
            df = pd.read_parquet(data_path / "jtvae" / "data" / "jtvae_dgl.parquet")
        elif embedding_model == "chemvae":
            df = pd.read_parquet(data_path / "chemvae" / "chemvae.parquet")

        if df is not None:
            emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
            assert emb.shape[0] == len(smiles)
        else:
            assert embedding_model == "zeros"
            emb = torch.zeros((len(smiles), 256))
    else:
        data_path = Path(data_path)
        assert data_path.exists(), f"{data_path} does not exist."
        df = pd.read_parquet(data_path)
        emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)

    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
