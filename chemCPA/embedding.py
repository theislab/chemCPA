from pathlib import Path
from typing import List
import pandas as pd
import torch
import numpy as np

from chemCPA.paths import EMBEDDING_DIR


def _check_smiles_availability(smiles: List[str], df: pd.DataFrame) -> tuple[set, set, list]:
    """
    Check which SMILES are present in the DataFrame and which are missing.
    
    Args:
        smiles: List of SMILES strings to check
        df: DataFrame containing embeddings with SMILES as index
        
    Returns:
        Tuple containing:
        - set of present SMILES
        - set of missing SMILES
        - list of embeddings (with zeros for missing SMILES)
    """
    present_smiles = set(smiles) & set(df.index)
    missing_smiles = set(smiles) - present_smiles
    
    print(f"Matching SMILES: {len(present_smiles)} out of {len(smiles)}")
    print(f"Missing SMILES: {len(missing_smiles)}")
    
    if len(missing_smiles) > 0 and len(missing_smiles) <= 10:
        print("Missing SMILES:")
        for s in missing_smiles:
            print(f"  {s}")
    elif len(missing_smiles) > 10:
        print("More than 10 SMILES are missing. First 10:")
        for s in list(missing_smiles)[:10]:
            print(f"  {s}")

    if len(missing_smiles) > 0:
        print("Using zero vectors for missing SMILES.")
    
    # Create embeddings, using zeros for missing SMILES
    emb_list = []
    print(f"DataFrame shape: {df.shape}")
    
    # Add shape verification
    shapes = set()
    for i, s in enumerate(smiles):
        if s in df.index:
            embedding = df.loc[s].values
            shapes.add(embedding.shape)
            if len(shapes) > 1:
                print(f"WARNING: Inconsistent shapes detected at index {i}")
                print(f"Current shapes found: {shapes}")
                print(f"Problematic SMILE: {s}")
                print(f"Current embedding shape: {embedding.shape}")
            emb_list.append(embedding)
        else:
            zero_vec = np.zeros(df.shape[1])
            shapes.add(zero_vec.shape)
            emb_list.append(zero_vec)
    
    print(f"All unique shapes found: {shapes}")
    
    # Verify each embedding individually
    for i, emb in enumerate(emb_list):
        if not isinstance(emb, np.ndarray):
            print(f"Warning: embedding {i} is not a numpy array, it's a {type(emb)}") 
            
    return present_smiles, missing_smiles, emb_list


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
    print("Looking for the following number of smiles:", len(smiles))
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
            df_path = data_path / "rdkit" / "data" / "embeddings" / "embeddings/rdkit/my_lincs_embeddings.parquet"
            print(df_path)
            df = pd.read_parquet(df_path)
            print(f"Total SMILES in the embedding file: {len(df)}")
        elif embedding_model == "jtvae":
            df = pd.read_parquet(data_path / "jtvae" / "data" / "jtvae_dgl.parquet")
        elif embedding_model == "chemvae":
            df = pd.read_parquet(data_path / "chemvae" / "chemvae.parquet")

        if df is not None:
            _, _, emb_list = _check_smiles_availability(smiles, df)
            emb = torch.tensor(np.stack(emb_list), dtype=torch.float32, device=device)
            assert emb.shape[0] == len(smiles)
        else:
            assert embedding_model == "zeros"
            emb = torch.zeros((len(smiles), 256))
    else:
        print(f"Loading embeddings from {data_path}")
        data_path = Path(data_path)
        assert data_path.exists(), f"{data_path} does not exist."
        df = pd.read_parquet(data_path)
        _, _, emb_list = _check_smiles_availability(smiles, df)
        emb = torch.tensor(np.stack(emb_list), dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)

    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
