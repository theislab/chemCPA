#!/usr/bin/env python
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import torch
import scanpy as sc
import numpy as np
import pandas as pd
from lightning_module import ChemCPA
from embeddings.rdkit.embedding_rdkit import embed_and_save_embeddings
from tqdm import tqdm
import random

# CONSTANTS
BATCH_SIZE = 4096
DATASET_FILE = "/home/user/test/chemCPA/project_folder/datasets/lincs_full_smiles_sciplex_genes.h5ad"
CKPT_RELATIVE_PATH = Path("training_output", "training_output", "lincs_pretrain", "epoch=199-step=819900.ckpt")
SMILES_KEY = "canonical_smiles"
EMBEDDINGS_OUTPUT_PATH = "/tmp/10_random_rdkit_embeddings.parquet"
EMBEDDING_THRESHOLD = 0.01
SKIP_VARIANCE_FILTER = True
DROP_COLS = ['latent_90', 'latent_103', 'latent_152', 'latent_164',
             'latent_176', 'latent_187', 'latent_196']

# Cache file paths
CONTROL_CACHE_PATH = "cached_control_genes.pt"
DRUG_EMBEDDINGS_CACHE_PATH = "cached_random_drug_embeddings.pt"

def remove_non_alphanumeric(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s)

def load_control_genes(file_path, device):
    if os.path.exists(CONTROL_CACHE_PATH):
        genes_tensor = torch.load(CONTROL_CACHE_PATH, map_location=device)
        print("Loading cached control genes from", CONTROL_CACHE_PATH)
        print("Control genes shape:", genes_tensor.shape)
        return genes_tensor
    adata = sc.read(file_path)
    if 'condition' not in adata.obs.columns:
        if 'pert_iname' in adata.obs.columns:
            adata.obs['condition'] = adata.obs['pert_iname'].apply(remove_non_alphanumeric)
        else:
            raise KeyError("Neither 'condition' nor 'pert_iname' found in adata.obs")
    control_adata = adata[adata.obs['condition'] == 'DMSO']
    if hasattr(control_adata.X, "toarray"):
        genes_tensor = torch.tensor(control_adata.X.toarray(), dtype=torch.float32, device=device)
    else:
        genes_tensor = torch.tensor(control_adata.X, dtype=torch.float32, device=device)
    torch.save(genes_tensor, CONTROL_CACHE_PATH)
    print("Saved cached control genes to", CONTROL_CACHE_PATH)
    print("Control genes shape:", genes_tensor.shape)
    return genes_tensor

def load_random_drug_embeddings(device):
    if os.path.exists(DRUG_EMBEDDINGS_CACHE_PATH):
        drug_tensor = torch.load(DRUG_EMBEDDINGS_CACHE_PATH, map_location=device)
        print("Loading cached random drug embeddings from", DRUG_EMBEDDINGS_CACHE_PATH)
        print("Drug embeddings shape:", drug_tensor.shape)
        return drug_tensor
    adata = sc.read(DATASET_FILE)
    if SMILES_KEY not in adata.obs.columns:
        raise KeyError(f"SMILES key '{SMILES_KEY}' not found in dataset!")
    all_smiles = adata.obs[SMILES_KEY].dropna().tolist()
    if len(all_smiles) < 10:
        raise ValueError("Dataset contains fewer than 10 SMILES strings!")
    random_smiles = list(np.random.choice(all_smiles, size=10, replace=False))
    print("Selected 10 random SMILES:")
    for s in random_smiles:
        print(s)
    embed_and_save_embeddings(
        random_smiles,
        threshold=EMBEDDING_THRESHOLD,
        embedding_path=EMBEDDINGS_OUTPUT_PATH,
        skip_variance_filter=SKIP_VARIANCE_FILTER
    )
    embeddings_df = pd.read_parquet(EMBEDDINGS_OUTPUT_PATH)
    present_drop_cols = [col for col in DROP_COLS if col in embeddings_df.columns]
    if present_drop_cols:
        embeddings_df = embeddings_df.drop(columns=present_drop_cols)
    print("Computed RDKit embeddings (after manual dropping):")
    print(embeddings_df)
    print("Final shape:", embeddings_df.shape)
    drug_tensor = torch.tensor(embeddings_df.values, dtype=torch.float32, device=device)
    torch.save(drug_tensor, DRUG_EMBEDDINGS_CACHE_PATH)
    print("Saved cached drug embeddings to", DRUG_EMBEDDINGS_CACHE_PATH)
    print("Drug embeddings shape:", drug_tensor.shape)
    return drug_tensor

def batched_predict_flat(model, genes, drug_embedding, cov_tensor, batch_size):
    total = genes.shape[0]
    preds_list = []
    cell_drug_list = []
    for start in tqdm(range(0, total, batch_size), desc="Batched prediction", unit="batch"):
        end = min(start + batch_size, total)
        batch_genes = genes[start:end]
        batch_drug = drug_embedding[start:end]
        batch_cov = cov_tensor[start:end]
        if start == 0:
            print("Inside batched_predict_flat, first batch shapes:")
            print("  batch_genes:", batch_genes.shape)
            print("  batch_drug :", batch_drug.shape)
            print("  batch_cov  :", batch_cov.shape)
        with torch.no_grad():
            preds, cell_drug_emb = model.model.compute_prediction(batch_genes, batch_drug, [batch_cov])
        preds_list.append(preds.cpu())
        cell_drug_list.append(cell_drug_emb.cpu())
    return torch.cat(preds_list, dim=0), torch.cat(cell_drug_list, dim=0)

def predict(model_path, drug_embeddings, covariate_ids, gene_control_data, device=None, use_pairs=True, batch_size=4096):
    """
    Args:
        model_path: path to the model checkpoint.
        drug_embeddings: Tensor of shape [n_drugs, emb_dim] (e.g. 10 x 193).
        covariate_ids: List of ints, one per covariate dimension.
        gene_control_data: Tensor of shape [n_cells, gene_dim].
        device: torch.device; if None, it is derived from the model.
        use_pairs: whether to include pairs of drugs in addition to singles.
        batch_size: batch size for prediction.
    Returns:
        preds: predictions, Tensor of shape [n_cells, n_combos, out_dim].
        cell_drug_emb: cell-drug embeddings, same shape as preds.
        drug_combos: a list of drug combinations (each is a list of drug indices).
    """
    model = ChemCPA.load_from_checkpoint(str(model_path))
    model.eval()
    if device is None:
        device = next(model.model.parameters()).device
    gene_control_data = gene_control_data.to(device)
    drug_embeddings = drug_embeddings.to(device)

    # Prepare drug combinations (singles and, if use_pairs True, pairs)
    n_drugs, emb_dim = drug_embeddings.shape
    drug_combos = [[i] for i in range(n_drugs)]
    if use_pairs:
        for i in range(n_drugs):
            for j in range(i+1, n_drugs):
                drug_combos.append([i, j])
    n_combos = len(drug_combos)
    max_combo = max(len(combo) for combo in drug_combos)
    combo_emb_input = torch.zeros((n_combos, max_combo, emb_dim), device=device)
    dosages = torch.zeros((n_combos, max_combo), device=device)
    drugs_idx = torch.zeros((n_combos, max_combo), dtype=torch.long, device=device)
    for idx, combo in enumerate(drug_combos):
        for pos, d_idx in enumerate(combo):
            combo_emb_input[idx, pos] = drug_embeddings[d_idx]
            dosages[idx, pos] = 1.0
            drugs_idx[idx, pos] = d_idx

    print("Drug embeddings shape:", drug_embeddings.shape)
    print("Drug embeddings with combos shape:", combo_emb_input.sum(dim=1).shape)

    with torch.no_grad():
        drug_combo_transformed = model.model._transform_drug_embeddings(combo_emb_input, dosages, drugs_idx)
    latent_dim = drug_combo_transformed.shape[-1]
    print("Transformed drug combo embeddings shape:", drug_combo_transformed.shape)

    # Process covariate_ids into one-hot vectors.
    num_covariates = model.model.num_covariates
    cov_config = []
    for cid, n in zip(covariate_ids, num_covariates):
        one_hot = torch.zeros((1, n), device=device)
        one_hot[0, cid] = 1.0
        cov_config.append(one_hot)
    with torch.no_grad():
        cov_emb_list = model.model.compute_covariate_embeddings_(cov_config)
    combined_cov = sum(cov_emb_list)
    if combined_cov.dim() != 2 or combined_cov.size(0) != 1:
        combined_cov = combined_cov.unsqueeze(0)
    print("Combined covariate shape:", combined_cov.shape)  # expected [1, latent_dim]

    # Expand inputs:
    n_cells, gene_dim = gene_control_data.shape
    genes_expanded = gene_control_data.unsqueeze(1).expand(n_cells, n_combos, gene_dim)
    drug_expanded = drug_combo_transformed.unsqueeze(0).expand(n_cells, n_combos, latent_dim)
    cov_expanded = combined_cov.unsqueeze(0).unsqueeze(0).expand(n_cells, n_combos, latent_dim)

    total = n_cells * n_combos
    print("n_cells =", n_cells, "n_combos =", n_combos)
    print("Total combined predictions to compute:", total)
    genes_flat = genes_expanded.reshape(total, gene_dim)
    drug_flat = drug_expanded.reshape(total, latent_dim)
    cov_flat = cov_expanded.reshape(total, latent_dim)
    print("genes_flat shape:", genes_flat.shape)
    print("drug_flat shape:", drug_flat.shape)
    print("cov_flat shape:", cov_flat.shape)

    preds_all_flat, cell_drug_emb_all_flat = batched_predict_flat(model, genes_flat, drug_flat, cov_flat, batch_size)
    out_dim = preds_all_flat.shape[-1]
    preds = preds_all_flat.reshape(n_cells, n_combos, out_dim)
    cell_drug_emb = cell_drug_emb_all_flat.reshape(n_cells, n_combos, out_dim)
    return preds, cell_drug_emb, drug_combos

def prepare():
    """
    Prepares the necessary inputs for prediction.
    Returns:
        gene_control_data: Tensor with gene control data.
        drug_embeddings: Tensor with drug embeddings.
        covariate_ids: List of ints (one per covariate dimension).
        model_path: Path to the model checkpoint.
        device: torch.device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_control_data = load_control_genes(DATASET_FILE, device)
    drug_embeddings = load_random_drug_embeddings(device)
    # For this example, assume the model has two covariate dimensions; use 0 for both.
    covariate_ids = [0, 0]
    model_path = Path(__file__).resolve().parent.parent / CKPT_RELATIVE_PATH
    return gene_control_data, drug_embeddings, covariate_ids, model_path, device

def main():
    gene_control_data, drug_embeddings, covariate_ids, model_path, device = prepare()
    preds, cell_drug_emb, drug_combos = predict(model_path, drug_embeddings, covariate_ids, gene_control_data, device=device)
    print("Final predictions shape:", preds.shape)
    print("Final cell-drug embedding shape:", cell_drug_emb.shape)
    print("Drug combos (singles and pairs):")
    for combo in drug_combos:
        print(combo)

if __name__ == "__main__":
    main()
