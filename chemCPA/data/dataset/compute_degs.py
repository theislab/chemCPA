import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

def compute_degs(drugs_names, covariate_names, dose_names, de_genes, var_names):
    """
    Compute differential gene expression (DEG) tensor for given drug-covariate-dose combinations.

    This vectorized version avoids explicit loops over combinations by using NumPy and PyTorch operations.

    Args:
        drugs_names (list): List of drug names.
        covariate_names (list): List of covariate names.
        dose_names (list): List of dose names (if dose-specific).
        de_genes (dict): Dictionary mapping drug-covariate-dose keys to lists of differentially expressed genes.
        var_names (list): List of all gene names to be considered.

    Returns:
        torch.Tensor: A binary tensor of shape (number of combinations, number of genes),
                      where 1 indicates the gene is differentially expressed for that combination,
                      and 0 indicates it is not.
    """
    start_time = time.time()
    dose_specific = len(list(de_genes.keys())[0].split("_")) == 3

    gene_to_index = {gene: i for i, gene in enumerate(var_names)}
    var_names_set = set(var_names)

    covariate_names = np.array(covariate_names, dtype=str)
    drugs_names = np.array(drugs_names, dtype=str)
    if dose_specific:
        dose_names = np.array(dose_names, dtype=str)

    if dose_specific:
        keys = np.char.add(np.char.add(np.char.add(covariate_names, '_'), drugs_names), '_')
        keys = np.char.add(keys, dose_names)
    else:
        keys = np.char.add(np.char.add(covariate_names, '_'), drugs_names)

    N = len(keys)
    print(f"Number of combinations: {N}")

    key_to_index = {key: i for i, key in enumerate(keys)}

    control_drugs = {'control', 'DMSO', 'Vehicle'}
    is_control = np.isin(drugs_names, list(control_drugs))

    degs = torch.zeros((N, len(var_names)), dtype=torch.float32)

    row_indices = []
    col_indices = []

    # Decode byte strings in de_genes keys and values
    de_genes_decoded = {}
    for key, genes in de_genes.items():
        # Decode the key if it's a byte string
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        # Decode each gene in the list if it's a byte string
        genes = [gene.decode('utf-8') if isinstance(gene, bytes) else gene for gene in genes]
        de_genes_decoded[key] = genes

    de_genes = de_genes_decoded

    for key, genes in tqdm(de_genes.items(), desc="Processing DEGs"):
        # Decode key if necessary
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        # Decode genes if they are byte strings
        genes = [gene.decode('utf-8') if isinstance(gene, bytes) else gene for gene in genes]
        idx = key_to_index.get(key)
        if idx is not None:
            if not is_control[idx]:
                valid_genes = var_names_set.intersection(genes)
                indices = [gene_to_index[gene] for gene in valid_genes]
                if indices:
                    row_indices.extend([idx] * len(indices))
                    col_indices.extend(indices)

    if row_indices:
        degs[row_indices, col_indices] = 1.0

    print(f"DEGs tensor shape: {degs.shape}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return degs

