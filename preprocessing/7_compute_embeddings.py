# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
# ---

# # Computes embeddings for the dataset and prints their dimensions

from chemCPA.paths import DATA_DIR, PROJECT_DIR, ROOT, EMBEDDING_DIR
import sys
import os
from tqdm.auto import tqdm
import pandas as pd

# Add the parent directory of embeddings to Python path
sys.path.append(str(ROOT))

import embeddings.rdkit.embedding_rdkit as embedding_rdkit

# Define the datasets to process with their corresponding SMILES keys
datasets = [
    #('lincs_smiles.h5ad', 'SMILES'),
    ('lincs_full_smiles.h5ad', 'canonical_smiles'),  # Changed SMILES key to lowercase
    #('sciplex_complete.h5ad', 'SMILES'),
    #('adata_MCF7.h5ad', 'SMILES'),
    #('adata_MCF7_lincs_genes.h5ad', 'SMILES'),
    #('adata_K562.h5ad', 'SMILES'),
    #('adata_K562_lincs_genes.h5ad', 'SMILES'),
    #('adata_A549.h5ad', 'SMILES'),
    #('adata_A549_lincs_genes.h5ad', 'SMILES'),
    ('sciplex_complete_subset_lincs_genes_v2.h5ad', 'SMILES'),
    #('sciplex_complete_middle_subset_v2.h5ad', 'SMILES'),
    #('sciplex_complete_middle_subset_lincs_genes_v2.h5ad', 'SMILES'),
    #('sciplex_complete_v2.h5ad', 'SMILES'),
    #('sciplex_complete_lincs_genes_v2.h5ad', 'SMILES'),
    ('combo_sciplex_prep_hvg_filtered.h5ad', 'smiles_rdkit')  # Added combinatorial dataset
]

# Define desired embedding dimension
FIXED_EMBEDDING_DIM = 200  # or whatever dimension you want

# Define whether to skip variance filtering to keep dimensions consistent
SKIP_VARIANCE_FILTER = True  # Set this to True to keep all dimensions

print("\nComputing and analyzing embeddings:")
print(f"Using fixed embedding dimension: {FIXED_EMBEDDING_DIM}")
print(f"Skip variance filtering: {SKIP_VARIANCE_FILTER}")
print("-" * 50)

# Process each dataset
for dataset, smiles_key in tqdm(datasets, desc="Computing RDKit embeddings"):
    h5ad_path = os.path.join(DATA_DIR, dataset)
    base_name = os.path.splitext(dataset)[0]
    output_filename = f"{base_name}_rdkit2D_embedding.parquet"
    output_path = os.path.join(EMBEDDING_DIR, 'rdkit', output_filename)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Compute embeddings without variance filtering
        embedding_rdkit.compute_rdkit_embeddings(
            h5ad_path, 
            output_path=output_path, 
            smiles_key=smiles_key,
            skip_variance_filter=SKIP_VARIANCE_FILTER
        )
        
        # Read and analyze the generated embeddings
        embeddings_df = pd.read_parquet(output_path)
        
        print(f"\nEmbedding analysis for {dataset}:")
        print(f"Shape: {embeddings_df.shape}")
        print(f"Number of features: {embeddings_df.shape[1]}")
        print(f"Memory usage: {embeddings_df.memory_usage().sum() / 1024**2:.2f} MB")
        print(f"File location: {output_path}")
        print("-" * 50)
        
    except Exception as e:
        tqdm.write(f"Error processing {dataset}: {str(e)}")
