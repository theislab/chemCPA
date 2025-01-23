import sys
import numpy as np
from chemCPA.paths import DATA_DIR, EMBEDDING_DIR
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import multiprocessing
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import logging
from pathlib import Path
from chemCPA.helper import canonicalize_smiles
import h5py
import argparse
import anndata

# Set up logging
def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "rdkit_embedding.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def embed_smile(smile):
    """Function to process a single SMILES string."""
    try:
        local_generator = MakeGenerator(("RDKit2D",))
        result = local_generator.process(smile)
        if result is None:
            logger.warning(f"Failed to process SMILES: {smile}")
        return result
    except Exception as e:
        logger.error(f"Error processing SMILES '{smile}': {str(e)}")
        return None

def embed_smiles_list(smiles_list, n_processes=16):
    """Create RDKit embeddings for a list of SMILES strings."""
    logger.info(f"Starting embedding generation for {len(smiles_list)} SMILES strings")
    
    # Filter down to unique SMILES
    unique_smiles_list = list(set(smiles_list))
    logger.info(f"Found {len(unique_smiles_list)} unique SMILES strings")
    
    # Generate embeddings in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        data = list(tqdm(
            pool.imap(embed_smile, unique_smiles_list),
            total=len(unique_smiles_list),
            desc="Generating RDKit embeddings",
            position=1,
            leave=False
        ))
    
    # Track failed SMILES
    failed_smiles = [s for s, d in zip(unique_smiles_list, data) if d is None]
    if failed_smiles:
        logger.warning(f"\nFailed to process {len(failed_smiles)} SMILES:")
        for s in failed_smiles[:10]:  # Show first 10
            logger.warning(f"  {s}")
        if len(failed_smiles) > 10:
            logger.warning("  ...")
    
    # Filter out None values
    valid_data = [(s, d) for s, d in zip(unique_smiles_list, data) if d is not None]
    unique_smiles_list = [s for s, _ in valid_data]
    data = [d for _, d in valid_data]
    
    embedding = np.array(data)
    
    # Handle NaNs and Infs
    drug_idx, feature_idx = np.where(np.isnan(embedding))
    drug_idx_infs, feature_idx_infs = np.where(np.isinf(embedding))
    drug_idx = np.concatenate((drug_idx, drug_idx_infs))
    feature_idx = np.concatenate((feature_idx, feature_idx_infs))
    
    if len(drug_idx) > 0:
        logger.warning(f"Found {len(drug_idx)} NaN/Inf values in embeddings")
                
    embedding[drug_idx, feature_idx] = 0
    
    # Map back to original SMILES list, filling with zeros if missing
    smiles_to_embedding = dict(zip(unique_smiles_list, embedding))
    embedding_dim = embedding.shape[1]
    full_embedding = []
    for smile in smiles_list:
        if smile in smiles_to_embedding:
            full_embedding.append(smiles_to_embedding[smile])
        else:
            logger.warning(f"SMILES '{smile}' missing from embeddings, filling with zeros")
            full_embedding.append(np.zeros(embedding_dim))
    
    full_embedding = np.array(full_embedding)
    
    logger.info(f"Successfully generated embeddings with shape {full_embedding.shape}")
    return full_embedding

def embed_and_save_embeddings(smiles_list, threshold=0.01, embedding_path=None, skip_variance_filter=False):
    """Process embeddings and save to parquet file."""
    logger.info("Starting embedding processing")
    logger.info(f"Number of SMILES strings loaded: {len(smiles_list)}")
    
    # Canonicalize SMILES
    canon_smiles_list = []
    for smile in smiles_list:
        canon_smile = canonicalize_smiles(smile)
        if canon_smile is not None:
            canon_smiles_list.append(canon_smile)
        else:
            logger.warning(f"Failed to canonicalize SMILES: {smile}")
    
    logger.info(f"Number of valid canonicalized SMILES: {len(canon_smiles_list)}")
    
    # Create embeddings using canonicalized SMILES
    full_embedding = embed_smiles_list(canon_smiles_list)
    
    # Create DataFrame with canonicalized SMILES as index
    df = pd.DataFrame(
        data=full_embedding,
        index=canon_smiles_list,
        columns=[f"latent_{i}" for i in range(full_embedding.shape[1])],
    )
    
    # Handle duplicate indices before processing
    if df.index.duplicated().any():
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate SMILES indices")
        df = df.loc[~df.index.duplicated(keep='first')]
    
    # Drop the first descriptor column (latent_0)
    df.drop(columns=["latent_0"], inplace=True)
    
    # Optionally drop low-variance columns
    if not skip_variance_filter:
        low_std_cols = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
        logger.info(f"Deleting columns with std<={threshold}: {low_std_cols}")
        df.drop(columns=low_std_cols, inplace=True)
    else:
        logger.info("Skipping low variance column filtering")
    
    # Normalize
    normalized_df = pd.DataFrame(
        (df - df.mean()) / df.std(),
        index=df.index,
        columns=df.columns
    )
    
    # Set output path
    if embedding_path is None:
        directory = EMBEDDING_DIR / "rdkit" / "data" / "embeddings"
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "rdkit2D_embedding.parquet"
    else:
        output_path = Path(embedding_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving embeddings for {len(normalized_df)} SMILES to {output_path}")
    normalized_df.to_parquet(output_path)
    return output_path


def validate(embedding_df, adata, smiles_key='SMILES'):
    """
    Validate by comparing canonical SMILES from the dataset vs. the
    canonical SMILES in the embedding DataFrame index.
    
    Splits on '..' if present, but NOT on single '.'.
    Then each piece is canonicalized the same way we do in embed_and_save_embeddings.
    If the canonical form is found in embedding_df.index, it won't be listed as missing.
    """
    logger.info("Starting validation of embeddings against dataset SMILES")

    dataset_canonical_smiles = set()
    for raw_smile in adata.obs[smiles_key]:
        # If ".." in raw_smile, split it into multiple
        splitted = [raw_smile]
        if ".." in raw_smile:
            splitted = [x.strip() for x in raw_smile.split("..") if x.strip()]
        
        # Canonicalize each splitted piece
        for s in splitted:
            c = canonicalize_smiles(s)
            if c is not None:
                dataset_canonical_smiles.add(c)

    # Compare canonical forms
    embedding_smiles = set(embedding_df.index)
    missing_smiles = dataset_canonical_smiles - embedding_smiles
    if missing_smiles:
        logger.warning(
            f"Found {len(missing_smiles)} SMILES in dataset that are missing from embeddings."
        )
        for smile in list(missing_smiles)[:10]:
            logger.warning(f"  {smile}")
        logger.warning("Continuing without raising an error.")
    else:
        logger.info("Validation successful! All combined SMILES are accounted for.")

    # Optional: note any extra SMILES in embeddings but not in the dataset
    extra_smiles = embedding_smiles - dataset_canonical_smiles
    if extra_smiles:
        logger.info(
            f"Note: Embeddings contain {len(extra_smiles)} additional SMILES not in dataset."
        )


def compute_rdkit_embeddings(h5ad_path, output_path=None, smiles_key='SMILES', skip_variance_filter=False):
    """
    Generate RDKit embeddings for SMILES strings from an h5ad file.
    
    Args:
        h5ad_path (str): Path to the h5ad file containing SMILES data
        output_path (str, optional): Path to save the embeddings. If None, saves to default location
        smiles_key (str): Key for SMILES data in the h5ad file
        skip_variance_filter (bool): If True, keeps all features without filtering low variance ones
    """
    # Create progress bar for main steps
    main_steps = ['Loading SMILES', 'Computing embeddings', 'Saving results', 'Validating']
    
    with tqdm(total=len(main_steps), desc="Overall progress", position=0) as pbar:
        # Step 1: Load SMILES and dataset
        logger.info(f"Loading dataset from: {h5ad_path}")
        adata = anndata.read_h5ad(h5ad_path)
        
        logger.info("Available keys in adata.obs:")
        logger.info(f"{list(adata.obs.columns)}")
        
        if smiles_key not in adata.obs.columns:
            logger.error(f"SMILES key '{smiles_key}' not found in available columns!")
            logger.info(f"Please use one of the available keys: {list(adata.obs.columns)}")
            raise KeyError(f"SMILES key '{smiles_key}' not found in dataset")
            
        raw_smiles_data = adata.obs[smiles_key].tolist()
        
        if not raw_smiles_data:
            logger.error("Failed to load SMILES data")
            return
        
        # Expand any ".." into multiple SMILES, but leave single-dot lines as-is
        expanded_smiles_data = []
        for raw_smile in raw_smiles_data:
            if ".." in raw_smile:
                splitted = [x.strip() for x in raw_smile.split("..") if x.strip()]
                expanded_smiles_data.extend(splitted)
            else:
                expanded_smiles_data.append(raw_smile)
        
        # De-duplicate
        smiles_data = list(set(expanded_smiles_data))
        logger.info(f"Total unique SMILES (after splitting '..'): {len(smiles_data)}")
        pbar.update(1)
        
        # Step 2: Process and compute embeddings
        pbar.set_description("Computing embeddings")
        output_file = embed_and_save_embeddings(
            smiles_data,
            embedding_path=output_path,
            skip_variance_filter=skip_variance_filter
        )
        pbar.update(1)
        
        # Step 3: Save and load verification
        pbar.set_description("Saving results")
        df = pd.read_parquet(output_file)
        logger.info(f"Successfully generated and saved embeddings with shape: {df.shape}")
        logger.info(f"Embeddings saved to: {output_file}")
        pbar.update(1)
        
        # Step 4: Validate (no error if missing)
        pbar.set_description("Validating")
        validate(df, adata, smiles_key)
        pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RDKit embeddings from SMILES data')
    parser.add_argument('h5ad_path', type=str, help='Path to the h5ad file containing SMILES data')
    parser.add_argument('--output_path', type=str, help='Path to save the embeddings', default=None)
    parser.add_argument('--smiles_key', type=str, default='SMILES', help='Key for SMILES data in the h5ad file')
    parser.add_argument('--skip_variance_filter', action='store_true', help='Skip dropping low-variance columns')

    args = parser.parse_args()
    
    compute_rdkit_embeddings(
        h5ad_path=args.h5ad_path,
        output_path=args.output_path,
        smiles_key=args.smiles_key,
        skip_variance_filter=args.skip_variance_filter
    )


