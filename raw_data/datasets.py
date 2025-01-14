# datasets.py

import os
# Forgive me for this
try:
    from .download_utils import (
        download_file,
        download_and_extract_gzip,
        extract_tar,
    )
except ImportError:
    from download_utils import (
        download_file,
        download_and_extract_gzip,
        extract_tar,
    )

import argparse

# Base project folder
PROJECT_FOLDER = "project_folder"

# Dataset information: (url, relative_path, is_gzip, is_tar, use_gdown)
DATASETS_INFO = {
    "adata_biolord_split_30": {
        "url": "https://drive.google.com/uc?export=download&id=18QkyADzuM8b7lMxRg94jufHaKRPkzEFw",
        "relative_path": "datasets/adata_biolord_split_30.h5ad",
        "is_gzip": False,
        "is_tar": False,
        "use_gdown": True,
    },
    "rdkit2D_embedding_biolord": {
        "url": "https://drive.google.com/uc?export=download&id=1oV2o5dVEVE3OwBVZzuuJTXuaamZJeFL9",
        "relative_path": "embeddings/rdkit/data/embeddings/rdkit2D_embedding_biolord.parquet",
        "is_gzip": False,
        "is_tar": False,
        "use_gdown": True,
    },
    
    "lincs_full": {
        "url": "https://f003.backblazeb2.com/file/chemCPA-datasets/lincs_full.h5ad.gz",
        "relative_path": "datasets/lincs_full.h5ad",
        "is_gzip": True,
        "is_tar": False,
        "use_gdown": False,
    },
    "cpa_binaries": {
        "url": "https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar",
        "relative_path": "binaries/cpa_binaries.tar",
        "is_gzip": False,
        "is_tar": True,
        "use_gdown": False,
        "extract_to": ".",  # Extract to the project root folder
    },
    "lincs_pert_info": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo.txt.gz",
        "relative_path": "datasets/GSE92742_Broad_LINCS_pert_info.txt",
        "is_gzip": True,
        "is_tar": False,
        "use_gdown": False,
    },
    "drugbank_all": {
        "url": "https://drive.google.com/uc?export=download&id=18MYC6ykf2CxxFIRrGYigPNfjvF8Mu6jL",
        "relative_path": "datasets/drug_bank/drugbank_all.csv",
        "is_gzip": False,
        "is_tar": False,
        "use_gdown": True,
    },
    "trapnell_final_v7": {
        "url": "https://drive.google.com/uc?export=download&id=1_JUg631r_QfZhKl9NZXXzVefgCMPXE_9",
        "relative_path": "datasets/trapnell_final_V7.h5ad",
        "is_gzip": False,
        "is_tar": False,
        "use_gdown": True,
    },
}


def get_dataset_path(relative_path):
    """Construct the full path for a dataset."""
    return os.path.join(PROJECT_FOLDER, relative_path)


def ensure_dataset(dataset_key):
    """Ensure that the dataset is downloaded and ready to use."""
    dataset = DATASETS_INFO[dataset_key]
    url = dataset["url"]
    relative_path = dataset["relative_path"]
    full_path = get_dataset_path(relative_path)

    # Add check here
    if not os.path.exists(full_path):
        print(f"\n❌ Dataset '{dataset_key}' not found at {full_path}. Prompting download.")

    is_gzip = dataset["is_gzip"]
    is_tar = dataset["is_tar"]
    use_gdown = dataset["use_gdown"]
    extract_to = dataset.get("extract_to")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Download and extract based on file type
    if is_gzip:
        download_and_extract_gzip(url, full_path)
    elif is_tar:
        # Download tar file if it doesn't exist
        download_file(url, full_path, use_gdown=use_gdown)
        # Extract if the tar file exists
        if os.path.exists(full_path):
            if extract_to is not None:
                if os.path.isabs(extract_to):
                    extract_dir = extract_to
                else:
                    extract_dir = os.path.join(PROJECT_FOLDER, extract_to)
            else:
                extract_dir = os.path.dirname(full_path)
            # Ensure the extraction directory exists
            os.makedirs(extract_dir, exist_ok=True)
            extract_tar(full_path, extract_dir)
    else:
        download_file(url, full_path, use_gdown=use_gdown)

    return full_path



def adata_biolord_split_30():
    """Return the path to the adata_biolord_split_30 dataset."""
    dataset_key = "adata_biolord_split_30"
    dataset_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    if not os.path.exists(dataset_path):
        ensure_dataset(dataset_key)
    # Reading logic to be added later
    # For now, return the path
    return dataset_path


def rdkit2D_embedding_biolord():
    """Return the path to the rdkit2D_embedding_biolord dataset."""
    dataset_key = "rdkit2D_embedding_biolord"
    dataset_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    if not os.path.exists(dataset_path):
        ensure_dataset(dataset_key)
    # Reading logic to be added later
    return dataset_path


def lincs_full():
    """Return the path to the lincs_full dataset."""
    dataset_key = "lincs_full"
    dataset_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    if not os.path.exists(dataset_path):
        ensure_dataset(dataset_key)
    # Reading logic to be added later
    return dataset_path


def cpa_binaries():
    """Return the directory path where cpa_binaries are extracted."""
    dataset_key = "cpa_binaries"
    tar_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    output_dir = os.path.dirname(tar_path)
    # The tar file would have been extracted to output_dir
    if not os.path.exists(output_dir):
        ensure_dataset(dataset_key)
    # Reading logic to be added later
    return output_dir


def lincs_pert_info():
    """Return the path to the LINCS perturbation info file."""
    dataset_key = "lincs_pert_info"
    dataset_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    if not os.path.exists(dataset_path):
        ensure_dataset(dataset_key)
    return dataset_path


def drugbank_all():
    """Return the path to the DrugBank dataset CSV file."""
    dataset_key = "drugbank_all"
    dataset_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    if not os.path.exists(dataset_path):
        ensure_dataset(dataset_key)
    return dataset_path


def list_available_datasets():
    """Print all available datasets."""
    print("\nAvailable datasets:")
    for dataset in DATASETS_INFO.keys():
        print(f"- {dataset}")


def _ensure_cpa_binaries():
    """Internal helper to ensure CPA binaries are downloaded and extracted."""
    dataset_key = "cpa_binaries"
    tar_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    
    sciplex_path = os.path.join(PROJECT_FOLDER, 'datasets', 'sciplex_raw_chunk_0.h5ad')
    norman_path = os.path.join(PROJECT_FOLDER, 'datasets', 'norman.h5ad')
    
    if not os.path.exists(sciplex_path) or not os.path.exists(norman_path):
        ensure_dataset(dataset_key)
    
    return PROJECT_FOLDER




def sciplex():
    """Return a list of paths to all ScipLex dataset chunks.
    
    The dataset is part of the CPA binaries and consists of 5 chunks:
    sciplex_raw_chunk_0.h5ad through sciplex_raw_chunk_4.h5ad
    """
    base_dir = _ensure_cpa_binaries()
    chunk_paths = []
    for i in range(5):
        chunk_path = os.path.join(base_dir, 'datasets', f'sciplex_raw_chunk_{i}.h5ad')
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"ScipLex chunk {i} not found at {chunk_path}. "
                                  "The CPA binaries might be corrupted.")
        chunk_paths.append(chunk_path)
    return chunk_paths


def norman():
    """Return the path to the Norman dataset.
    
    The dataset is part of the CPA binaries.
    """
    base_dir = _ensure_cpa_binaries()
    dataset_path = os.path.join(base_dir, 'datasets', 'norman.h5ad')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Norman dataset not found at {dataset_path}. "
                              "The CPA binaries might be corrupted.")
    return dataset_path


def trapnell_final_v7():
    """Return the path to the Trapnell final V7 dataset."""
    dataset_key = "trapnell_final_v7"
    dataset_path = get_dataset_path(DATASETS_INFO[dataset_key]["relative_path"])
    if not os.path.exists(dataset_path):
        ensure_dataset(dataset_key)
    return dataset_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and prepare datasets for the project.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to download. Use "all" to download all datasets.')
    parser.add_argument('--list', action='store_true', help='List all available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        exit(0)
        
    if args.dataset:
        if args.dataset.lower() == 'all':
            print("Downloading all datasets...")
            for dataset_key in DATASETS_INFO.keys():
                print(f"\nProcessing {dataset_key}...")
                ensure_dataset(dataset_key)
                print(f"✅ {dataset_key} downloaded successfully")
        else:
            if args.dataset not in DATASETS_INFO:
                print(f"Error: Dataset '{args.dataset}' not found.")
                list_available_datasets()
                exit(1)
            
            print(f"Downloading {args.dataset}...")
            ensure_dataset(args.dataset)
            print(f"✅ {args.dataset} downloaded successfully")
    else:
        print("Please specify a dataset to download using --dataset or use --list to see available datasets")
        parser.print_help()
