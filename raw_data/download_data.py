import gdown
import os
import urllib.request
import gzip
import tarfile
from tqdm import tqdm
import argparse
import shutil


def download_file(url, output, use_gdown=True):
    if not os.path.exists(output):
        print(f"Downloading file from {url}")
        if use_gdown:
            gdown.download(url, output, quiet=False)
        else:
            download_file_with_progress(url, output)
        print(f"File downloaded as {output}")
    else:
        print(f"File {output} already exists. Skipping download.")


def download_file_with_progress(url, output):
    response = urllib.request.urlopen(url)
    total_size = int(response.info().get('Content-Length', -1))
    block_size = 8192  # 8 KB

    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=output) as progress_bar:
        with open(output, 'wb') as file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                size = file.write(buffer)
                progress_bar.update(size)


def download_and_extract_gzip(url, output, max_retries=3):
    # Check if final output file already exists
    if os.path.exists(output):
        print(f"File {output} already exists. Skipping download and extraction.")
        return
        
    gzip_file = output + '.gz'
    
    for attempt in range(max_retries):
        try:
            if not os.path.exists(gzip_file) or attempt > 0:
                download_file(url, gzip_file, use_gdown=False)
            
            print(f"Extracting {gzip_file}")
            with gzip.open(gzip_file, 'rb') as f_in:
                with open(output, 'wb') as f_out:
                    with tqdm(unit='B', unit_scale=True, desc="Extracting", total=os.path.getsize(gzip_file)) as pbar:
                        while True:
                            chunk = f_in.read(8192)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
            
            os.remove(gzip_file)
            print(f"Extracted file saved as {output}")
            return  # Successful extraction, exit the function
        except (EOFError, gzip.BadGzipFile) as e:
            print(f"Error during extraction (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if os.path.exists(gzip_file):
                os.remove(gzip_file)
            if os.path.exists(output):
                os.remove(output)
            if attempt == max_retries - 1:
                print(f"Failed to download and extract {url} after {max_retries} attempts.")
                raise


def extract_tar(tar_file, output_dir):
    print(f"Extracting {tar_file} to {output_dir}")
    with tarfile.open(tar_file, 'r') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, output_dir)
    os.remove(tar_file)
    print(f"Extracted files to {output_dir}")


def download_files(force_redownload=False):
    project_folder = "project_folder"
    
    # List of files to download: (url, relative_path, is_gzip, is_tar)
    files_to_download = [
        ("https://drive.google.com/uc?export=download&id=18QkyADzuM8b7lMxRg94jufHaKRPkzEFw",
         "datasets/adata_biolord_split_30.h5ad", False, False),
        ("https://drive.google.com/uc?export=download&id=1oV2o5dVEVE3OwBVZzuuJTXuaamZJeFL9",
         "embeddings/rdkit/data/embeddings/rdkit2D_embedding_biolord.parquet", False, False),
        ("https://f003.backblazeb2.com/file/chemCPA-datasets/lincs_full.h5ad.gz",
         "datasets/lincs_full.h5ad", True, False),
        ("https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar",
         "binaries/cpa_binaries.tar", False, True),
    ]

    for url, relative_path, is_gzip, is_tar in files_to_download:
        # Create the full path
        full_path = os.path.join(project_folder, relative_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # If force_redownload is True, remove the existing file
        if force_redownload and os.path.exists(full_path):
            print(f"Removing existing file: {full_path}")
            os.remove(full_path)
        
        # Download and extract based on file type
        if is_gzip:
            download_and_extract_gzip(url, full_path)
        elif is_tar:
            # Download tar file if it doesn't exist
            download_file(url, full_path, use_gdown=False)
            # Extract if the tar file exists
            if os.path.exists(full_path):
                extract_tar(full_path, os.path.dirname(full_path))
        else:
            download_file(url, full_path)

