# dataset_downloader.py

import gdown
import os
import urllib.request
import gzip
import tarfile
from tqdm import tqdm
import shutil
import requests


def get_file_size(url, use_gdown=False):
    """Get file size in bytes."""
    try:
        if use_gdown:
            # For Google Drive, we need to use a different approach
            response = requests.head(url, allow_redirects=True)
            size = None  # Google Drive doesn't provide size in headers directly
        else:
            response = requests.head(url, allow_redirects=True)
            size = int(response.headers.get('content-length', 0))
        return size
    except:
        return None


def format_size(size_in_bytes):
    """Convert size in bytes to human readable format."""
    if size_in_bytes is None:
        return "unknown size"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"


def confirm_download(url, path, use_gdown=False):
    """Ask user for confirmation before downloading."""
    size = get_file_size(url, use_gdown)
    formatted_size = format_size(size)
    
    print(f"\n📦 Preparing to download:")
    print(f"   • File: {os.path.basename(path)}")
    print(f"   • Size: {formatted_size}")
    print(f"   • Destination: {path}")
    
    while True:
        response = input("\n⚡ Continue with download? [y/n]: ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("❌ Download cancelled")
            return False
        print("Please answer 'y' or 'n'")


def download_file(url, path, use_gdown=False):
    """Download a file from a URL to a specified path."""
    if os.path.exists(path):
        print(f"✨ File already exists at {path}")
        return
    
    if not confirm_download(url, path, use_gdown):
        exit(0)
    
    print(f"🚀 Starting download...")
    
    if use_gdown:
        gdown.download(url, path, quiet=False)
    else:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total_size)
                    downloaded_formatted = format_size(downloaded)
                    total_formatted = format_size(total_size)
                    print(f"\r💫 Progress: [{'=' * done}{' ' * (50-done)}] {downloaded_formatted}/{total_formatted}", end='')
        print("\n✅ Download complete!")


def download_and_extract_gzip(url, output_path):
    """Download a gzip file and extract it."""
    if os.path.exists(output_path):
        print(f"✨ File already exists at {output_path}")
        return
    
    # Download to temporary gzip file
    temp_gz_path = output_path + '.gz'
    download_file(url, temp_gz_path)
    
    print(f"📂 Extracting {temp_gz_path}...")
    with gzip.open(temp_gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up
    os.remove(temp_gz_path)
    print("✨ Extraction complete!")


def extract_tar(tar_path, output_dir):
    """Extract a tar file to the specified directory."""
    print(f"📂 Extracting {tar_path}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)
    print("✨ Extraction complete!")
