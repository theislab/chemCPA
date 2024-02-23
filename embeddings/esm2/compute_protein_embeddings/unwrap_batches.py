import os
import torch
from tqdm import tqdm

#The compute_protein_embeddings.py script works on batches
# because gpu <-> cpu transport is expensive
# but its easier to work with single files per protein
# so we uwnrap each batch into a set of files

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_and_unwrap_pt_files(source_directory, target_directory):
    ensure_directory_exists(target_directory)
    filenames = [f for f in os.listdir(source_directory) if f.endswith(".pt")]
    for filename in tqdm(filenames, desc="Processing files"):
        file_path = os.path.join(source_directory, filename)
        data = torch.load(file_path)
        for item in data:
            entry_id_part = item['entry_id'].split('|')[1]
            target_file_path = os.path.join(target_directory, f"{entry_id_part}.pt")
            torch.save(item, target_file_path)

source_directory = "../ctx_full_esm2_3b_all_embs"
target_directory = "../ctx_full_esm2_3b_all_embs_unwrapped"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unwrap .pt files from source to target directory.")
    # These are batches, that is a single .pt file contains multiple protein embeddings - is an array of dicts, 
    parser.add_argument("source_directory", type=str, help="Source directory containing .pt files")
    # These are the unwrapped files, that is a .pt file contains a single protein embedding - is a single dict 
    parser.add_argument("target_directory", type=str, help="Target directory to save unwrapped .pt files")
    args = parser.parse_args()
    read_and_unwrap_pt_files(args.source_directory, args.target_directory)
