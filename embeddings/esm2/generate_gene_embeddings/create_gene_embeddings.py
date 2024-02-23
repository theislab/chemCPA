import os

import numpy as np
import torch
from tqdm import tqdm
from create_gene_protein_map.create_gene_protein_map import create_gene_protein_map
import argparse
import requests


def create_protein_entry_name_to_path_map(path):
    """
    Dictionary mapping protein accession numbers to their corresponding file paths
    """
    return {file.rsplit('.', 1)[0]: os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')}


def filter_protein_map_to_existing_embeddings(protein_accession_numbers, file_path_map):
    """
    Removes accession nubmers that don't have an emebdding in the embeddings folder
    """
    return [accession_number for accession_number in protein_accession_numbers if accession_number in file_path_map]


def filter_gene_protein_map(gene_protein_map, file_path_map):
    """ 
    Keep genes for which we have at least 1 protein embedding
    """
    filtered_map = {gene: filter_protein_map_to_existing_embeddings(protein_accession_numbers, file_path_map) for
                    gene, protein_accession_numbers in gene_protein_map.items()}
    return {gene: valid_accession_numbers for gene, valid_accession_numbers in filtered_map.items() if
            len(valid_accession_numbers) > 0}


def print_filtered_genes_info(original_gene_protein_map, filtered_gene_protein_map):
    original_count = len(original_gene_protein_map)
    filtered_count = len(filtered_gene_protein_map)
    filtered_out_count = original_count - filtered_count
    total_proteins_original = sum(len(proteins) for proteins in original_gene_protein_map.values())
    total_proteins_filtered = sum(len(proteins) for proteins in filtered_gene_protein_map.values())
    lost_protein_embeddings = total_proteins_original - total_proteins_filtered
    print(f"Original gene count: {original_count}")
    print(f"Filtered gene count: {filtered_count}")
    print(f"Number of genes filtered out: {filtered_out_count}")
    print(f"Total protein embeddings lost: {lost_protein_embeddings}")


def create_gene_embeddings(gene_protein_map, path, verbose=False):
    """
    Calculates gene embeddings based on its associated proteins.
    For each gene, this method finds corresponding proteins in the protein map, loads their embeddings and computes the average of them.

    Parameters:
    - gene_protein_map (dict): { [gene]  : protein_accession_number1, protein_accession_number2, protein_accession_number3 ...}.
    - path - path to a directory, containing a file for every protein in the gene_protein_map, with files named as {Entry}.pt, where Entry is the accession number 

    Returns:
    - dict: A dictionary of the form { [Gene name] : embedding }
    """
    file_path_map = create_protein_entry_name_to_path_map(path)
    filtered_gene_protein_map = filter_gene_protein_map(gene_protein_map, file_path_map)
    if verbose:
        print_filtered_genes_info(gene_protein_map, filtered_gene_protein_map)
    map_with_progress = tqdm(filtered_gene_protein_map.items(), desc="Creating gene embeddings")
    return {gene: torch.stack(
        [torch.load(file_path_map[prot_id])['mean_representations'][36] for prot_id in prot_ids]).mean(dim=0)
         for gene, prot_ids in map_with_progress}



def fetch_uniprot_data(output_path):
    """
    Fetches UniProtKB data for Homo sapiens (human) and saves it to a TSV file.
    This is required so that we can reconstruct an embedding map.

    Parameters:
    - output_path (str): The path to save the fetched data.
    """

    url = "https://rest.uniprot.org/uniprotkb/stream"
    query_params = {
        "query": "model_organism:9606",
        "format": "tsv",
        "fields": "gene_names,accession"
    }
    response = requests.get(url, params=query_params, stream=True)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
    else:
        print(f"Failed to fetch data: {response.status_code}, Error: {response.text}")


# Example usage:
# python create_gene_embeddings --download_tsv_file --tsv_file="./homo_sapiens.tsv" --embeddings_path="./../../ctx_full_esm2_3b_all_embs_unwrapped"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gene embeddings from protein embeddings.")
    parser.add_argument("--tsv_file", type=str, required=True, help="Path to a TSV file, probably from UniProtKB, "
                                                                      "should contain 'Gene Names' and 'Entry' "
                                                                      "columns.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Directory containing protein embeddings.")
    parser.add_argument("--output_file", type=str, default="gene_vector_averages.pt", help="Path to save the gene embeddings.")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity.")
    parser.add_argument("--download_tsv_file", action="store_true", help="Download the TSV file from UniProtKB.")
    args = parser.parse_args()
    if args.download_tsv_file:
        fetch_uniprot_data(args.tsv_file)
    gene_protein_map = create_gene_protein_map(args.tsv_file)
    gene_vector_averages = create_gene_embeddings(gene_protein_map, args.embeddings_path, args.verbose)
    torch.save(gene_vector_averages, args.output_file)
