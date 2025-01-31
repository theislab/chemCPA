import logging
import warnings
from typing import List, Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix
import hydra
import h5py
from tqdm import tqdm
import os

from chemCPA.helper import canonicalize_smiles
from chemCPA.data.perturbation_data_module import PerturbationDataModule
from chemCPA.data.dataset.dataset import Dataset, SubDataset

warnings.simplefilter(action="ignore", category=FutureWarning)

indx = lambda a, i: a[i] if a is not None else None

def safe_decode_categorical(group):
        try:
            categories = group['categories'][:]
            categories = [c.decode('utf-8') if isinstance(c, bytes) else c for c in categories]
            codes = group['codes'][:]
            return np.array([categories[code] for code in codes])
        except KeyError:
            # If 'categories' doesn't exist, return the data as is
            return group[()]


def load_smiles(obs_group: h5py.Group, smiles_key: str, verbose: bool = False):
    """
    Load SMILES data from an HDF5 obs group.
    
    Args:
        obs_group: h5py.Group containing observation data
        smiles_key: Key for SMILES data in obs group
        verbose: Whether to print debug information
    """
    try:
        if smiles_key in obs_group:
            smiles_dataset = obs_group[smiles_key]
            if isinstance(smiles_dataset, h5py.Dataset):
                smiles_data = smiles_dataset[:]
                if smiles_data.dtype.kind == 'S':
                    smiles_data = np.array([x.decode('utf-8') for x in smiles_data])
            elif isinstance(smiles_dataset, h5py.Group):
                smiles_data = safe_decode_categorical(smiles_dataset)
            else:
                raise TypeError(f"'{smiles_key}' in 'obs' group is neither a dataset nor a recognizable group.")
            if verbose:
                print(f"Loaded {len(smiles_data)} SMILES strings.")
            return smiles_data
        else:
            print(obs_group.keys());
            raise KeyError(f"'{smiles_key}' not found in 'obs' group.")
    except Exception as e:
        print(f"An error occurred while loading SMILES data: {e}")
        return None


def load_uns_data(uns_group, degs_key: str, verbose: bool = False) -> dict:
    """
    Load unstructured data from the uns group of an H5 file.
    
    Args:
        uns_group: h5py.Group containing unstructured data
        degs_key: Key for differential expression genes data
        verbose: Whether to print debug information
    
    Returns:
        Dictionary containing the loaded uns data
    """
    verbose = True;
    if verbose:
        print("\nuns_group keys:", list(uns_group.keys()))

    if degs_key not in uns_group:
        if verbose:
            print(f"\nWarning: '{degs_key}' not found in 'uns' group.")
            print(f"Available keys in 'uns' group: {list(uns_group.keys())}")
        raise KeyError(f"'{degs_key}' not found in 'uns' group.")

    degs_data = {}
    degs_group = uns_group[degs_key]
    for key in degs_group:
        if isinstance(degs_group[key], h5py.Dataset):
            degs = degs_group[key][:]
            if degs.dtype.kind == 'S':
                degs = [x.decode('utf-8') for x in degs]
            degs_data[key] = degs
    
    return {degs_key: degs_data}


def load_obs_data(obs_group, obs_keys: List[str], verbose: bool = False) -> dict:
    """
    Load observation data from the obs group of an H5 file.
    
    Args:
        obs_group: h5py.Group containing observation data
        obs_keys: List of keys to load from the obs group
        verbose: Whether to print debug information
    
    Returns:
        Dictionary containing the loaded observation data
    """
    if verbose:
        print("obs_group keys:", list(obs_group.keys()))

    obs_dict = {}
    for key in obs_keys:
        if key and key in obs_group:
            if isinstance(obs_group[key], h5py.Dataset):
                data = obs_group[key][:]
                if data.dtype.kind == 'S':
                    data = np.array([x.decode('utf-8') for x in data])
                obs_dict[key] = data
            elif isinstance(obs_group[key], h5py.Group):
                obs_dict[key] = safe_decode_categorical(obs_group[key])
            else:
                raise TypeError(f"'{key}' in 'obs' group is neither a dataset nor a recognizable group.")
        elif key:
            available_keys = list(obs_group.keys())
            raise KeyError(f"'{key}' not found in 'obs' group. Available keys: {available_keys}")
    
    if verbose:
        print(f"Loaded obs keys: {list(obs_dict.keys())}")
    
    return obs_dict


def load_var_data(var_group, verbose: bool = False) -> np.ndarray:
    """
    Load variable names data from the var group of an H5 file.
    
    Args:
        var_group: h5py.Group containing variable names data
        verbose: Whether to print debug information
    
    Returns:
        numpy array containing the variable names
    """
    if verbose:
        print("var_group keys:", list(var_group.keys()))

    if 'index' in var_group and isinstance(var_group['index'], h5py.Dataset):
        var_names = var_group['index'][:]
    elif '_index' in var_group and isinstance(var_group['_index'], h5py.Dataset):
        var_names = var_group['_index'][:]
    else:           
        raise KeyError("Neither 'index' nor '_index' found or is not a dataset in 'var' group.")

    var_names = [v.decode('utf-8') if isinstance(v, bytes) else v for v in var_names]
    var_names = np.array(var_names)
    
    if verbose:
        print(f"Number of var_names: {len(var_names)}")
    
    return var_names


def load_x_data(x_group, verbose: bool = False):
    """
    Load X data from an HDF5 group.
    
    Args:
        x_group: h5py.Group or Dataset containing X data
        verbose: Whether to print debug information
    
    Returns:
        The loaded X data as either a csr_matrix or numpy array
    """
    if verbose:
        print("X group type:", type(x_group))
    
    if isinstance(x_group, h5py.Group):
        if 'data' in x_group:
            data = x_group['data'][:]
            indices = x_group['indices'][:]
            indptr = x_group['indptr'][:]
            shape = x_group.attrs['shape']
            x_data = csr_matrix((data, indices, indptr), shape=shape)
        else:
            x_data = x_group[:]
    elif isinstance(x_group, h5py.Dataset):
        x_data = x_group[:]
    else:
        raise TypeError("'X' is neither a group nor a dataset.")
    
    if verbose:
        print(f"X shape: {x_data.shape}")
    
    return x_data


def load_data(
    data_path: str,
    perturbation_key: Optional[str] = None,
    dose_key: Optional[str] = None,
    covariate_keys: Optional[Union[List[str], str]] = None,
    smiles_key: Optional[str] = None,
    degs_key: str = "rank_genes_groups_cov",
    pert_category: str = "cov_drug_dose_name",
    split_key: str = "split",
    verbose: bool = True,
): 
    if verbose:
        logging.info(f"Loading data from {data_path}")
    
    try:
        with h5py.File(data_path, 'r') as file:
            if verbose:
                print(f"Loading data from {data_path}")
                print(f"File structure: {list(file.keys())}")

            # Prepare obs_keys for obs loader
            obs_keys = [perturbation_key, dose_key, pert_category, smiles_key, 'control', split_key] + \
                      ([covariate_keys] if isinstance(covariate_keys, str) else (covariate_keys or []))

            # Define loader functions with descriptions
            loaders = [
                {
                    'name': 'Loading expression matrix (X)',
                    'key': 'X',
                    'condition': lambda: 'X' in file,
                    'loader': lambda f: {'X': load_x_data(f['X'], verbose)}
                },
                {
                    'name': 'Loading gene names (var)',
                    'key': 'var',
                    'condition': lambda: 'var' in file,
                    'loader': lambda f: {'var_names': load_var_data(f['var'], verbose)}
                },
                {
                    'name': 'Loading metadata (obs)',
                    'key': 'obs',
                    'condition': lambda: 'obs' in file,
                    'loader': lambda f: {'obs': load_obs_data(f['obs'], obs_keys, verbose)}
                },
                {
                    'name': 'Loading unstructured data (uns)',
                    'key': 'uns',
                    'condition': lambda: 'uns' in file,
                    'loader': lambda f: {'uns': load_uns_data(f['uns'], degs_key, verbose)}
                },
                {
                    'name': 'Loading SMILES data',
                    'condition': lambda: smiles_key is not None and 'obs' in file,
                    'loader': lambda f: {'smiles': load_smiles(f['obs'], smiles_key, verbose)}
                }
            ]

            data_dict = {}
            # Filter out disabled loaders
            active_loaders = [loader for loader in loaders if loader['condition']()]
            
            # Create progress bar
            pbar = tqdm(active_loaders, desc="Loading dataset components")
            
            for loader in pbar:
                pbar.set_description(f"Processing {loader['name']}")
                try:
                    result = loader['loader'](file)
                    if result:
                        data_dict = data_dict | result
                except Exception as e:
                    logging.error(f"Error in {loader['name']}: {str(e)}")
                    raise

    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise

    return data_dict


def load_dataset_splits(
    dataset_path: str = None,
    perturbation_key: Union[str, None] = None,
    dose_key: Union[str, None] = None,
    covariate_keys: Union[list, str, None] = None,
    smiles_key: Union[str, None] = None,
    degs_key: str = "rank_genes_groups_cov",
    pert_category: str = "cov_drug_dose_name",
    split_key: str = "split",
    return_dataset: bool = False,
    use_drugs_idx=False,
    verbose: bool = False,
):
    data_dict = load_data(
        dataset_path,
        perturbation_key=perturbation_key,
        dose_key=dose_key,
        covariate_keys=covariate_keys,
        smiles_key=smiles_key,
        degs_key=degs_key,
        pert_category=pert_category,
        split_key=split_key,
        verbose=verbose,
    )
    dataset = Dataset(
        data_dict,
        perturbation_key=perturbation_key,
        dose_key=dose_key,
        covariate_keys=covariate_keys,
        smiles_key=smiles_key,
        degs_key=degs_key,
        pert_category=pert_category,
        split_key=split_key,
        use_drugs_idx=use_drugs_idx,
    )

    splits = {
        "training": dataset.subset("train", "all"),
        "training_control": dataset.subset("train", "control"),
        "training_treated": dataset.subset("train", "treated"),
        "test": dataset.subset("test", "all"),
        "test_control": dataset.subset("test", "control"),
        "test_treated": dataset.subset("test", "treated"),
        "ood": dataset.subset("ood", "all"),
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits

