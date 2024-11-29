import logging
import time
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
import h5py
from tqdm import tqdm

from chemCPA.helper import canonicalize_smiles
from .subdataset import SubDataset
from .compute_degs import compute_degs
from .drug_names_to_once_canon_smiles import drug_names_to_once_canon_smiles

warnings.simplefilter(action="ignore", category=FutureWarning)

indx = lambda a, i: a[i] if a is not None else None

class Dataset:
    covariate_keys: Optional[List[str]]
    drugs: torch.Tensor  # Stores the (OneHot * dosage) encoding of drugs / combinations of drugs
    drugs_idx: torch.Tensor  # Stores the integer index of the drugs applied to each cell
    max_num_perturbations: int  # How many drugs are applied to each cell at the same time?
    dosages: torch.Tensor  # Shape: (dataset_size, max_num_perturbations)
    drugs_names_unique_sorted: np.ndarray  # Sorted list of all drug names in the dataset
    canon_smiles_unique_sorted: List[str]  # List of canonical SMILES strings corresponding to drugs

    def __init__(
        self,
        data_dict,
        perturbation_key=None,
        dose_key=None,
        covariate_keys=None,
        smiles_key=None,
        degs_key="rank_genes_groups_cov",
        pert_category="cov_drug_dose_name",
        split_key="split",
        use_drugs_idx=False,
    ):
        """
        :param data_dict: Dictionary containing the data loaded from the h5ad file.
        :param covariate_keys: Names of obs columns which stores covariate names (e.g., cell type).
        :param perturbation_key: Name of obs column which stores perturbation name (e.g., drug name).
            Combinations of perturbations are separated with `+`.
        :param dose_key: Name of obs column which sGtores perturbation dose.
            Combinations of perturbations are separated with `+`.
        :param pert_category: Name of obs column that stores covariate + perturbation + dose as one string.
            Example: cell type + drug name + drug dose. This is used during evaluation.
        :param use_drugs_idx: Whether or not to encode drugs via their index, instead of via a OneHot encoding
        """
        self.use_drugs_idx = use_drugs_idx
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        self.covariate_keys = covariate_keys if covariate_keys else []
        self.smiles_key = smiles_key
        self.degs_key = degs_key
        self.pert_category = pert_category
        self.split_key = split_key

        if isinstance(covariate_keys, str):
            self.covariate_keys = [covariate_keys]
        elif covariate_keys:
            self.covariate_keys = covariate_keys
        else:
            self.covariate_keys = []

        data_X = data_dict['X']
        var_names = data_dict['var_names']
        obs = data_dict['obs']
        uns = data_dict['uns']

        self.var_names = pd.Series([name.decode('utf-8') if isinstance(name, bytes) else name for name in var_names])

        if isinstance(data_X, csr_matrix):
            data_X = data_X.toarray()
        else:
            data_X = np.array(data_X)

        self.genes = torch.Tensor(data_X)

        if perturbation_key is not None:
            print("PERTURBATION START", perturbation_key)
            start_time = time.time()
            if dose_key is None:
                raise ValueError(f"A 'dose_key' is required when provided a 'perturbation_key' ({perturbation_key}).")
            self.pert_categories = np.array(obs[self.pert_category])
            self.de_genes = uns[self.degs_key]

            # Decode byte strings in de_genes keys and values
            de_genes_decoded = {}
            for key, genes in self.de_genes.items():
                # Decode the key if it's a byte string
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                # Decode each gene in the list if it's a byte string and convert to numpy array
                genes = np.array([gene.decode('utf-8') if isinstance(gene, bytes) else gene for gene in genes])
                de_genes_decoded[key] = genes

            self.de_genes = de_genes_decoded

            self.drugs_names = np.array(obs[perturbation_key])
            self.dose_names = np.array(obs[dose_key])

            # Process dose_values to handle combinations and convert to floats
            self.dose_values = [
                [float(d) for d in str(dose_str).split("+")]
                for dose_str in self.dose_names
            ]

            # Optionally, compute total dosage per sample if needed
            self.total_dose = np.array([sum(dose_list) for dose_list in self.dose_values])

            # Get unique drugs
            drugs_names_unique = set()
            for d in self.drugs_names:
                [drugs_names_unique.add(i) for i in d.split("&")]

            self.drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))

            self._drugs_name_to_idx = {drug: idx for idx, drug in enumerate(self.drugs_names_unique_sorted)}

            # Initialize canonical SMILES strings
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), obs, perturbation_key, smiles_key
            )

            self.max_num_perturbations = max(len(name.split("&")) for name in self.drugs_names)

            if not use_drugs_idx:
                # Prepare a OneHot encoding for each unique drug in the dataset
                self.encoder_drug = OneHotEncoder(sparse_output=False, categories=[self.drugs_names_unique_sorted])
                self.encoder_drug.fit(self.drugs_names_unique_sorted.reshape(-1, 1))
                # Stores a drug name -> OHE mapping (np float array)
                self.atomic_drugs_dict = dict(
                    zip(
                        self.drugs_names_unique_sorted,
                        self.encoder_drug.transform(self.drugs_names_unique_sorted.reshape(-1, 1)),
                    )
                )
                # Get drug combination encoding
                drugs = []
                for i, comb in enumerate(self.drugs_names):
                    drugs_combos = self.encoder_drug.transform(np.array(comb.split("+")).reshape(-1, 1))
                    dose_combos = str(self.dose_names[i]).split("+")
                    for j, d in enumerate(dose_combos):
                        if j == 0:
                            drug_ohe = float(d) * drugs_combos[j]
                        else:
                            drug_ohe += float(d) * drugs_combos[j]
                    drugs.append(drug_ohe)
                self.drugs = torch.Tensor(np.array(drugs))

                # Store a mapping from int -> drug_name
                self.drug_dict = {}
                atomic_ohe = self.encoder_drug.transform(self.drugs_names_unique_sorted.reshape(-1, 1))
                for idrug, drug in enumerate(self.drugs_names_unique_sorted):
                    i = np.where(atomic_ohe[idrug] == 1)[0][0]
                    self.drug_dict[i] = drug
            else:
                assert self.max_num_perturbations == 1, "Index-based drug encoding only works with single perturbations"
                drugs_idx = [self.drug_name_to_idx(drug) for drug in self.drugs_names]
                self.drugs_idx = torch.tensor(drugs_idx, dtype=torch.long)
                dosages = [float(dosage) for dosage in self.dose_names]
                self.dosages = torch.tensor(dosages, dtype=torch.float32)
            print("PERTURBATION END time:", time.time() - start_time)
        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.atomic_drugs_dict = None
            self.drug_dict = None
            self.drugs = None
            self.canon_smiles_unique_sorted = None

        if self.covariate_keys:
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self.atomic_covars_dict = {}
            self.covariates = []
            for cov in self.covariate_keys:
                self.covariate_names[cov] = np.array(obs[cov])
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names = self.covariate_names_unique[cov]
                encoder_cov = OneHotEncoder(sparse_output=False)
                encoder_cov.fit(names.reshape(-1, 1))

                self.atomic_covars_dict[cov] = dict(zip(list(names), encoder_cov.transform(names.reshape(-1, 1))))

                names = self.covariate_names[cov]
                self.covariates.append(torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float())       
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.atomic_covars_dict = None
            self.covariates = None

        self.ctrl = obs["control"]

        if perturbation_key is not None:
            self.ctrl_name = list(np.unique(self.drugs_names[self.ctrl == 1]))
        else:
            self.ctrl_name = None

        if self.covariates is not None:
            self.num_covariates = [len(names) for names in self.covariate_names_unique.values()]
        else:
            self.num_covariates = [0]
        self.num_genes = self.genes.shape[1]
        self.num_drugs = len(self.drugs_names_unique_sorted) if self.drugs_names_unique_sorted is not None else 0

        # Build indices
        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(self.ctrl == 1)[0].tolist(),
            "treated": np.where(self.ctrl != 1)[0].tolist(),
            "train": np.where(obs[self.split_key] == "train")[0].tolist(),
            "test": np.where(obs[self.split_key] == "test")[0].tolist(),
            "ood": np.where(obs[self.split_key] == "ood")[0].tolist(),
        }

        self.degs = compute_degs(
            self.drugs_names,
            self.covariate_names[self.covariate_keys[0]] if self.covariate_keys else None,
            self.dose_names,
            self.de_genes,
            self.var_names
        )

    def subset(self, split, condition="all", dosage_range=None, dosage_filter=None):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))

        # Filter based on dosage_range if provided
        if dosage_range is not None:
            min_dose, max_dose = dosage_range
            idx = [
                i for i in idx
                if all(min_dose <= dose <= max_dose for dose in self.dose_values[i])
            ]

        # Apply a custom dosage_filter function if provided
        if dosage_filter is not None:
            idx = [i for i in idx if dosage_filter(self.dose_values[i])]

        return SubDataset(self, idx)

    def drug_name_to_idx(self, drug_name: str):
        return self._drugs_name_to_idx[drug_name]

    def __getitem__(self, i):
        if self.use_drugs_idx:
            return (
                self.genes[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                *[indx(cov, i) for cov in self.covariates],
            )
        else:
            return (
                self.genes[i],
                indx(self.drugs, i),
                indx(self.degs, i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return len(self.genes)

