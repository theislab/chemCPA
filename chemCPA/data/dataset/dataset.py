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
    drugs_idx: torch.Tensor  # Stores the integer index/indices of the drugs applied to each cell
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
        :param covariate_keys: Names of obs columns which store covariate names (e.g., cell type).
        :param perturbation_key: Name of obs column which stores perturbation name (e.g., drug name).
            Combinations of perturbations are separated with `+`.
        :param dose_key: Name of obs column which stores perturbation dose. Also separated with `+`.
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

        data_X = data_dict["X"]
        var_names = data_dict["var_names"]
        obs = data_dict["obs"]
        uns = data_dict["uns"]

        # Decode var names if they are bytes
        self.var_names = pd.Series([name.decode("utf-8") if isinstance(name, bytes) else name for name in var_names])

        # Convert X to a numpy array if needed
        if isinstance(data_X, csr_matrix):
            data_X = data_X.toarray()
        else:
            data_X = np.array(data_X)

        # Store gene expression as a tensor
        self.genes = torch.Tensor(data_X)

        if perturbation_key is not None:
            print("PERTURBATION START", perturbation_key)
            start_time = time.time()

            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key' ({perturbation_key})."
                )

            self.pert_categories = np.array(obs[self.pert_category])  # e.g., "cellType_drug_dose"
            self.de_genes = uns[self.degs_key]

            # Decode byte strings in the dict of DE genes
            de_genes_decoded = {}
            for key, genes_ in self.de_genes.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                genes_ = np.array([g.decode("utf-8") if isinstance(g, bytes) else g for g in genes_])
                de_genes_decoded[key] = genes_
            self.de_genes = de_genes_decoded

            self.drugs_names = np.array(obs[perturbation_key])  # e.g. "DrugA+DrugB"
            self.dose_names = np.array(obs[dose_key])           # e.g. "0.1+0.5"

            # Convert each dose string into a list of floats
            self.dose_values = [
                [float(d) for d in str(dose_str).split("+")] for dose_str in self.dose_names
            ]

            # Optionally, compute total dosage per sample if needed
            self.total_dose = np.array([sum(dose_list) for dose_list in self.dose_values])

            # Collect unique drug names
            drugs_names_unique = set()
            print(self.drugs_names)
            for d in self.drugs_names:
                [drugs_names_unique.add(x) for x in d.split("+")]

            self.drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
            self._drugs_name_to_idx = {
                drug: idx for idx, drug in enumerate(self.drugs_names_unique_sorted)
            }

            # Initialize canonical SMILES strings
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), obs, perturbation_key, smiles_key
            )

            # How many maximum drugs per cell
            self.max_num_perturbations = max(len(name.split("+")) for name in self.drugs_names)

            if not use_drugs_idx:
                # ------------------------------ One-Hot + dosage approach -----------------------------
                self.encoder_drug = OneHotEncoder(
                    sparse_output=False, categories=[self.drugs_names_unique_sorted]
                )
                self.encoder_drug.fit(self.drugs_names_unique_sorted.reshape(-1, 1))

                # Map each unique drug to its one-hot vector
                self.atomic_drugs_dict = dict(
                    zip(
                        self.drugs_names_unique_sorted,
                        self.encoder_drug.transform(
                            self.drugs_names_unique_sorted.reshape(-1, 1)
                        ),
                    )
                )

                # Build array: for each row, sum up the one-hot encodings times dosage
                all_drugs_enc = []
                for i, comb in enumerate(self.drugs_names):
                    drug_list = comb.split("+")
                    dose_list = self.dose_names[i].split("+")
                    # Sum one-hot for each drug * dose
                    combo_encoding = None
                    for j, drug_j in enumerate(drug_list):
                        dose_j = float(dose_list[j])
                        oh_j = self.atomic_drugs_dict[drug_j]
                        if combo_encoding is None:
                            combo_encoding = dose_j * oh_j
                        else:
                            combo_encoding += dose_j * oh_j
                    all_drugs_enc.append(combo_encoding)
                self.drugs = torch.Tensor(np.array(all_drugs_enc))

                # Store a mapping from integer → drug_name (for debugging)
                self.drug_dict = {}
                atomic_ohe = self.encoder_drug.transform(
                    self.drugs_names_unique_sorted.reshape(-1, 1)
                )
                for idrug, drug in enumerate(self.drugs_names_unique_sorted):
                    i = np.where(atomic_ohe[idrug] == 1)[0][0]
                    self.drug_dict[i] = drug

            else:

                list_of_idx = []
                list_of_dos = []
                for i, comb_str in enumerate(self.drugs_names):
                    # e.g. "DrugA+DrugB+DrugC"
                    drug_combo = comb_str.split("+")
                    dose_combo = str(self.dose_names[i]).split("+")

                    # Convert to integer indices and floats
                    these_idx = [self.drug_name_to_idx(dname) for dname in drug_combo]
                    these_dos = [float(d) for d in dose_combo]

                    list_of_idx.append(these_idx)
                    list_of_dos.append(these_dos)

                N = len(list_of_idx)
                M = self.max_num_perturbations

                # Create zero-filled arrays
                drugs_idx_arr = np.zeros((N, M), dtype=np.int64)
                dosages_arr = np.zeros((N, M), dtype=np.float32)

                for i in range(N):
                    combo_indices = list_of_idx[i]
                    combo_doses = list_of_dos[i]
                    for j, (drug_j, dose_j) in enumerate(zip(combo_indices, combo_doses)):
                        drugs_idx_arr[i, j] = drug_j
                        dosages_arr[i, j] = dose_j

                self.drugs_idx = torch.from_numpy(drugs_idx_arr)
                self.dosages = torch.from_numpy(dosages_arr)

            print("PERTURBATION END time:", time.time() - start_time)

        else:
            # No perturbation key => control data or something else
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.atomic_drugs_dict = None
            self.drug_dict = None
            self.drugs = None
            self.canon_smiles_unique_sorted = None

        # Handle covariates if provided
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

                self.atomic_covars_dict[cov] = dict(
                    zip(list(names), encoder_cov.transform(names.reshape(-1, 1)))
                )

                # For each cell in obs, store the one-hot
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(self.covariate_names[cov].reshape(-1, 1))).float()
                )
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.atomic_covars_dict = None
            self.covariates = None

        # Control vs. treated
        self.ctrl = obs["control"]
        if perturbation_key is not None:
            self.ctrl_name = list(np.unique(self.drugs_names[self.ctrl == 1]))
        else:
            self.ctrl_name = None

        # Number of covariates
        if self.covariates is not None:
            self.num_covariates = [len(names) for names in self.covariate_names_unique.values()]
        else:
            self.num_covariates = [0]

        # Genes, drugs, etc.
        self.num_genes = self.genes.shape[1]
        self.num_drugs = (
            len(self.drugs_names_unique_sorted) if self.drugs_names_unique_sorted is not None else 0
        )

        # Build train/test/ood indices
        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(self.ctrl == 1)[0].tolist(),
            "treated": np.where(self.ctrl != 1)[0].tolist(),
            "train": np.where(obs[self.split_key] == "train")[0].tolist(),
            "test": np.where(obs[self.split_key] == "test")[0].tolist(),
            "ood": np.where(obs[self.split_key] == "ood")[0].tolist(),
        }

        # DEGs
        self.degs = compute_degs(
            self.drugs_names,
            self.covariate_names[self.covariate_keys[0]] if self.covariate_keys else None,
            self.dose_names,
            self.de_genes,
            self.var_names,
        )

    def subset(self, split, condition="all", dosage_range=None, dosage_filter=None):
        # Intersect the specified split with control/treated
        idx = list(set(self.indices[split]) & set(self.indices[condition]))

        # Filter based on dosage_range if provided
        if dosage_range is not None:
            min_dose, max_dose = dosage_range
            idx = [
                i
                for i in idx
                if all(min_dose <= dose <= max_dose for dose in self.dose_values[i])
            ]

        # Apply a custom dosage_filter function if provided
        if dosage_filter is not None:
            idx = [i for i in idx if dosage_filter(self.dose_values[i])]

        return SubDataset(self, idx)

    def drug_name_to_idx(self, drug_name: str) -> int:
        return self._drugs_name_to_idx[drug_name]

    def __getitem__(self, i):
        """
        For each sample index i:
         - genes[i] => the expression profile
         - if use_drugs_idx=False => return (genes, one-hot combo, degs, covariates…)
         - if use_drugs_idx=True  => return (genes, drug_idx array, dosage array, degs, covariates…)
        """
        if self.use_drugs_idx:
            return (
                self.genes[i],
                indx(self.drugs_idx, i),  # shape (max_num_perturbations,)
                indx(self.dosages, i),    # shape (max_num_perturbations,)
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

    def debug_print(self, num_entries: int = 5):
        print("=== Dataset Debug Info ===")
        print(f"Genes shape: {self.genes.shape}")
        # Check Genes
        n_genes_nans = torch.isnan(self.genes).sum().item()
        if n_genes_nans > 0:
            print(f"WARNING: Genes contain {n_genes_nans} NaNs!")
        print(f"Genes (first {num_entries} rows):\n{self.genes[:num_entries]}")

        if self.use_drugs_idx:
            # drugs_idx / dosages
            print(f"drugs_idx shape: {self.drugs_idx.shape}")
            n_idx_nans = torch.isnan(self.drugs_idx.float()).sum().item()
            if n_idx_nans > 0:
                print(f"WARNING: drugs_idx has {n_idx_nans} NaNs!")

            # Range check for integer indices
            bad_idx = (self.drugs_idx < 0) | (self.drugs_idx >= self.num_drugs)
            if bad_idx.any():
                print("WARNING: Some drugs_idx values are out of range [0..num_drugs-1]!")
                print("First few invalid indices:\n", self.drugs_idx[bad_idx][:10])

            print(f"First {num_entries} entries of drugs_idx:\n{self.drugs_idx[:num_entries]}")

            print(f"dosages shape: {self.dosages.shape}")
            n_dosage_nans = torch.isnan(self.dosages).sum().item()
            if n_dosage_nans > 0:
                print(f"WARNING: dosages contain {n_dosage_nans} NaNs!")
            print(f"First {num_entries} entries of dosages:\n{self.dosages[:num_entries]}")

        else:
            if self.drugs is not None:
                print(f"drugs (one-hot) shape: {self.drugs.shape}")
                n_drugs_nans = torch.isnan(self.drugs).sum().item()
                if n_drugs_nans > 0:
                    print(f"WARNING: One-hot drugs contain {n_drugs_nans} NaNs!")
                print(f"First {num_entries} entries of drugs:\n{self.drugs[:num_entries]}")

        if self.covariates is not None:
            for idx, cov in enumerate(self.covariates):
                print(f"Covariate {idx} shape: {cov.shape}")
                n_cov_nans = torch.isnan(cov).sum().item()
                if n_cov_nans > 0:
                    print(f"WARNING: covariate {idx} contains {n_cov_nans} NaNs!")
                print(f"First {num_entries} rows:\n{cov[:num_entries]}")

        print(f"Number of genes: {self.num_genes}")
        print(f"Number of drugs: {self.num_drugs}")
        print(f"Indices dict keys: {list(self.indices.keys())}")
        print("=========================================\n")
 


    def __len__(self):
        return len(self.genes)


