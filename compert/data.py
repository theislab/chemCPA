# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import warnings

import numpy as np
import torch

warnings.simplefilter(action="ignore", category=FutureWarning)
from typing import List, Optional, Union

import pandas as pd
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder

from compert.helper import canonicalize_smiles


def ranks_to_df(data, key="rank_genes_groups"):
    """Converts an `sc.tl.rank_genes_groups` result into a MultiIndex dataframe.

    You can access various levels of the MultiIndex with `df.loc[[category]]`.

    Params
    ------
    data : `AnnData`
    key : str (default: 'rank_genes_groups')
        Field in `.uns` of data where `sc.tl.rank_genes_groups` result is
        stored.
    """
    d = data.uns[key]
    dfs = []
    for k in d.keys():
        if k == "params":
            continue
        series = pd.DataFrame.from_records(d[k]).unstack()
        series.name = k
        dfs.append(series)

    return pd.concat(dfs, axis=1)


def drug_names_to_once_canon_smiles(
    drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
):
    """
    Converts a list of drug names to a list of SMILES. The ordering is of the list is preserved.

    TODO: This function will need to be rewritten to handle datasets with combinations.
    This is not difficult to do, mainly we need to standardize how combinations of SMILES are stored in anndata.
    """
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in dataset.obs.groupby(
            [perturbation_key, smiles_key]
        ).groups.keys()
    }
    return [name_to_smiles_map[name] for name in drug_names]


indx = lambda a, i: a[i] if a is not None else None


class Dataset:
    covariate_keys: Optional[List[str]]
    drugs: torch.Tensor  # stores the (OneHot * dosage) encoding of drugs / combinations of drugs
    drugs_idx: torch.Tensor  # stores the integer index of the drugs applied to each cell.
    max_num_perturbations: int  # how many drugs are applied to each cell at the same time?
    dosages: torch.Tensor  # shape: (dataset_size, max_num_perturbations)
    drugs_names_unique_sorted: np.ndarray  # sorted list of all drug names in the dataset

    def __init__(
        self,
        fname: str,
        perturbation_key=None,
        dose_key=None,
        covariate_keys=None,
        smiles_key=None,
        pert_category="cov_drug_dose_name",
        split_key="split",
        use_drugs_idx=False,
    ):
        """
        :param covariate_keys: Names of obs columns which stores covariate names (eg cell type).
        :param perturbation_key: Name of obs column which stores perturbation name (eg drug name).
            Combinations of perturbations are separated with `+`.
        :param dose_key: Name of obs column which stores perturbation dose.
            Combinations of perturbations are separated with `+`.
        :param pert_category: Name of obs column with stores covariate + perturbation + dose.
            Example: cell type + drug name + drug dose. This seems unused?
        :param use_drugs_idx: Whether or not to encode drugs via their index, instead of via a OneHot encoding
        """
        logging.info(f"Starting to read in data: {fname}\n...")
        data = sc.read(fname)
        logging.info(f"Finished data loading.")
        self.genes = torch.Tensor(data.X.A)
        self.var_names = data.var_names

        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys
        self.smiles_key = smiles_key
        self.use_drugs_idx = use_drugs_idx

        if perturbation_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key'({perturbation_key})."
                )
            self.pert_categories = np.array(data.obs[pert_category].values)
            self.de_genes = data.uns["rank_genes_groups_cov"]
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            # get unique drugs
            drugs_names_unique = set()
            for d in self.drugs_names:
                [drugs_names_unique.add(i) for i in d.split("+")]

            self.drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))

            self._drugs_name_to_idx = {
                smiles: idx for idx, smiles in enumerate(self.drugs_names_unique_sorted)
            }
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), data, perturbation_key, smiles_key
            )
            self.max_num_perturbations = max(
                len(name.split("+")) for name in self.drugs_names
            )

            if not use_drugs_idx:
                # prepare a OneHot encoding for each unique drug in the dataset
                # use the same sorted ordering of drugs as for indexing
                self.encoder_drug = OneHotEncoder(
                    sparse=False, categories=[list(self.drugs_names_unique_sorted)]
                )
                self.encoder_drug.fit(self.drugs_names_unique_sorted.reshape(-1, 1))
                # stores a drug name -> OHE mapping (np float array)
                self.atomic_drugs_dict = dict(
                    zip(
                        self.drugs_names_unique_sorted,
                        self.encoder_drug.transform(
                            self.drugs_names_unique_sorted.reshape(-1, 1)
                        ),
                    )
                )
                # get drug combination encoding: for each cell we calculate a single vector as:
                # combination_encoding = dose1 * OneHot(drug1) + dose2  * OneHot(drug2) + ...
                drugs = []
                for i, comb in enumerate(self.drugs_names):
                    # here (in encoder_drug.transform()) is where the init_dataset function spends most of it's time.
                    drugs_combos = self.encoder_drug.transform(
                        np.array(comb.split("+")).reshape(-1, 1)
                    )
                    dose_combos = str(data.obs[dose_key].values[i]).split("+")
                    for j, d in enumerate(dose_combos):
                        if j == 0:
                            drug_ohe = float(d) * drugs_combos[j]
                        else:
                            drug_ohe += float(d) * drugs_combos[j]
                    drugs.append(drug_ohe)
                self.drugs = torch.Tensor(np.array(drugs))

                # store a mapping from int -> drug_name, where the integer equals the position
                # of the drug in the OneHot encoding. Very convoluted, should be refactored.
                self.drug_dict = {}
                atomic_ohe = self.encoder_drug.transform(
                    self.drugs_names_unique_sorted.reshape(-1, 1)
                )
                for idrug, drug in enumerate(self.drugs_names_unique_sorted):
                    i = np.where(atomic_ohe[idrug] == 1)[0][0]
                    self.drug_dict[i] = drug
            else:
                assert (
                    self.max_num_perturbations == 1
                ), "Index-based drug encoding only works with single perturbations"
                drugs_idx = [self.drug_name_to_idx(drug) for drug in self.drugs_names]
                self.drugs_idx = torch.tensor(
                    drugs_idx,
                    dtype=torch.long,
                )
                dosages = [float(dosage) for dosage in self.dose_names]
                self.dosages = torch.tensor(
                    dosages,
                    dtype=torch.float32,
                )

        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.atomic_drugs_dict = None
            self.drug_dict = None
            self.drugs = None

        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self.atomic_сovars_dict = {}
            self.covariates = []
            for cov in covariate_keys:
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names = self.covariate_names_unique[cov]
                encoder_cov = OneHotEncoder(sparse=False)
                encoder_cov.fit(names.reshape(-1, 1))

                self.atomic_сovars_dict[cov] = dict(
                    zip(list(names), encoder_cov.transform(names.reshape(-1, 1)))
                )

                names = self.covariate_names[cov]
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float()
                )
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.atomic_сovars_dict = None
            self.covariates = None

        self.ctrl = data.obs["control"].values

        if perturbation_key is not None:
            self.ctrl_name = list(
                np.unique(data[data.obs["control"] == 1].obs[self.perturbation_key])
            )
        else:
            self.ctrl_name = None

        if self.covariates is not None:
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.num_covariates = [0]
        self.num_genes = self.genes.shape[1]
        self.num_drugs = (
            len(self.drugs_names_unique_sorted)
            if self.drugs_names_unique_sorted is not None
            else 0
        )

        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs["control"] == 1)[0].tolist(),
            "treated": np.where(data.obs["control"] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def drug_name_to_idx(self, drug_name: str):
        """
        For the given drug, return it's index. The index will be persistent for each dataset (since the list is sorted).
        Raises ValueError if the drug doesn't exist in the dataset.
        """
        return self._drugs_name_to_idx[drug_name]

    def __getitem__(self, i):
        if self.use_drugs_idx:
            return (
                self.genes[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                *[indx(cov, i) for cov in self.covariates],
            )
        else:
            return (
                self.genes[i],
                indx(self.drugs, i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset: Dataset, indices):
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.smiles_key = dataset.smiles_key

        self.covars_dict = dataset.atomic_сovars_dict

        self.genes = dataset.genes[indices]
        self.use_drugs_idx = dataset.use_drugs_idx
        if self.use_drugs_idx:
            self.drugs_idx = indx(dataset.drugs_idx, indices)
            self.dosages = indx(dataset.dosages, indices)
        else:
            self.perts_dict = dataset.atomic_drugs_dict
            self.drugs = indx(dataset.drugs, indices)
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = indx(dataset.ctrl_name, 0)

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

    def __getitem__(self, i):
        if self.use_drugs_idx:
            return (
                self.genes[i],
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                *[indx(cov, i) for cov in self.covariates],
            )
        else:
            return (
                self.genes[i],
                indx(self.drugs, i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
    dataset_path: str,
    perturbation_key: Union[str, None],
    dose_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    smiles_key: Union[str, None],
    pert_category: str = "cov_drug_dose_name",
    split_key: str = "split",
    return_dataset: bool = False,
    use_drugs_idx=False,
):

    dataset = Dataset(
        dataset_path,
        perturbation_key,
        dose_key,
        covariate_keys,
        smiles_key,
        pert_category,
        split_key,
        use_drugs_idx,
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
