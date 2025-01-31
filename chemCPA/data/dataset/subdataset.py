from typing import List
import numpy as np

indx = lambda a, i: a[i] if a is not None else None

class SubDataset:
    """
    Subsets a `Dataset` or another `SubDataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        # Store a reference to the parent dataset (could be Dataset or SubDataset)
        self.dataset = dataset
        self.use_drugs_idx = dataset.use_drugs_idx

        # Map indices to the original dataset if necessary
        if isinstance(dataset, SubDataset):
            # Map original indices to keep track for future subsetting
            self.original_indices = [dataset.original_indices[i] for i in indices]
        else:
            self.original_indices = indices

        # Access data using indices directly from the parent dataset
        self.genes = dataset.genes[indices]

        if self.use_drugs_idx:
            self.drugs_idx = indx(dataset.drugs_idx, indices)
            self.dosages = indx(dataset.dosages, indices)
        else:
            self.drugs = indx(dataset.drugs, indices)

        # Retrieve dosage values from the parent dataset
        self.dose_values = [dataset.dose_values[i] for i in indices]

        # Proceed with other attributes
        self.degs = indx(dataset.degs, indices)
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_keys = dataset.covariate_keys
        self.covariate_names = {}
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = dataset.ctrl_name

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

        # **Added missing attributes**
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.smiles_key = dataset.smiles_key
        self.degs_key = dataset.degs_key
        self.pert_category = dataset.pert_category
        self.split_key = dataset.split_key

    def subset(self, dosage_range=None, dosage_filter=None):
        """
        Creates a new SubDataset by filtering the current SubDataset based on dosage criteria.

        Parameters:
            dosage_range (tuple): A tuple (min_dose, max_dose) to filter samples where all dosages fall within this range.
            dosage_filter (callable): A function that takes a list of dosages and returns True if the sample should be included.

        Returns:
            SubDataset: A new SubDataset instance with the filtered data.
        """
        idx = list(range(len(self.genes)))

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
