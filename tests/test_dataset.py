import numpy
import torch.testing

from chemCPA.data.data import Dataset


def test_dataset_idx_ohe():
    kwargs = {
        "perturbation_key": "condition",
        "pert_category": "cov_drug_dose_name",
        "dose_key": "dose",
        "covariate_keys": "cell_type",
        "smiles_key": "SMILES",
        "split_key": "split",
    }
    d_idx = Dataset(
        fname="datasets/trapnell_cpa_subset.h5ad",
        **kwargs,
        use_drugs_idx=True,
    )

    d_ohe = Dataset(
        fname="datasets/trapnell_cpa_subset.h5ad",
        **kwargs,
        use_drugs_idx=False,
    )

    numpy.testing.assert_equal(
        d_ohe.encoder_drug.categories_[0], d_idx.drugs_names_unique_sorted
    )

    for i in range(len(d_idx)):
        genes_idx, idx, dosage, cov_idx = d_idx[i]
        genes_ohe, drug, cov_ohe = d_ohe[i]
        torch.testing.assert_close(genes_idx, genes_ohe)
        # make sure the OHE and the index representation encode the same info
        torch.testing.assert_close(drug[idx], dosage)
        torch.testing.assert_close(cov_idx, cov_ohe)
