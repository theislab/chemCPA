import numpy
import pandas as pd
import torch.testing

from compert.data import Dataset
from compert.embedding import get_chemical_representation
from compert.model import ComPert


def test_embedding_idx_roundtrip():
    # test to make sure that the same drug embeddings are computed for all drugs
    # in trapnell_subset, independent of whether we use indices or one-hot-encodings
    kwargs = {
        "perturbation_key": "condition",
        "pert_category": "cov_drug_dose_name",
        "dose_key": "dose",
        "covariate_keys": "cell_type",
        "smiles_key": "SMILES",
        "mol_featurizer": "canonical",
        "split_key": "split",
    }

    # load the embedding of DSMO
    control_emb = torch.tensor(
        pd.read_parquet("embeddings/grover/data/embeddings/grover_base.parquet")
        .loc["CS(C)=O"]
        .values
    )

    for use_drugs_idx in [True, False]:
        dataset = Dataset(
            fname="datasets/trapnell_cpa_subset.h5ad",
            **kwargs,
            use_drugs_idx=use_drugs_idx
        )
        embedding = get_chemical_representation(
            smiles=dataset.smiles_unique_sorted,
            embedding_model="grover_base",
        )
        device = embedding.weight.device

        # make sure "control" is correctly encoded as the all zero vector
        control = torch.tensor(
            list(dataset.drugs_names_unique_sorted).index("control"),
            device=device,
        )
        torch.testing.assert_equal(embedding(control), control_emb.to(device))

        model = ComPert(
            dataset.num_genes,
            dataset.num_drugs,
            dataset.num_covariates,
            device=device,
            doser_type="sigm",
            drug_embeddings=embedding,
            use_drugs_idx=use_drugs_idx,
        )
        if use_drugs_idx:
            genes, idx, dosages, covariates = dataset[:]
            idx_emb = model.compute_drug_embeddings_(drugs_idx=idx, dosages=dosages)
        else:
            genes, drugs, covariates = dataset[:]
            ohe_emb = model.compute_drug_embeddings_(drugs=drugs)

    # assert both model return the same embedding for the drugs in the dataset
    torch.testing.assert_close(idx_emb, ohe_emb)
