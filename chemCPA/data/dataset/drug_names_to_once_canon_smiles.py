from typing import List

from chemCPA.helper import canonicalize_smiles


def drug_names_to_once_canon_smiles(
    drug_names: List[str],
    obs: dict,
    perturbation_key: str,
    smiles_key: str
):
    """
    Converts a list of drug names to a list of canonical SMILES strings.
    The ordering of the list is preserved.

    Args:
        drug_names (List[str]): List of drug names.
        obs (dict): Dictionary containing observation data.
        perturbation_key (str): Key for drug names in obs.
        smiles_key (str): Key for SMILES strings in obs.

    Returns:
        List[str]: List of canonical SMILES strings corresponding to the drug_names.
    """
    # Create a mapping from drug name to SMILES
    drug_names_array = obs[perturbation_key]
    smiles_array = obs[smiles_key]

    # Get unique pairs of drug names and SMILES
    unique_pairs = set(zip(drug_names_array, smiles_array))
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in unique_pairs
    }

    return [name_to_smiles_map[name] for name in drug_names]

