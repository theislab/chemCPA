from typing import List
from chemCPA.helper import canonicalize_smiles

def drug_names_to_once_canon_smiles(
    drug_names: List[str],
    obs: dict,
    perturbation_key: str,
    smiles_key: str
):
    """
    For each row in obs, split combination drug names (on '+')
    and combo SMILES (on '..'), then canonicalize each sub-part
    and store them in a dictionary keyed by the *individual* drug name.
    
    That way, if drug_names includes both single and sub-drug names,
    we have an entry for each sub-drug.
    """
    drug_names_array = obs[perturbation_key]
    smiles_array = obs[smiles_key]

    # Build a set of (full_combo_drug_name, full_combo_smiles)
    unique_pairs = set(zip(drug_names_array, smiles_array))
    name_to_smiles_map = {}

    for combo_name, combo_smiles in unique_pairs:
        # If this row doesn't have valid strings, skip
        if not isinstance(combo_name, str) or not isinstance(combo_smiles, str):
            continue

        # Split the drug name on '+'
        sub_drugs = combo_name.split('+')
        # Split the SMILES on '..'
        sub_smiles = combo_smiles.split('..')

        # If lengths don't match, handle or skip
        if len(sub_drugs) != len(sub_smiles):
            # Example: skip this row or raise an error
            continue

        # Canonicalize each sub-smiles, store in map keyed by each sub-drug
        for drug, raw_smi in zip(sub_drugs, sub_smiles):
            drug = drug.strip()
            smi = raw_smi.strip()
            try:
                # Canonicalize each sub-smiles
                canon = canonicalize_smiles(smi)
            except Exception as e:
                # Optionally handle parsing errors
                canon = None
            name_to_smiles_map[drug] = canon

    # Now build the output list: for each (sub-)drug_name in the requested list
    # return the canonical SMILES if present, else None or raise an error
    result = []
    for name in drug_names:
        name = name.strip()
        if name in name_to_smiles_map:
            result.append(name_to_smiles_map[name])
        else:
            # Decide how to handle unknown sub-drugs
            result.append(None)

    return result


