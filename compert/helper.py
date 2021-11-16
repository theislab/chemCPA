import warnings
from typing import Optional

import dgl
import pandas as pd
import scanpy as sc
from dgllife.utils import (
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
    smiles_to_bigraph,
)
from rdkit import Chem


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        print(cov_cat)
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict


def graph_from_smiles(
    drug_smiles_df,
    perturbation_key,
    smiles_key,
    drugs_ordering,
    mol_featuriser="canonical",
):
    """Translates molecular graph information from the drug SMILES and outputs the graph tuple.

    Params
    ------
    drug_smiles_df : pd.DataFrame()
        Must contain columns `perturbation_key` and `smiles_key`
    perturbation_key : str
        Column name in `drug_smiles_df` with compound names
    smiles_key : str
        Column name in `drug_smiles_df` with SMILES
    drugs_ordering : list of drugnames
        the ordering of the drugs in the list is used to order perturbations in `drug_smiles_df`
    mol_featuriser : str
        Molecule featurizer. Must be one of `['canonical', 'AttentiveFP', 'Pretrain']`.

    Returns
    -------
    batched_graph_collection: DGLGraph
        Batched version of individual SMILES graphs
    idx_wo_smiles: list
        Indices of drug_smiles_df which have an empty SMILES string
    graph_feats_shape: tuple
        The tuple that indicates the shape of the batched graph collection.
    """

    # Order drugs as in drugs_ordering
    df = drug_smiles_df.drop_duplicates(subset=[perturbation_key])
    df = df.set_index(perturbation_key).reindex(drugs_ordering).reset_index()
    assert df[perturbation_key].isna().sum() == 0

    # Define featurisers
    featurisers, graph_feats_shape = get_featurisers(mol_featuriser, return_shape=True)
    node_feats, edge_feats = featurisers

    # List of valid graphs from SMILES
    smiles2bidirected_graph = lambda smiles: smiles_to_bigraph(
        smiles=smiles,
        add_self_loop=True,
        node_featurizer=node_feats,
        edge_featurizer=edge_feats,
    )

    def check_smiles(smiles, mol_featuriser, verbose=False):
        smiles_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if smiles != smiles_canonical:
            print(f"{smiles} -> {smiles_canonical}") if verbose else None
        if mol_featuriser == "Pretrain":
            if "*." in smiles_canonical:
                new_smiles = smiles_canonical.replace("*.", "")
                print(f"Ignore unknown atom for smiles: {smiles} -> {new_smiles}.")
                smiles = new_smiles
        return smiles

    idx_wo_smiles = []
    idx_bad_smiles = []
    valid_graphs = []
    for i, smiles in enumerate(df[smiles_key]):
        smiles = check_smiles(smiles, mol_featuriser)
        if smiles == "":
            idx_wo_smiles.append(i)
            continue
        graph = smiles2bidirected_graph(smiles)
        valid_graphs.append(graph) if graph is not None else idx_bad_smiles.append(i)

    def send_warning(idx, signal):
        if len(idx) > 0:
            pk = perturbation_key
            warnings.warn(f"Got {signal} SMILES for {pk}: {df.loc[idx, pk].values}")

    send_warning(idx_bad_smiles, "invalid")
    send_warning(idx_wo_smiles, "empty")

    batched_graph_collection = dgl.batch(valid_graphs)

    return batched_graph_collection, idx_wo_smiles, graph_feats_shape


def get_featurisers(mol_featurizer: str = "canonical", return_shape: bool = True):
    """
    Returns the tuple (node_feats, edge_feats) for the specified molecule featurizer
    and optionally their shape
    """
    featurizers = ["canonical", "AttentiveFP", "Pretrain"]
    if mol_featurizer == "canonical":
        node_feats = CanonicalAtomFeaturizer(atom_data_field="h")
        edge_feats = CanonicalBondFeaturizer(bond_data_field="h", self_loop=True)
        graph_feats_shape = (node_feats.feat_size("h"), edge_feats.feat_size("h"))
    elif mol_featurizer == "AttentiveFP":
        node_feats = AttentiveFPAtomFeaturizer(atom_data_field="h")
        edge_feats = AttentiveFPBondFeaturizer(bond_data_field="h", self_loop=True)
        graph_feats_shape = (node_feats.feat_size("h"), edge_feats.feat_size("h"))
    elif mol_featurizer == "Pretrain":
        node_feats = PretrainAtomFeaturizer()
        edge_feats = PretrainBondFeaturizer()
        graph_feats_shape = (2, 2)
    else:
        raise ValueError(f"mol_featurizer:'{mol_featurizer}' is not in {featurizers}")

    if return_shape:
        return (node_feats, edge_feats), graph_feats_shape
    return node_feats, edge_feats


def canonicalize_smiles(smiles: Optional[str]):
    if smiles:
        return Chem.CanonSmiles(smiles)
    else:
        return None
