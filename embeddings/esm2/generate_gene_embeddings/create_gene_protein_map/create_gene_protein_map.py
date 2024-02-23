import pandas as pd


def create_gene_protein_map(path: str):
    """
    Parameters:
    - path (str): A TSV file, probably from UniProtKB, should contain 'Gene Names' and 'Entry' columns.
    Returns:
    - dict: a dictionary mapping gene names to a list of corresponding protein entry names.
    """
    return (pd.read_csv(path, delimiter='\t', usecols=['Gene Names', 'Entry'])
            .assign(**{'Gene Names': lambda x: x['Gene Names'].str.split(' ')})
            .explode('Gene Names')
            .groupby('Gene Names')['Entry']
            .apply(list)
            .to_dict())
