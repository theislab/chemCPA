import unittest
from unittest import TestCase

import pandas as pd
from create_gene_protein_map import create_gene_protein_map
from unittest.mock import patch, mock_open

# For peace of midnd that the code does what it should do

class Test(TestCase):
    @patch("pandas.read_csv")
    def test_simple(self, mock_read_csv):
        expected = {
            'gene1': {'protein2'},
            'gene2': {'protein1'}
        }
        mock_read_csv.return_value = pd.DataFrame({
            'Gene Names': ['gene2', 'gene1'],
            'Entry': ['protein1', 'protein2']
        })
        result = create_gene_protein_map("fake_file.tsv")
        self.assertEqual(result, expected)

    @patch("pandas.read_csv")
    def test_multiple_proteins_same_gene(self, mock_read_csv):
        expected = {
            'gene1': {'protein1', 'protein3'}
        }
        mock_read_csv.return_value = pd.DataFrame({
            'Gene Names': ['gene1', 'gene1'],
            'Entry': ['protein1', 'protein3']
        })
        result = create_gene_protein_map("fake_file.tsv")
        self.assertEqual(result, expected)

    @patch("pandas.read_csv")
    def test_multiple_genes_per_protein(self, mock_read_csv):
        expected = {
            'gene1': {'protein1'},
            'gene2': {'protein1'},
            'gene3': {'protein2'},
            'gene4': {'protein2'}
        }
        mock_read_csv.return_value = pd.DataFrame({
            'Gene Names': ['gene1 gene2', 'gene3 gene4'],
            'Entry': ['protein1', 'protein2']
        })
        result = create_gene_protein_map("fake_file.tsv")
        self.assertEqual(result, expected)
