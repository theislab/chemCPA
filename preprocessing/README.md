# Preprocessing

This folder contains preprocessing notebooks that convert the raw data to datasets that may
be used for training. 

## Description
Briefly:
1. The first notebook cleans up the LINCS dataset, computes DEGS, and splits.
2. The second notebook adds the SMILES information to the LINCS dataset.
3. The third notebook finds matching genes between LINCS and SciPlex-3 datasets, and creates datasets with only subsets of genes that match in some way.
4. The fourth notebook adds the SMILES information to the SciPlex-3 dataset
5. The fifth notebook creates various sub-datasets with varying observations and  train/test/ood splits
6. The sixth notebook computes a baseline dataset

For more details read the notebooks.

### Clarifcation on the avaialable datasets

The `lincs_full.h5ad` as a combination of the available L1000 datasets from phase 1 and phase 2, available here: 
- L1000 Connectivity Map Phase I: [GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138)
- L1000 Connectivity Map Phase II: [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)

We combined the data which is only available in the `.gctx` together with its metadata, cf. `<...>_inst_info.txt`, and make it available as scanpy compatible `.h5ad` object. 

Note that not all perturbations correspond to small molecules which is why we subsetted the data to only contain perturbation types `trt_cp` and `ctl_vehicle`, resulting in a total of 1034271 observations.

The provided data is normalised. 

For the training on the LINCS data, we ignored the treatment time, `adata_lincs_full.obs["pert_time"]`.

#### Preprocess data 
The data preprocessing should run thorugh with the provided files. For the matching of genes between LINCS and the SciPlex-3 data in `3_lincs_sciplex_gene_matching.ipynb`, we provide a [`symbols_dict`](https://drive.google.com/file/d/16V5nyj3xKlsUk_cJtRYtkpFiglGzP9Xl/view?usp=sharing) which replaces the matching via `sfaira`. Note that you have to execute `4_sciplex_SMILES.ipynb` for both gene sets. The same notebook also contains multiple check against the [`trapnell_final_V7.h5ad`](https://drive.google.com/file/d/1_JUg631r_QfZhKl9NZXXzVefgCMPXE_9/view?usp=share_link) file to make sure that the SMILES are correctly matched. You could ignore these and implement your own solution. For this, we provide `drug_dict.json` file.

#### Files produced

