# Notes on JTVAE

GPU runs out of memory if too many workers are used.
Currently the training code in the main repository is broken, a fix is at
https://github.com/siboehm/dgl-lifesci/tree/jtvae.

- `lincs_trapnell.smiles`: 17870 SMILES
- `~/.dgl/jtvae/train.txt` (ZINC): 220011 SMILES