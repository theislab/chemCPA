In this experiment we test how the pretraining on lincs with smaller set of genes helps to increase the performance on a larger amount of genes for sciplex. We use a split for this where 1-2 drugs from each pathway are are choosen as ood, see `'split_ho_pathway'` in `lincs_sciplex_gene_matching.ipynb`. 85% of the maximum dosage observations of these drugs are set as `'ood'` and 15% are set as `'test'`. 

This is biologically relevant as different single-cell datasets have different important gene sets that explain their variation.

Experiment steps:

1. Pretrain on LINCS (~900 genes), finetune on Trapnell (same ~900 genes) - `'config_sciplex_finetune_lincs_genes.yaml'`
2. Pretrain on LINCS (~900 genes), finetune on Trapnell (2000 genes)
3. Train from Scratch on Trapnell (900 genes) - `'config_sciplex_finetune_lincs_genes.yaml'`
4. Train from Scratch on Trapnell (2000 genes)

Compare performances between CCPA with pretraining (1. and 2.) and CCPA without pretraining (3. and 4.) for each of the two settings.