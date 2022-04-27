**Summary**
* In this experiment we test how the pretraining on lincs with a smaller set of genes helps to increase the performance on a larger gene set for sciplex. We use the ood split `split_ood_finetuning` in `lincs_sciplex_gene_matching.ipynb`.

**Why is this interesting?**
* This is biologically relevant as different single-cell datasets have different important gene sets that explain their variation.

**Experiment steps**:

1. Pretrain on LINCS (~900 genes), finetune on Trapnell (same ~900 genes) - `'config_sciplex_finetune_lincs_genes.yaml'`
2. Pretrain on LINCS (~900 genes), finetune on Trapnell (2000 genes)
3. Train from Scratch on Trapnell (900 genes) - `'config_sciplex_finetune_lincs_genes.yaml'`
4. Train from Scratch on Trapnell (2000 genes)

Compare performances between chemCPA with pretraining (1. and 2.) and chemCPA without pretraining (3. and 4.) for each of the two settings.