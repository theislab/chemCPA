
# Predicting ood drug perturbations via transfer learning: Experiment setup
This file explains the experiment setup for the chemical compositional perturbational autoencoder (CCPA). The idea of this project is to investigate how well counterfactual predictions can be made for unseen drugs at a single cell level. 

To this end, we extended the original [CPA model](https://github.com/facebookresearch/CPA) to include drug embedding neural networks, that can map SMILES strings to a chemically meanigful embedding. Since these neural networks are not limited to a set of drugs used in a real experiment, it is possible to investigate perturbations for any drug. We hypothesise that such an ood perturbational prediction can only be meaningful when the model has seen a sufficient number of different drugs. However, scRNA-seq perturbation screens are limited in the number of drugs they can investiate. The sciplex dataset, for example, contains only 188 drugs. To alleviate this shortcoming, we aim to enrich the drug latent space with a transfer learning approach. We investigate how and, more importantly, if a blukSeq perturbation screen, LINCS, which include more than 17k different drugs, can be used to improve the ood generalisation of the CCPA model for single-cell resolved perturbation predictions. 

We split the experiments in multiple parts:  

## 1. EXP: `lincs_rdkit_hparam`
This experiment is used to identify model configurations that have sufficient performance on the LINCS datasets. The resulting models configurations are then transferred for the finetuning experiments `sciplex_hparam`, `fintuning_num_genes`, and `fintuning_OOD_prediciton`.  

This experiment runs in 2 steps:
1. We run a (large) hyperparameter sweep using CCPA with RDKit embeddings. We use the results to pick good hyperparameters for the autoencoder and the adversarial predictors. See `config_lincs_rdkit_hparam_sweep.yaml`.
2. We run a (small) hyperparameter sweep using CCPA with all other embeddings. We sweep just over the embedding-related hparams (drug encoder, drug doser), while fixing the AE & ADV related hparams as selected through (1). See `config_lincs_all_embeddings_hparam_sweep.yaml`.


## 2. EXP: `sciplex_hparam`
This experiment is run to determine suitable optimisation hparams for the adversary when fine-tuning from LINCS to the sciplex dataset. These hparams are meant to be shared when evaluating transfer performace for different drug embedding models. This experiment is preliminary work for `finetuning_num_genes` and `finetuning_OO_prediction`.  

Similar to `lincs_rdkit_hparam`, we subset to the `grover_base` and `rdkit` embedding to be considerate wrt to compute resources. 

Setup: 
- Importantly, we sweep over a split that set some drugs as ood. In this setting the vanilla model is not applicable anymore. The drugs were chosen according to the results from the original [sciplex publication](https://www.science.org/doi/full/10.1126/science.aax6234) and the ood set includes only drugs that have introduced a significant perturbation, cf. Fig.S6 in the supplements of the publication. 
- Additionally, we include the `split_ho_pathway` split for further validation. Here, only the maximum dosage of some drugs is true ood. Hence, the vanilla model is applicable in this scenario. 

## 3. EXP: `finetuning_num_genes`
In this experiment we test how the pretraining on lincs with a smaller set of genes helps to increase the performance on a larger gene set for sciplex. We use a split where some drugs from several pathways are chosen as ood, see explanations in `sciplex_hparam`. 

**Why is this interesting?**
* This is biologically relevant as different single-cell datasets have different important gene sets that explain their variation.

**Experiment steps**:

1. Pretrain on LINCS (~900 genes), finetune on Trapnell (same ~900 genes) - `'config_sciplex_finetune_lincs_genes.yaml'`
2. Pretrain on LINCS (~900 genes), finetune on Trapnell (2000 genes)
3. Train from Scratch on Trapnell (900 genes) - `'config_sciplex_finetune_lincs_genes.yaml'`
4. Train from Scratch on Trapnell (2000 genes)

Compare performances between CCPA with pretraining (1. and 2.) and CCPA without pretraining (3. and 4.) for each of the two settings.

## 4. EXP: `finetuning_OOD_prediction`
Test how and if pretraining on LINCS helps with improving OOD drug prediction on Trapnell. This is similar to `sciplex_hparam` but run on more models (drug embedders) and multiple splits.  

**Why is this interesting?**
* Would allow accurate predictions of single-cell response to unseen drugs, without spending more money on the datasets. It further evaluates to which degree the finetuning works.

**Experiment steps:**
1. Pick 1-3 drugs that exist in both LINCS and Trapnell. These are drugs that have a large effect on the transcriptome like Quisinostat (epigenetic), Flavopiridol (cell cycle regulation), and BMS-754807 (tyrosine kinase signaling).
2. Pretrain 2 models:
    - One model that is trained on all the LINCS data - already done in '**EXP:** `lincs_rdkit_hparam`'
    - One model that is trained on the LINCS data, with the 30 drugs to be tested left out. This is the `'split_ood_drugs'` split in the LINCS dataset. 
3. Finetune the pretrained models on Trapnell (3 splits: `'split_epigenetic_ood'`, `'split_tyrosine_ood'`, `'split_cellcycle_ood'`)
4. Train a model on the same three splits on Trapnell without pre-training
5. Compute the $r^2$-score for left out drugs in terms of counterfactuals. That is, encoding a control and adding the latent drug embedding before decoding. 

**Ideal outcome**
* The pretrained models perform better than the non-pretrained model.
* The model that has seen the hold-out drugs on LINCS performs better than the pre-trained model that has not seen the drugs before.
