**Summary**
* Test how much pretraining on LINCS helps with improving OOD drug prediction on Trapnell.

**Why is this interesting?**
* Would allow accurate predictions of single-cell response to unseen drugs, without spending more money on the datasets. It further evaluate to which degree the finetuning works.

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