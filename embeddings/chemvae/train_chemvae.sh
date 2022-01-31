#!/usr/bin/env bash

python train.py vae --train_load ../lincs_trapnell_zinc.csv \
                    --val_load ../zinc_smiles_test.txt  \
                    --config_save /storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings/chemvae/config.txt \
                    --model_save /storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings/chemvae/model.pt \
                    --vocab_save /storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings/chemvae/vocab.txt \
                    --device cuda:0 \
                    --kl_w_end 1.0