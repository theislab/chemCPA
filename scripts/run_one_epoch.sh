#!/bin/bash

python -m compert.train --dataset_path datasets/GSM_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm
python -m compert.train --dataset_path datasets/pachter_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type logsigm
python -m compert.train --dataset_path datasets/cross_species_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type mlp --split_key split4

python -m compert.train --dataset_path datasets/sciplex3_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm
python -m compert.train --dataset_path datasets/sciplex3_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type sigm --split_key split1

python -m compert.train --dataset_path datasets/Norman2019_prep_new.h5ad       --save_dir /tmp --max_epochs 1  --doser_type linear
for i in 1 21 22
do
   python -m compert.sweep --dataset_path datasets/Norman2019_prep_new.h5ad     --save_dir /tmp --doser_type linear --split_key split$i  --decoder_activation ReLU --max_epochs 1
done


# python -m compert.train --dataset_path datasets/GSM_new.h5ad                   --save_dir /tmp --max_epochs 1  --doser_type sigm
# python -m compert.train --dataset_path datasets/pachter.h5ad                   --save_dir /tmp --max_epochs 1  --doser_type logsigm
# python -m compert.train --dataset_path datasets/cross_species_new.h5ad         --save_dir /tmp --max_epochs 1  --doser_type  mlp


# Please don't delete it for now.
# python -m compert.train --dataset_path datasets/GSM_2k_hvg.h5ad            --save_dir /tmp --max_epochs 1  --doser_type sigm
# python -m compert.train --dataset_path datasets/GSM_4k_hvg.h5ad            --save_dir /tmp --max_epochs 1  --doser_type sigm

# python -m compert.train --dataset_path datasets/pachter.h5ad               --save_dir /tmp --max_epochs 1  --doser_type sigm
# python -m compert.train --dataset_path datasets/cross_species.h5ad         --save_dir /tmp --max_epochs 1  --doser_type mlp

# python -m compert.train --dataset_path datasets/Norman2019.h5ad            --save_dir /tmp --max_epochs 1  --doser_type linear
# python -m compert.train --dataset_path datasets/sciplex3_prepared.h5ad     --save_dir /tmp --max_epochs 1  --doser_type sigm

## Negative binomial option doesn't work yet
# python -m compert.train --dataset_path datasets/pachter.h5ad               --save_dir /tmp --max_epochs 1 --loss_ae nb     --doser_type sigm
# python -m compert.train --dataset_path datasets/GSM_4k_hvg.h5ad            --save_dir /tmp --max_epochs 1 --loss_ae nb     --doser_type sigm
# python -m compert.train --dataset_path datasets/cross_species.h5ad         --save_dir /tmp --max_epochs 1 --loss_ae nb     --doser_type mlp

