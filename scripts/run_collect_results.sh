#!/bin/bash

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_GSM_new_logsigm
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_pachter_new
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_cross_species_new
for i in {1..4}
do
   python -m compert.collect_results  --save_dir /checkpoint/$USER/sweep_cross_species_new_split$i 
done

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_new_logsigm
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_new

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_split1

for i in 1 21 22
do
   python -m compert.collect_results   --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i
done

for i in {1..28}
do
   python -m compert.collect_results  --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_split$i
done

for i in {2..23}
do
   python -m compert.collect_results  --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_split$i
done



for i in {2..20}
do
   python -m compert.collect_results  --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i
done

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split23


python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_new
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_GSM_new
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_pachter_new_logsigm



for i in 24 25
do
   python -m compert.collect_results  --save_dir /checkpoint/$USER/sweep_Norman2019_prep_new_relu_split$i
done

#python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_old_logsigm
#python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_old

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_logsigm
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_sciplex3_old_reproduced_sigm

python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_lincs_logsigm
python -m compert.collect_results     --save_dir /checkpoint/$USER/sweep_lincs_sigm

python -m compert.collect_results     --save_dir /checkpoint/$USER/kang_split


# to remove soon
# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_GSM_2k_hvg "$@"
# echo "                                          [0.954, 0.931, 0.824, 0.567] (0.819, old)"

# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_GSM_4k_hvg "$@"
# echo "                                          [0.919, 0.756, 0.827, 0.256] (0.689, old)"

# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_pachter "$@"
# echo "                                          [0.957, 0.881, 0.878, 0.152] (0.717, old)"

# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_cross_species "$@"
# echo "                                          [0.973, 0.894, 0.922, 0.838] (0.906, old)"

# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_Norman2019 "$@"

# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_sciplex3_prepared "$@"

# python -m compert.collect_results --save_dir /checkpoint/$USER/sweep_sciplex3_prepared_logsigm "$@"

