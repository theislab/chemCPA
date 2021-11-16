#!/usr/bin/env zsh

#SBATCH -o slurm_output.txt
#SBATCH -e slurm_error.txt
#SBATCH -J seq2seq
#SBATCH --partition gpu_p
#SBATCH --cpus-per-task 6
#SBATCH --mem=16G
#SBATCH --exclude=supergpu05
#SBATCH --gres=gpu:1
#SBATCH --gres=mps:40
#SBATCH --qos=gpu
#SBATCH --time 05:00:00
#SBATCH --nice=10000

echo "Started running $(date)"
/home/icb/simon.boehm/miniconda3/envs/cpa_seq2seq/bin/python3 train_model.py
echo "Ending $(date)"
