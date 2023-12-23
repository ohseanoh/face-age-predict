#!/usr/bin/bash
#SBATCH -J main-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y7
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

#SBATCH --mail-type=END
#SBATCH --mail-user=seanoh5403@gmail.com

ipython 1_less_param_BW_CNN.py
