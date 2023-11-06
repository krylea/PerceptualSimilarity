#!/bin/bash
#SBATCH --job-name=trainlpips
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=45GB

python3 train_embeds.py $1