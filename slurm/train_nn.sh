#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --partition=shared-gpu
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/train_nn_emb.err
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/train_nn_emb.txt
#SBATCH --job-name=train_nn_emb
#SBATCH --mem-per-gpu=1G

python /nfs/home/students/t.reim/bachelor/pytorchtest/main.py