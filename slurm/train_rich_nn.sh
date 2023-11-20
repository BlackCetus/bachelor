#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --partition=exbio-gpu
#SBATCH --nodelist=gpu02.exbio.wzw.tum.de
#SBATCH --exclude=gpu01.exbio.wzw.tum.de
#SBATCH --time=08:00:00
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/%j.out
#SBATCH --job-name=train_rich_nn
#SBATCH --mem-per-gpu=10G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/main.py -data gold_stand -model richoux -max 2000 -lr 0.001 -epoch 25 -batch 512 -sub -subsize 0.25 -emb -emb_dim 1280 -wandb