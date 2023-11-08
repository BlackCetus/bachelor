#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --partition=shared-gpu
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/esm_embeddings.err
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/esm_embeddings.txt
#SBATCH --job-name=esm_embeddings
#SBATCH --mem-per-gpu=1G

srun esm-extract esm2_t36_3B_UR50D /nfs/home/students/t.reim/bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta   /nfs/home/students/t.reim/bachelor/pytorchtest/data/embeddings/esm2_t36_3B  --include mean
