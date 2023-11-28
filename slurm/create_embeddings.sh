#!/bin/bash
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=40
#SBATCH --partition=shared-cpu
#SBATCH --exclude=gpu01.exbio.wzw.tum.de
#SBATCH --time=2-0
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/%j.out
#SBATCH --job-name=create_embeddings

srun --exclusive -n1 esm-extract esm2_t48_15B_UR50D /nfs/home/students/t.reim/bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner_10k.fasta /nfs/home/students/t.reim/bachelor/pytorchtest/data/embeddings/esm2_t48_15B/mean --include mean &
srun --exclusive -n1 esm-extract esm2_t48_15B_UR50D /nfs/home/students/t.reim/bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner_10k.fasta /nfs/home/students/t.reim/bachelor/pytorchtest/data/embeddings/esm2_t48_15B/per_tok --include per_tok &
wait