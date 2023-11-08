#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/dummy.err
#SBATCH --time=00:00:59
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/dummy.txt
#SBATCH --job-name=dummy
#SBATCH --mem-per-cpu=100

echo 'test'