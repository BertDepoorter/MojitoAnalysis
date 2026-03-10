#!/bin/bash -l
#SBATCH -t 8:00:00
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_a100 
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bert.depoorter@student.kuleuven.be
#SBATCH -A lp_lisagw

nvidia-smi
conda activate lisatools_env

cdw
cd LISA/Mojito_analysis

python PE_validation.py --source_index=0