#!/bin/bash -l
#SBATCH -t 20:00:00
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
pwd
HOME_FOLDER=/data/leuven/367/vsc36785/LISA/Mojito_analysis
SCRIPT=$HOME_FOLDER/PE_validation.py
python $SCRIPT --source=0 --cluster=vsc
