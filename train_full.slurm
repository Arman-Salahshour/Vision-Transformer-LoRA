#!/bin/bash
#SBATCH --job-name=vit-full
#SBATCH --account=ec674
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1

module purge
module load JupyterLab/4.2.5-GCCcore-13.3.0
source /projects/ec674/venvs/in5310h26/bin/activate

cd $SLURM_SUBMIT_DIR
make run MODE=full