#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name="graphkernels"
#SBATCH --partition=tcourt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output=output_graphkernels.txt
#SBATCH --error=error_graphkernels.txt
#
#SBATCH --ntasks=1
#SBATCH --nodes=2
#SBATCH --cpus-per-task=56
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4000

srun hostname
srun cd /home/2017018/ljia01/graphkit-learn/notebooks
srun python3 run_spkernel.py
