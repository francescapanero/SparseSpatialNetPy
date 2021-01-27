#!/bin/bash
#SBATCH -A oxwasp
#SBATCH -J job01                    
#SBATCH --time=00:01:00                  
#SBATCH --mail-user=panero@stats.ox.ac.uk  
#SBATCH --mail-type=ALL     
#SBATCH --output=/tmp/slurm-%u-%A-std-output
#SBATCH --error=/tmp/slurm-%u-%A-err-output

python slurm_test.py

