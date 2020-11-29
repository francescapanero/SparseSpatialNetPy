# eventually I'll get how to run the code on the server


#!/bin/bash
#SBATCH -A oxwasp                       # Account to be used, e.g. academic, acadrel, aims, bigbayes, opig, oxcsml, oxwasp, rstudent, statgen, statml, visitors
#SBATCH -J job01                          # Job name, can be useful but optional
#SBATCH --time=00:10:00                   # Walltime - run time of just 30 seconds
#SBATCH --mail-user=panero@stats.ox.ac.uk     # set email address to use, change to your own email address instead of "me"
#SBATCH --mail-type=ALL                   # Caution: fine for debug, but not if handling hundreds of jobs!
#SBATCH --output="/tmp/slurm-%u-%A-std-output" # To avoid slurm job failing if it can't write to current directory for standard output
#SBATCH --error="/tmp/slurm-%u-%A-err-output" # To avoid slurm job failing if it can't write to current directory for standard error

echo Starting on `hostname`              # This goes to the job output
hostname > /tmp/${USER}-test             # Will be written to this file again on the host this job runs on
python3 degree_distr.py