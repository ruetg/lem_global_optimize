#!/bin/bash
#SBATCH --job-name=lem    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=garueten@syr.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=38
#SBATCH --nodes=1
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --output=serial_test_%j.log   # Standard output and error log
. "/projects/gregr13210@xsede.org/miniconda3/etc/profile.d/conda.sh"
srun /projects/gregr13210@xsede.org/miniconda3/bin/python3.9 Run_montecarlo_diffusion.py 
