#!/bin/bash
#SBATCH -J xView2_model_Training
#SBATCH -o /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di97ren/xView2_Subset/logfiles/001/stdout.logfile
#SBATCH -e /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di97ren/xView2_Subset/logfiles/001/stderr.logfile
#SBATCH -D /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di97ren
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:3  # Request 1 GPU
#SBATCH --cpus-per-task=40    
#SBATCH --mem=100gb
#SBATCH --mail-type=all
#SBATCH --mail-user=elena.scholz@stud-mail.uni-wuerzburg.de
#SBATCH --export=NONE
#SBATCH --time=40:00:00
#SBATCH --account=di97ren

# Load necessary modules (if required by the cluster)
module load slurm_setup
module load python
module load uv
module load cuda

# Python script
script="02-DOTA_FasterRCNN.py"

# Run your Python script
uv run --no-project -p /dss/dsshome1/0A/di38tac/04-geo-oma24/course_material_04_geo_oma24/.venv python /dss/dsshome1/0A/di38tac/DOTA-Net/code/$script