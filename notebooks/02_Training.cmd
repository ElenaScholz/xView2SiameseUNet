#!/bin/bash
#SBATCH -J xView2_model_Training
#SBATCH -o /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di97ren/xView2_Subset/logfiles/001/stdout.logfile
#SBATCH -e /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di97ren/xView2_Subset/logfiles/001/stderr.logfile
#SBATCH -D /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di97ren
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40    
#SBATCH --mem=100gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elena.scholz@stud-mail.uni-wuerzburg.de
#SBATCH --export=NONE
#SBATCH --time=40:00:00
#SBATCH --account=pn39sa-c

# Lade Module
module load slurm_setup
module load python
module load uv
module load cuda

# Variablen
VENV_PATH="/dss/dsshome1/08/di97ren/04-geo-oma24/xView2SiameseUNet/.venv"
SCRIPT_PATH="/dss/dsshome1/08/di97ren/04-geo-oma24/xView2SiameseUNet/notebooks/developer_main.py"

# Debug-Ausgabe
echo "Starte Training mit uv..."
echo "Verwende virtuelle Umgebung: $VENV_PATH"
echo "Führe Skript aus: $SCRIPT_PATH"

# Starte Trainingsskript über uv
uv run --no-project -p "$VENV_PATH" python "$SCRIPT_PATH"
