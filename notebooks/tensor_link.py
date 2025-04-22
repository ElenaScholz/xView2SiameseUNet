from utils.helperfunctions import get_data_folder
import os
from pathlib import Path
import yaml
base_dir = Path(__file__).resolve().parent.parent
config_path = base_dir / "notebooks" / "00_config.yaml"
print(base_dir)

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

USER = config["data"]["user"]

DATA_ROOT, TRAIN_ROOT, TRAIN_IMG, TRAIN_LABEL, TRAIN_TARGET, TRAIN_PNG_IMAGES = get_data_folder(config["data"]["training_name"], main_dataset = config["data"]["use_main_dataset"])

# Definiere den Pfad, wo die TensorBoard-Logs physisch gespeichert werden sollen
EXPERIMENT_GROUP = config["data"]["experiment_group"]
EXPERIMENT_ID = config["data"]["experiment_id"]
PHYSICAL_TENSORBOARD_DIR = DATA_ROOT / EXPERIMENT_GROUP / "tensorboard_logs" / EXPERIMENT_ID
PHYSICAL_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Zielordner, wo deine TensorBoard-Logs physisch gespeichert sind
TARGET_TENSORBOARD_DIR = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/data/xview2/xView2_Experiments/tensorboard_logs")

# Symlink in deinem Home-Verzeichnis
USER_HOME_PATH = Path(f"/dss/dsshome1/08/{USER}")
HOME_TENSORBOARD_LINK = USER_HOME_PATH / "tensorboard_logs"

# Lösche den vorhandenen Symlink oder Ordner falls er existiert und ein Symlink ist
import os
if HOME_TENSORBOARD_LINK.exists():
    if HOME_TENSORBOARD_LINK.is_symlink():
        HOME_TENSORBOARD_LINK.unlink()  # Entferne den bestehenden Symlink
    else:
        import shutil
        shutil.rmtree(HOME_TENSORBOARD_LINK)  # Entferne den bestehenden Ordner

# Erstelle den Symlink
os.symlink(TARGET_TENSORBOARD_DIR, HOME_TENSORBOARD_LINK)