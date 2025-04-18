import os
import torch

def iterate_through_dir(directory_path):
    for directory_path, directorynames, filenames in os.walk(directory_path):
        print(f"There are {len(directorynames)} directories and {len(filenames)} images in '{directory_path}'.")

from pathlib import Path
import torch
def get_data_folder(folder_name: str,
    main_dataset: bool):  # possible names: ["test", "tier1", "tier3", "hold", "train", "val"] 
    ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")

    DATA_PATH = ROOT / "data"
    

    if main_dataset:
        DATASET_ROOT = DATA_PATH / "xview2"
        DATA_FOLDER = DATASET_ROOT / folder_name # this has to be changed in respect to the folder (tier1, tier3, hold, test)
        IMAGE_FOLDER = DATA_FOLDER / "images/"
        LABEL_FOLDER = DATA_FOLDER / "labels/"
        TARGET_FOLDER = DATA_FOLDER / "targets/"
        PNG_FOLDER = DATA_FOLDER / "png_images/"

    else:         # Path Configuration to the xview2 Subset
        DATASET_ROOT = DATA_PATH / "xview2"
        DATA_FOLDER = DATASET_ROOT / "test" # this has to be changed in respect to the folder (tier1, tier3, hold, test)
        IMAGE_FOLDER = DATA_FOLDER / "images/"
        LABEL_FOLDER = DATA_FOLDER / "labels/"
        TARGET_FOLDER = DATA_FOLDER / "targets/"
        PNG_FOLDER = DATA_FOLDER / "png_images/"

    return DATASET_ROOT, DATA_FOLDER, IMAGE_FOLDER, LABEL_FOLDER, TARGET_FOLDER, PNG_FOLDER


import os
from pathlib import Path
import torch

def find_best_checkpoint(checkpoint_dir, experiment_id):
    """Find the best model checkpoint by filename convention."""
    checkpoint_dir = Path(checkpoint_dir)
    best_checkpoint = checkpoint_dir / f"{experiment_id}_best_siamese_unet_state.pth"
    
    if best_checkpoint.exists():
        return best_checkpoint
    else:
        print(f"No best checkpoint found for experiment {experiment_id}!")
        return None

import torch

def load_checkpoint(model, checkpoint_path):
    """Lädt ein Checkpoint und passt das state_dict automatisch an DataParallel an."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Bestimmen ob es ein raw state_dict ist oder ein dict mit Metadaten
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded full checkpoint (epoch {checkpoint.get('epoch')}, loss {checkpoint.get('loss')})")
    else:
        state_dict = checkpoint
        print(f"Loaded raw state_dict from {checkpoint_path}")

    # Modell in DataParallel packen, falls nicht schon geschehen
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Check ob state_dict schon den "module."-Präfix hat
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

    if not has_module_prefix:
        # Prefix hinzufügen
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Checkpoint erfolgreich in DataParallel-Modell geladen.")
    return model


# def find_best_checkpoint(checkpoint_dir, experiment_id):
#     """Find checkpoint with lowest validation loss in filename."""
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoints = list(checkpoint_dir.glob(f"*{experiment_id}*.pth"))
    
#     if not checkpoints:
#         print("No checkpoints found!")
#         return None
    
#     # Get the most recent checkpoint (usually best if you only save on improvement)
#     latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(x))
#     return latest_checkpoint


# def load_checkpoint(model, checkpoint_path):
#     """Load checkpoint accounting for DataParallel wrapper if needed."""
#     checkpoint = torch.load(checkpoint_path)
#     state_dict = checkpoint['model_state_dict']
    
#     # Wenn du mit DataParallel trainiert hast
#     if torch.cuda.device_count() > 1 or 'module.' in next(iter(checkpoint['model_state_dict'].keys())):
#         model = torch.nn.DataParallel(model)
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         # Wenn du ohne DataParallel trainiert hast oder das Modell für Inferenz auf einer GPU verwenden möchtest
#         state_dict = checkpoint['model_state_dict']
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if key.startswith('module.'):
#                 new_state_dict[key[7:]] = value
#             else:
#                 new_state_dict[key] = value
#         model.load_state_dict(new_state_dict)



#     # Check if the state_dict was saved with DataParallel
#     if list(state_dict.keys())[0].startswith('module.'):
#         if not isinstance(model, torch.nn.DataParallel):
#             new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
#             state_dict = new_state_dict
#     else:
#         if isinstance(model, torch.nn.DataParallel):
#             new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
#             state_dict = new_state_dict
    
#     model.load_state_dict(state_dict)
#     print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
#     return model