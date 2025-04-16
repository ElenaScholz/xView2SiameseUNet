# import os
# from pathlib import Path

# def find_best_checkpoint(checkpoint_dir, experiment_id=experiment_id):
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

import torch
import os
from pathlib import Path

def find_best_checkpoint(checkpoint_dir, experiment_id):
    """Find checkpoint with latest modification time (usually best one)."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob(f"*{experiment_id}*.pth"))
    
    if not checkpoints:
        print("No checkpoints found!")
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(x))
    return latest_checkpoint


def load_checkpoint(model, checkpoint_path, device=None):
    """Load model checkpoint, handling DataParallel if needed."""
    map_location = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix if model was trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(map_location)

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} with loss {checkpoint.get('loss', '?')}")
    return model
