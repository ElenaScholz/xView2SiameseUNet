import gc
import json
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torch



from utils.helperfunctions import get_data_folder
from utils.dataset import xView2Dataset, collate_fn, transform, image_transform
from model.siameseNetwork import SiameseUnet
from model.uNet import UNet_ResNet50
from model.loss import FocalLoss, combined_loss_function
from utils.training_preparations import calculate_class_counts,  save_class_counts, load_class_counts, get_sample_weights, save_sample_weights, load_sample_weights, calculate_class_weights
from utils.train_step import train_step
from utils.val_step import val_step
# from utils.training_preparations import create_weighted_dataloader
from torch.utils.data import WeightedRandomSampler



DATA_ROOT, TRAIN_ROOT, TRAIN_IMG, TRAIN_LABEL, TRAIN_TARGET, TRAIN_PNG_IMAGES = get_data_folder("tier1", main_dataset = True)

DATA_ROOT, VAL_ROOT, VAL_IMG, VAL_LABEL, VAL_TARGET, VAL_PNG_IMAGES = get_data_folder("hold", main_dataset = True)

# Pathes to store experiment informations in:
EXPERIMENT_GROUP = "xView2_Experiments"
EXPERIMENT_ID = "001"

EXPERIMENT_DIR = DATA_ROOT / EXPERIMENT_GROUP /"tensorboard_logs"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
print(EXPERIMENT_DIR)
# Auch Checkpoints-Verzeichnis erstellen
CHECKPOINTS_DIR = DATA_ROOT / EXPERIMENT_GROUP / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Create the Datasets for Training and Validation

train_dataset = xView2Dataset(png_path = TRAIN_PNG_IMAGES, target_path = TRAIN_TARGET, transform = transform(), image_transform = image_transform())
val_dataset = xView2Dataset(png_path = VAL_PNG_IMAGES, target_path = VAL_TARGET, transform = transform(), image_transform = image_transform())

# Basis-Ordner (geht hoch aus notebooks/)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else os.path.abspath("../")

# Korrekte Pfade
class_counts_path = os.path.join(base_dir, "precalculations", "class_counts.json")
sample_weights_path = os.path.join(base_dir, "precalculations", "sample_weights.pt")

# Wenn die Datei existiert, laden, sonst berechnen
if os.path.exists(class_counts_path):
    print("Lade gespeicherte Class Counts...")
    pre_counts, post_counts = load_class_counts(class_counts_path)
else:
    print("Berechne Class Counts...")
    pre_counts, post_counts = calculate_class_counts(train_dataset)
    save_class_counts(pre_counts, post_counts, class_counts_path)

if os.path.exists(sample_weights_path):
    print("Lade gespeicherte Sample Weights...")
    sample_weights = load_sample_weights(sample_weights_path)
else:
    print("Berechne Sample Weights...")
    sample_weights = get_sample_weights(train_dataset)
    save_sample_weights(sample_weights, sample_weights_path)


# Class Weights berechnen
pre_weights = calculate_class_weights(pre_counts)
post_weights = calculate_class_weights(post_counts)

# Device bestimmen (wichtig für CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Weights in Tensoren umwandeln
class_weights_pre = torch.tensor([
    pre_weights.get(0, 1.0), 
    pre_weights.get(1, 10.0)
], device=device)

class_weights_post = torch.tensor([
    post_weights.get(0, 1.0), 
    post_weights.get(1, 10.0),
    post_weights.get(2, 30.0),
    post_weights.get(3, 20.0),
    post_weights.get(4, 50.0),
    post_weights.get(5, 100.0)
], device=device)

# Loss-Funktionen definieren

criterion_pre = nn.CrossEntropyLoss(weight=class_weights_pre)
criterion_post = nn.CrossEntropyLoss(weight=class_weights_post)


# Constants and Setup
NUM_CLASSES = 6
EPOCHS = 50


# Create focal loss instances with class weights
focal_loss_pre = FocalLoss(alpha=class_weights_pre, gamma=2)
focal_loss_post = FocalLoss(alpha=class_weights_post, gamma=2)
# In your main script, before calling train_step
print(f"Type of focal_loss_pre: {type(focal_loss_pre)}")
print(f"Type of focal_loss_post: {type(focal_loss_post)}")
# At the beginning of your script
torch.set_default_tensor_type(torch.FloatTensor)
# Set up model
model = SiameseUnet(num_pre_classes=2, num_post_classes=6)
if torch.cuda.device_count() > 1:
    print(f"Verwende {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# Set up tensorboard writer
writer = SummaryWriter(EXPERIMENT_DIR / EXPERIMENT_ID)

# Set up optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Etwas niedrigere Startlernrate
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,          # Längerer erster Zyklus
    T_mult=2,
    eta_min=1e-6
)


if device == "cuda":
    num_workers = torch.cuda.device_count() * 4

else:
    num_workers = 4

print("Create Training Dataloader")
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 8,
    shuffle = True,
    num_workers = num_workers,
    collate_fn = collate_fn,
    pin_memory = True
)
print("Done")

print("Create Validation Dataloader")
val_dataloader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)



# Training loop
best_val_loss = float('inf')
best_checkpoint_path = None
print("starting Training Loop")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    # Training step
    avg_train_loss = train_step(model, train_dataloader, optimizer, epoch, writer, focal_loss_pre, focal_loss_post)
    
    print(f"Train Loss: {avg_train_loss:.4f}")
    
    # Validation step
    avg_val_loss = val_step(model, train_dataloader, optimizer, epoch, writer, focal_loss_pre, focal_loss_post)
    print(f"Val Loss: {avg_val_loss:.4f}")
    
    # Update learning rate
    scheduler.step()
    
    # Save checkpoint if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_checkpoint_path = CHECKPOINTS_DIR / f'{EXPERIMENT_GROUP}_{EXPERIMENT_ID}_epoch_{epoch+1}.pth'
        
        print(f"Attempting to save checkpoint to {best_checkpoint_path}")
        print(f"Directory exists: {CHECKPOINTS_DIR.exists()}")
        print(f"Directory is writable: {os.access(CHECKPOINTS_DIR, os.W_OK)}")
        
        try:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
            }, best_checkpoint_path)
            print(f'Successfully saved checkpoint at {best_checkpoint_path}')
            best_checkpoint_path_str = f"val_loss:{best_val_loss:.4f}@{best_checkpoint_path}"
            writer.add_text("Best Checkpoint Path", best_checkpoint_path_str, epoch)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    # Log current learning rate
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

writer.close()
print("Done!")

torch.save(model.state_dict(), 'precalculations/siamese_unet_weights.pth')
