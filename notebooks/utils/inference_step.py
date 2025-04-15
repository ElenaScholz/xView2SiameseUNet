import torch
from torch.utils.data import DataLoader  # Falls du mit einem DataLoader arbeitest
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path  

def inference(model, dataloader):
    model.eval()
    
    results = {
        'pre_predictions': [],
        'post_predictions': [],
        'pre_names': [],
        'post_names': [],
        'pre_images': [],
        'post_images': []
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for pre_imgs, post_imgs, pre_names, post_names in dataloader:
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            
            # Forward pass
            outputs = model(pre_imgs, post_imgs)
            
            # Überprüfe die Form der Ausgabe
            print(f"Outputs shape: {outputs.shape}")
            
            # Trenne die Ausgaben für Pre- und Post-Disaster
            pre_outputs = outputs[:, :2]  # 2 Klassen für pre-disaster
            post_outputs = outputs[:, 2:]  # 6 Klassen für post-disaster
            
            # Detaillierte Logit-Statistiken
            print(f"Pre-outputs stats: min={pre_outputs.min().item():.4f}, max={pre_outputs.max().item():.4f}, mean={pre_outputs.mean().item():.4f}")
            print(f"Post-outputs stats: min={post_outputs.min().item():.4f}, max={post_outputs.max().item():.4f}, mean={post_outputs.mean().item():.4f}")

