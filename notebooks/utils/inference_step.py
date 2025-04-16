import torch
from torch.utils.data import DataLoader  # Falls du mit einem DataLoader arbeitest
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path  
import numpy as np

# def inference(model, dataloader):
#     model.eval()
    
#     results = {
#         'pre_predictions': [],
#         'post_predictions': [],
#         'pre_names': [],
#         'post_names': [],
#         'pre_images': [],
#         'post_images': []
#     }
    
#     total_samples = 0
    
#     with torch.no_grad():
#         for pre_imgs, post_imgs, pre_names, post_names in dataloader:
#             pre_imgs = pre_imgs.to(device)
#             post_imgs = post_imgs.to(device)
            
#             # Forward pass
#             outputs = model(pre_imgs, post_imgs)
            
#             # Überprüfe die Form der Ausgabe
#             print(f"Outputs shape: {outputs.shape}")
            
#             # Trenne die Ausgaben für Pre- und Post-Disaster
#             pre_outputs = outputs[:, :2]  # 2 Klassen für pre-disaster
#             post_outputs = outputs[:, 2:]  # 6 Klassen für post-disaster
            
#             # Detaillierte Logit-Statistiken
#             print(f"Pre-outputs stats: min={pre_outputs.min().item():.4f}, max={pre_outputs.max().item():.4f}, mean={pre_outputs.mean().item():.4f}")
#             print(f"Post-outputs stats: min={post_outputs.min().item():.4f}, max={post_outputs.max().item():.4f}, mean={post_outputs.mean().item():.4f}")
            
#             # Schaue dir die erste Batch an
#             for b in range(min(1, pre_outputs.shape[0])):
#                 print(f"Bild {b}:")
#                 for c in range(pre_outputs.shape[1]):
#                     print(f"  Pre-Klasse {c}: min={pre_outputs[b,c].min().item():.4f}, max={pre_outputs[b,c].max().item():.4f}, mean={pre_outputs[b,c].mean().item():.4f}")
#                 for c in range(post_outputs.shape[1]):
#                     print(f"  Post-Klasse {c}: min={post_outputs[b,c].min().item():.4f}, max={post_outputs[b,c].max().item():.4f}, mean={post_outputs[b,c].mean().item():.4f}")
            
#             # Prüfe die Unterschiede zwischen den Logits an einem bestimmten Pixel
#             sample_x, sample_y = 100, 100
#             print(f"Sample at position ({sample_x},{sample_y}):")
#             print(f"  Pre-logits: {pre_outputs[0, :, sample_x, sample_y]}")
#             print(f"  Post-logits: {post_outputs[0, :, sample_x, sample_y]}")
            
#             # Softmax anwenden
#             pre_probs = torch.softmax(pre_outputs, dim=1)
#             post_probs = torch.softmax(post_outputs, dim=1)
            
#             # Prüfe die Wahrscheinlichkeiten am selben Pixel
#             print(f"  Pre-probs: {pre_probs[0, :, sample_x, sample_y]}")
#             print(f"  Post-probs: {post_probs[0, :, sample_x, sample_y]}")
            
#             # Vorhersagen erstellen
#             pre_pred = torch.argmax(pre_probs, dim=1).cpu().numpy()
#             post_pred = torch.argmax(post_probs, dim=1).cpu().numpy()
            
#             # Prüfe die Verteilung der Vorhersagen
#             pre_classes, pre_counts = np.unique(pre_pred, return_counts=True)
#             post_classes, post_counts = np.unique(post_pred, return_counts=True)
            
#             print(f"Pre-disaster class distribution: {dict(zip(pre_classes, pre_counts))}")
#             print(f"Post-disaster class distribution: {dict(zip(post_classes, post_counts))}")
            
#             # Speichere Bilder für die Visualisierung
#             pre_images_cpu = pre_imgs.cpu().numpy()
#             post_images_cpu = post_imgs.cpu().numpy()
            
#             # Speichere Ergebnisse
#             results['pre_predictions'].extend(pre_pred)
#             results['post_predictions'].extend(post_pred)
#             results['pre_names'].extend(pre_names)
#             results['post_names'].extend(post_names)
#             results['pre_images'].extend(pre_images_cpu)
#             results['post_images'].extend(post_images_cpu)
            
#             total_samples += len(pre_names)
    
#     # Leistungsmetriken (optional)
#     results['performance'] = {
#         'total_samples': total_samples
#     }
    
#     return results

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

def inference(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    
    results = {
        'pre_predictions': [],
        'post_predictions': [],
        'pre_names': [],
        'post_names': [],
        'pre_images': [],
        'post_images': []
    }
    
    total_samples = 0
    processing_time = 0
    
    with torch.no_grad():
        # Füge tqdm für Fortschrittsanzeige hinzu
        progress_bar = tqdm(dataloader, desc="Running inference", unit="batch")
        
        for pre_imgs, post_imgs, pre_names, post_names in progress_bar:
            batch_start_time = time.time()
            
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            
            # Forward pass
            outputs = model(pre_imgs, post_imgs)
            
            # Trenne die Ausgaben für Pre- und Post-Disaster
            pre_outputs = outputs[:, :2]  # 2 Klassen für pre-disaster
            post_outputs = outputs[:, 2:]  # 6 Klassen für post-disaster
            
            # Softmax anwenden
            pre_probs = torch.softmax(pre_outputs, dim=1)
            post_probs = torch.softmax(post_outputs, dim=1)
            
            # Vorhersagen erstellen
            pre_pred = torch.argmax(pre_probs, dim=1).cpu().numpy()
            post_pred = torch.argmax(post_probs, dim=1).cpu().numpy()
            
            # Speichere Bilder für die Visualisierung
            pre_images_cpu = pre_imgs.cpu().numpy()
            post_images_cpu = post_imgs.cpu().numpy()
            
            # Speichere Ergebnisse
            results['pre_predictions'].extend(pre_pred)
            results['post_predictions'].extend(post_pred)
            results['pre_names'].extend(pre_names)
            results['post_names'].extend(post_names)
            results['pre_images'].extend(pre_images_cpu)
            results['post_images'].extend(post_images_cpu)
            
            batch_time = time.time() - batch_start_time
            processing_time += batch_time
            total_samples += len(pre_names)
            
            # Update progress bar
            progress_bar.set_postfix({"Samples": total_samples, "Avg time/batch": f"{processing_time / (progress_bar.n + 1):.4f}s"})
    
    # Berechne Klassenverteilungen für Leistungsmetriken
    all_pre_preds = np.concatenate(results['pre_predictions'])
    all_post_preds = np.concatenate(results['post_predictions'])
    
    pre_classes, pre_counts = np.unique(all_pre_preds, return_counts=True)
    post_classes, post_counts = np.unique(all_post_preds, return_counts=True)
    
    pre_distribution = {int(cls): int(count) for cls, count in zip(pre_classes, pre_counts)}
    post_distribution = {int(cls): int(count) for cls, count in zip(post_classes, post_counts)}
    
    # Leistungsmetriken
    results['performance'] = {
        'total_samples': total_samples,
        'total_processing_time': processing_time,
        'avg_time_per_sample': processing_time / total_samples if total_samples > 0 else 0,
        'pre_class_distribution': pre_distribution,
        'post_class_distribution': post_distribution
    }
    
    return results
