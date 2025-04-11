
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm  # Optional, für Fortschrittsanzeige
import json

def calculate_class_counts(dataset):
    pre_class_counts = Counter()
    post_class_counts = Counter()
    
    # Durch den gesamten Datensatz iterieren
    for i in tqdm(range(len(dataset)), desc="Berechne Klassenverteilung"):
        # Annahme: Dataset gibt ein Tuple (pre_img, post_img, pre_mask, post_mask) zurück
        _, _, pre_mask, post_mask = dataset[i]
        
        # Konvertiere zu NumPy, falls es ein Torch-Tensor ist
        if isinstance(pre_mask, torch.Tensor):
            pre_mask = pre_mask.numpy()
        if isinstance(post_mask, torch.Tensor):
            post_mask = post_mask.numpy()
        
        # Zähle die Vorkommen jeder Klasse in den Masken
        # Für pre_mask (typischerweise Klassen 0 und 1)
        pre_classes = np.unique(pre_mask)
        for cls in pre_classes:
            pre_class_counts[int(cls)] += np.sum(pre_mask == cls)
        
        # Für post_mask (typischerweise Klassen 0 bis 5)
        post_classes = np.unique(post_mask)
        for cls in post_classes:
            post_class_counts[int(cls)] += np.sum(post_mask == cls)
    
    return pre_class_counts, post_class_counts
    
import json

def save_class_counts(pre_counts, post_counts, filepath):
    data = {
        'pre': {k: str(v) for k, v in pre_counts.items()},
        'post': {k: str(v) for k, v in post_counts.items()}
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)  # optional schön formatiert

def load_class_counts(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return Counter({str(k): v for k, v in data['pre'].items()}), Counter({int(k): v for k, v in data['post'].items()})

from collections import Counter

def calculate_class_weights(class_counts):
    total = sum(class_counts.values())
    weights = {cls: total / count for cls, count in class_counts.items()}
    # Normalize weights
    sum_weights = sum(weights.values())
    weights = {cls: weight / sum_weights * len(weights) for cls, weight in weights.items()}
    return weights

def get_sample_weights(dataset):
    sample_weights = []
    for i in range(len(dataset)):
        _, _, pre_mask, post_mask = dataset[i]
        # Calculate rare class pixels
        pre_rare = torch.sum(pre_mask == 1).float()  # Building class in pre-disaster
        post_rare = sum(torch.sum(post_mask == c).float() for c in [1, 2, 3, 4, 5])  # Damage classes
        
        # Total pixels
        total_pixels = pre_mask.numel()
        
        # Weight based on inverse frequency
        weight = 1.0 + (pre_rare + post_rare) / (total_pixels + 1e-8) * 10.0  # Adjust multiplier as needed
        sample_weights.append(weight.item())
    
    # Normalize weights
    sample_weights = torch.tensor(sample_weights)
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    return sample_weights


def save_sample_weights(weights, filepath):
    torch.save(weights, filepath)

def load_sample_weights(filepath):
    return torch.load(filepath)

# Create weighted DataLoader
def create_weighted_dataloader(dataset, sample_weights, batch_size=32, num_workers=4):
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create DataLoader with the sampler and collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader