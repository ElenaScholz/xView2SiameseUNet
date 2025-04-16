
import torch
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import sys

# Füge hier deine Importe ein
from utils.inference_step import inference
from utils.helperfunctions import load_checkpoint, find_best_checkpoint, get_data_folder
from utils.dataset import xView2Dataset, collate_fn_test, image_transform
from pathlib import Path
from torch.utils.data import DataLoader
import os
import yaml
from model.siameseNetwork import SiameseUnet
from model.uNet import UNet_ResNet50
from model.loss import FocalLoss, combined_loss_function
import torch
import yaml
base_dir = Path(os.getcwd()).parent  # Gehe einen Ordner zurück vom aktuellen Arbeitsverzeichnis
config_path = base_dir / "notebooks" / "config1.yaml"
print(base_dir)

with open(config_path, "r") as file:
    config = yaml.safe_load(file)


def inference(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    device = next(model.parameters()).device
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Führe Inferenz mit dem SiameseUnet Modell durch")
    parser.add_argument("--user", required=True, help="Benutzername")
    parser.add_argument("--experiment_group", required=True, help="Experiment-Gruppe")
    parser.add_argument("--experiment_id", required=True, help="Experiment-ID")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch-Größe für die Inference")
    parser.add_argument("--save_results", action="store_true", help="Ergebnisse speichern")
    args = parser.parse_args()
    
    # Pfade aufbauen
    USER_HOME_PATH = Path(f"/dss/dsshome1/08/{args.user}")
    EXPERIMENT_GROUP = args.experiment_group
    EXPERIMENT_ID = args.experiment_id
    
    # Verzeichnisse für Ergebnisse erstellen
    RESULTS_DIR = USER_HOME_PATH / EXPERIMENT_GROUP / "inference_results" / EXPERIMENT_ID
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Inference-Ergebnisse werden gespeichert in: {RESULTS_DIR}")
    
    # Checkpoints-Verzeichnis
    CHECKPOINTS_DIR = USER_HOME_PATH / EXPERIMENT_GROUP / "checkpoints"
    
    # CUDA überprüfen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verwende Gerät: {device}")
    
    try:
        # Modell initialisieren
        from model.siameseNetwork import SiameseUnet  # Anpassen an deine Importstruktur
        model = SiameseUnet(num_pre_classes=2, num_post_classes=6)
        model.to(device)
        
        # Besten Checkpoint finden und laden
        from utils.helperfunctions import find_best_checkpoint, load_checkpoint  # Anpassen an deine Importstruktur
        best_checkpoint_path = find_best_checkpoint(CHECKPOINTS_DIR, EXPERIMENT_ID)
        print(f"Lade besten Checkpoint: {best_checkpoint_path}")
        model = load_checkpoint(model, best_checkpoint_path)
        
        from utils.helperfunctions import get_data_folder
        DATA_ROOT, TEST_ROOT, TEST_IMG, TEST_LABEL, TEST_TARGET, TEST_PNG_IMAGES = get_data_folder(config["data"]["test_name"], main_dataset = config["data"]["use_main_dataset"])

        # Testdaten laden
        from utils.dataset import xView2Dataset, collate_fn_test  # Anpassen an deine Importstruktur
        test_dataset = xView2Dataset(
            png_path=TEST_PNG_IMAGES,
            image_transform=image_transform(),
            inference=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,  # Verwende die Batch-Größe aus den Argumenten
            collate_fn=collate_fn_test,
            shuffle=False,  # Bei Inference nicht shuffeln
            num_workers=5
        )
                
        # Inference durchführen
        print("Starte Inference...")
        start_time = time.time()
        results = inference(model, test_dataloader)
        total_time = time.time() - start_time
        print(f"Inference abgeschlossen in {total_time:.2f} Sekunden")
        
        # Ergebnisse speichern
        if args.save_results:
            results_file = RESULTS_DIR / f"inference_results_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Ergebnisse wurden gespeichert in: {results_file}")
            
            # Speichere auch eine Textdatei mit Performance-Metriken
            metrics_file = RESULTS_DIR / f"inference_metrics_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(metrics_file, 'w') as f:
                f.write(f"Inference Performance Metriken\n")
                f.write(f"Experiment ID: {EXPERIMENT_ID}\n")
                f.write(f"Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Gesamtzeit: {total_time:.2f} Sekunden\n")
                f.write(f"Anzahl Samples: {results['performance']['total_samples']}\n")
                f.write(f"Zeit pro Sample: {results['performance']['avg_time_per_sample']:.4f} Sekunden\n\n")
                f.write("Pre-Disaster Klassenverteilung:\n")
                for cls, count in results['performance']['pre_class_distribution'].items():
                    f.write(f"  Klasse {cls}: {count} Pixel\n")
                f.write("\nPost-Disaster Klassenverteilung:\n")
                for cls, count in results['performance']['post_class_distribution'].items():
                    f.write(f"  Klasse {cls}: {count} Pixel\n")
            print(f"Metriken wurden gespeichert in: {metrics_file}")
    
    except Exception as e:
        print(f"Fehler während der Inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)