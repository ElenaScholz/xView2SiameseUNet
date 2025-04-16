
import os
import subprocess
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import time
from IPython.display import display, HTML
import yaml

base_dir = Path(os.getcwd()).parent  # Gehe einen Ordner zurück vom aktuellen Arbeitsverzeichnis
config_path = base_dir / "notebooks" / "config1.yaml"
print(base_dir)

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
USER = config["data"]["user"]
USER_HOME_PATH = Path(f"/dss/dsshome1/08/{USER}")

# Pathes to store experiment informations in:
EXPERIMENT_GROUP = config["data"]["experiment_group"]
EXPERIMENT_ID = config["data"]["experiment_id"]

USER_HOME_PATH = Path(f"/dss/dsshome1/08/{USER}")
RESULTS_DIR = USER_HOME_PATH / EXPERIMENT_GROUP / "inference_results" / EXPERIMENT_ID
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Inference-Ergebnisse werden gespeichert in: {RESULTS_DIR}")


existing_results = list(RESULTS_DIR.glob("inference_results_*.pkl"))
if existing_results:
    print("Gefundene bestehende Inference-Ergebnisse:")
    for i, result_file in enumerate(existing_results):
        file_time = time.ctime(result_file.stat().st_mtime)
        file_size = result_file.stat().st_size / (1024 * 1024)  # MB
        print(f"{i+1}. {result_file.name} - {file_time} ({file_size:.2f} MB)")
    
    use_existing = input("\nBestehende Ergebnisse verwenden? (y/n): ").lower()
    if use_existing == 'y':
        idx = int(input("Nummer des zu verwendenden Ergebnisses eingeben: ")) - 1
        results_file = existing_results[idx]
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Ergebnisse aus {results_file} geladen.")
    else:
        # Inference-Skript ausführen
        run_inference = True
else:
    run_inference = True

if 'run_inference' in locals() and run_inference:
    print("Starte Inference-Prozess...")
    try:
        result = subprocess.run([
            "python", "inference_subprocess.py",
            "--user", USER,
            "--experiment_group", config["data"]["experiment_group"],
            "--experiment_id", config["data"]["experiment_id"],
            "--batch_size", "1",
            "--save_results"
        ], capture_output=True, text=True, check=False)  # check=False um die Exception zu vermeiden
        
        if result.returncode != 0:
            print("Fehler beim Ausführen des Inference-Skripts:")
            print(result.stderr)
        else:
            print("Inference erfolgreich abgeschlossen!")
            print(result.stdout)
    except Exception as e:
        print(f"Fehler beim Ausführen des Subprozesses: {e}")
    # try:
    #     # Du kannst hier Argumente anpassen
    #     result = subprocess.run([
    #         "python", "utils/inference_subprocess.py",
    #         "--user", USER,
    #         "--experiment_group", EXPERIMENT_GROUP,
    #         "--experiment_id", EXPERIMENT_ID,
    #         "--batch_size", "8",
    #         "--save_results"
    #     ], capture_output=True, text=True, check=True)
        
    #     print("Inference erfolgreich abgeschlossen!")
    #     print(result.stdout)
        
    #     # Suche nach der neuesten Ergebnis-Datei
    #     latest_result = max(RESULTS_DIR.glob("inference_results_*.pkl"), key=os.path.getmtime)
    #     with open(latest_result, 'rb') as f:
    #         results = pickle.load(f)
    #     print(f"Neueste Ergebnisse aus {latest_result} geladen.")
        
    # except subprocess.CalledProcessError as e:
    #     print("Fehler beim Ausführen des Inference-Skripts:")
    #     print(e.stderr)
    #     raise
