import os

def iterate_through_dir(directory_path):
    for directory_path, directorynames, filenames in os.walk(directory_path):
        print(f"There are {len(directorynames)} directories and {len(filenames)} images in '{directory_path}'.")

from pathlib import Path

def get_data_folder(folder_name: str,
    main_dataset: bool):  # possible names: ["test", "tier1", "tier3", "hold"] 
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

