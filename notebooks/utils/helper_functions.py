import os
def iterate_through_dir(directory_path):
    for directory_path, directorynames, filenames in os.walk(directory_path):
        print(f"There are {len(directorynames)} directories and {len(filenames)} images in '{directory_path}'.")

from pathlib import Path

def get_data_folders(dataset_name, USER):
    """
    Returns a dictionary with important data folders for a given dataset split.
    
    Args:
        dataset_name (str): Name of the dataset split (e.g., 'test', 'tier1', 'tier3', 'hold')
    
    Returns:
        dict: Contains paths for 'data', 'images', 'labels', 'targets', 'png'
    """

        # Basis-Pfade

    ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")
    USER_PATH = ROOT / f"users/{USER}"
    DATA_PATH = ROOT / "data"
    DATASET_ROOT = DATA_PATH / "xview2"
    data_folder = DATASET_ROOT / dataset_name
    return {
        "data": data_folder,
        "images": data_folder / "images/",
        "labels": data_folder / "labels/",
        "targets": data_folder / "targets/",
        "png_images": data_folder / "png_images/",
    }
