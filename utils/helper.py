import os

def iterate_through_dir(directory_path):
    for directory_path, directorynames, filenames in os.walk(directory_path):
        print(f"There are {len(directorynames)} directories and {len(filenames)} images in '{directory_path}'.")