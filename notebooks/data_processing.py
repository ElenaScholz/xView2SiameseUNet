from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from shapely.wkt import loads
import json
import os
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely import Polygon
import numpy as np


from utils.helper_functions import get_data_folders
from utils.preprocessing import extract_features, load_label_data, process_label_metadata, process_features, make_label_dictionary, geotiff_converter, create_disaster_targets


ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")
##########################################################
# Define your username here:
USER = "di97ren"
USER_PATH = ROOT / f"users/{USER}"



labels = os.listdir(test_folder['labels'])
print(labels)
label_paths = []

for l in labels:
    label_paths.append(os.path.join(labels / l))

label_data = []

for label in label_paths:
    with open(label, "r") as file:
        label_data.append(pd.read_json(file))

# define building damage codes
damage_codes = {
    'no-damage' : 1,
    'minor-damage' : 2,
    'major-damage' : 3,
    'destroyed' : 4,
    'un-classified' : 5
}

label_dictionary = make_label_dictionary(test_folder['labels'], damage_codes)

geotiff_converter(test_folder['images'], test_folder['png_images'])
