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


def extract_features(features):
    """Extract polygons, feature types, and damage classes from features."""
    return {
        'geometries': [feature['wkt'] for feature in features],
        'class_name': [feature['properties'].get('feature_type', 'unknown') for feature in features],
        'damage_class': [feature['properties'].get('subtype', 'no-damage') for feature in features]
    }

def load_label_data(label_paths):
    """Load label data from the specified paths."""
    label_data = []
    for label in label_paths:
        with open(label, "r") as file:
            label_data.append(pd.read_json(file))
    return label_data

def process_label_metadata(label):
     """Process metadata from the label."""
     metadata = label['metadata']
     return {
        'img_name': metadata['img_name'][:-4],  # Remove file extension
        'disaster': metadata['disaster'],
        'disaster_type': metadata['disaster_type']
    }

def process_features(label, damage_codes):
     """Process features from the label and apply damage codes."""
     
     feature_data = label['features']['xy']
     feature_dict = extract_features(feature_data)
    
     df = pd.DataFrame(feature_dict)
    
    # Add metadata columns to the dataframe
     metadata = process_label_metadata(label)
     for key, value in metadata.items():
        df[key] = value

    # Apply damage codes
     df['damage_code'] = df['damage_class'].apply(lambda x: damage_codes.get(x, 999))

    # Convert damage codes to integers
     df['damage_code'] = df['damage_code'].astype(int)

     return df

def make_label_dictionary(input_directory, damage_codes):
    """Create a dictionary of labels with associated metadata and damage codes."""
    label_paths = [os.path.join(input_directory, f) for f in os.listdir(input_directory)]
    label_data = load_label_data(label_paths)
    
    label_dictionary = {}
    
    for label in label_data:
        img_name = label['metadata']['img_name'][:-4]  # Remove file extension
        label_df = process_features(label, damage_codes)
        
        # Add the processed dataframe to the dictionary
        label_dictionary[img_name] = label_df

    return label_dictionary


def geotiff_converter(image_directoy: dir , output_directory: dir):

    '''
    This function takes the input geotiff images and converts them to png images 
    '''

    images = os.listdir(image_directoy) # get all image names 

        
    # Check if the directory exists
    if not os.path.exists(output_directory):
        # Create the directory if it doesn't exist
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")
    else:
        print(f"Directory '{output_directory}' already exists.")


    for i in images: # iterate over each image and open it with rasterio

        png_name = i[:-4] + ".png"

        with rio.open( image_directoy / i) as src:
            r , g , b = src.read(1), src.read(2), src.read(3)

            img = np.stack([r, g, b], axis = -1) # Stack the bands to create and np image array 

            # normalize image values:
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            png_image = Image.fromarray(img) # make it an image

            png_image.save( output_directory / png_name) # save the image

def create_disaster_targets (png_image_directory: dir,
                             label_dictionary: dict, 
                             target_output_directory: dir):
    
    
    # Check if the directory exists
    if not os.path.exists(target_output_directory):
        # Create the directory if it doesn't exist
        os.makedirs(target_output_directory)
        print(f"Directory '{target_output_directory}' created.")
    else:
        print(f"Directory '{target_output_directory}' already exists.")

    
    pngs = os.listdir(png_image_directory)


    for image_name in pngs:

        if "pre_disaster" in image_name:
            label = label_dictionary[image_name[:-4]]['geometries'] # retrieving geometries from the label
            gdf = gpd.GeoDataFrame(geometry=label.apply(wkt.loads)) # creating a geodataframe


            image = Image.open(png_image_directory / image_name) # open the corresponding image

            width,height = image.size # getting width and height information

            # Erstelle eine leere Maske (0 = Hintergrund, 1 = Gebäude/Label)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Rasterisiere die Polygone in die Maske
            shapes = [(geom, 1) for geom in gdf.geometry]  # Alle Polygone mit Wert 1 versehen
            mask = rio.features.rasterize(shapes, out_shape=(height, width))
            mask_img = Image.fromarray(mask.astype(np.uint8))
            #mask_img = Image.fromarray(mask * 255)  # Skaliere 0/1 auf 0/255 für Darstellung
            mask_img.save(target_output_directory / image_name )

        else:
            label = label_dictionary[image_name[:-4]]
            gdf_post = gpd.GeoDataFrame({
                'geometry': [wkt.loads(wkt_string) for wkt_string in label['geometries']],
                'damage_code': label['damage_code']
            })

            image = Image.open(png_image_directory / image_name) # open the corresponding image

            width,height = image.size # getting width and height information

            # Erstelle eine leere Maske (0 = Hintergrund, 1 = Gebäude/Label)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Rasterisiere die Polygone in die Maske
            shapes = [(geom, damage) for geom, damage in zip(gdf_post.geometry, gdf_post.damage_code)]
            mask = rio.features.rasterize(shapes, out_shape=(height, width))

            mask_img = Image.fromarray(mask.astype(np.uint8))

            mask_img.save(target_output_directory / image_name)

