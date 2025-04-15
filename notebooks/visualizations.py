import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image
from pathlib import Path

def plot_images(rgb_images, mask_images, figsize=(10, 6)):
    """
    Plot RGB images and their corresponding classification masks
    
    Parameters:
    -----------
    rgb_images : list of numpy arrays
        List of RGB images (2 images expected)
    mask_images : list of numpy arrays
        List of classification mask images (2 images expected)
    figsize : tuple
        Figure size
    """
    # Define a colorblind-friendly color palette for the masks
    # Using a ColorBrewer palette that is colorblind-friendly and shows progression
    colors = [
        '#000000',  # 0: Background (black)
        '#4575b4',  # 1: No damage (blue)
        '#91bfdb',  # 2: Minor damage (light blue)
        '#ffffbf',  # 3: Major damage (light yellow)
        '#fc8d59',  # 4: Destroyed (orange)
        '#d73027'   # 5: Unclassified (red)
    ]
    
    # Create a custom colormap
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create the figure and subplots with reduced spacing
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Reduce space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    
    # Custom titles
    titles = [
        "Pre-Disaster RGB Image",
        "Pre-Disaster Mask",
        "Post-Disaster RGB Image",
        "Post-Disaster Mask"
    ]
    
    # Plot RGB images on the left
    for i, rgb_img in enumerate(rgb_images):
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(titles[i*2])
        axes[i, 0].axis('off')
    
    # Plot classification masks on the right
    for i, mask_img in enumerate(mask_images):
        axes[i, 1].imshow(mask_img, cmap=cmap, norm=norm)
        axes[i, 1].set_title(titles[i*2+1])
        axes[i, 1].axis('off')
    
    # Create a legend
    legend_elements = [
        Patch(facecolor=colors[0], label='0: Background'),
        Patch(facecolor=colors[1], label='1: No Damage'),
        Patch(facecolor=colors[2], label='2: Minor Damage'),
        Patch(facecolor=colors[3], label='3: Major Damage'),
        Patch(facecolor=colors[4], label='4: Destroyed'),
        Patch(facecolor=colors[5], label='5: Unclassified')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.suptitle('Building Damage Classification', fontsize=16, y=0.98)
    plt.show()

# Load the images using the provided paths
if __name__ == "__main__":
    # Load RGB images
    rgb1 = np.array(Image.open(Path(r"C:\Users\elena\Documents\tier1\png_images\hurricane-harvey_00000177_pre_disaster.png")))
    rgb2 = np.array(Image.open(Path(r"C:\Users\elena\Documents\tier1\png_images\hurricane-harvey_00000177_post_disaster.png")))

    # Load mask images
    mask1 = np.array(Image.open(Path(r"C:\Users\elena\Documents\tier1\targets\hurricane-harvey_00000177_pre_disaster.png")))
    mask2 = np.array(Image.open(Path(r"C:\Users\elena\Documents\tier1\targets\hurricane-harvey_00000177_post_disaster.png")))

    # Plot the images
    plot_images([rgb1, rgb2], [mask1, mask2])