import numpy as np
import matplotlib.pyplot as plt
# TODO Write the function in a way, that it plots transformed and not transformed images (to compare)
def create_Control_plots(Dataset_Sample):
    pre_img, post_img, pre_target, post_target = Dataset_Sample

    # Konvertiere die Tensoren zurück zu NumPy-Arrays für die Anzeige
    pre_img_np = pre_img.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
    post_img_np = post_img.permute(1, 2, 0).numpy()
    pre_mask_np = pre_target.squeeze().numpy()  # (1, H, W) → (H, W)
    post_mask_np = post_target.squeeze().numpy()

    # Skaliere die Bilder zurück auf den Bereich [0, 255]
    pre_img_np = pre_img_np * 255.0
    post_img_np = post_img_np * 255.0

    # Erstelle eine Liste der Farben für jede Klasse
    colors = {
        0: 'white',       # Background
        1: 'green',       # No damage
        2: 'yellow',      # Minor damage
        3: 'orange',      # Major damage
        4: 'red'          # Destroyed
    }

    # Erstelle ein 2x2-Subplot für die Bilder
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Visualisiere die Bilder
    axes[0, 0].imshow(pre_img_np.astype(np.uint8))  # Achte darauf, dass die Daten im richtigen Bereich sind
    axes[0, 0].set_title("Pre-Disaster Image")
    axes[0, 1].imshow(post_img_np.astype(np.uint8))  # Achte darauf, dass die Daten im richtigen Bereich sind
    axes[0, 1].set_title("Post-Disaster Image")
    axes[1, 0].imshow(pre_mask_np, cmap='gray')  # Pre-mask mit Graustufen
    axes[1, 0].set_title("Pre-Mask")
    axes[1, 1].imshow(post_mask_np, cmap=plt.cm.colors.ListedColormap([colors[0], colors[1], colors[2], colors[3], colors[4]]))
    axes[1, 1].set_title("Post-Mask")

    # Erstelle eine zusätzliche Achse für die Legende
    fig.subplots_adjust(bottom=0.2)  # Platz für die Legende unten schaffen
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])  # [left, bottom, width, height] in fraction der Figur

    # Füge die Legende hinzu (horizontal)
    cbar = plt.colorbar(axes[1, 1].imshow(post_mask_np, cmap=plt.cm.colors.ListedColormap([colors[0], colors[1], colors[2], colors[3], colors[4]])),
                        cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Damage Class')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Background', 'Building/No Damage', 'Minor Damage', 'Major Damage', 'Destroyed'])

    # Anzeige der Plots
    #plt.tight_layout()
    plt.show()
def visualize_predictions(results, num_samples=1, random_seed=None, save_dir='predictions_visualizations'):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define colorblind-friendly color maps and labels
    # Using a consistent ColorBrewer palette
    pre_colors = ['#000000', '#4575b4']  # Black for "no building", Blue for "building"
    pre_cmap = ListedColormap(pre_colors)
    pre_bounds = [0, 1, 2]
    pre_norm = BoundaryNorm(pre_bounds, pre_cmap.N)
    
    # Post-Disaster color map with different colors for damage classes
    post_colors = ['#000000',   # Background (black)
                  '#4575b4',    # No damage (blue)
                  '#91bfdb',    # Minor damage (light blue)
                  '#ffffbf',    # Major damage (light yellow)
                  '#fc8d59',    # Destroyed (orange)
                  '#004D40']    # Unclassified (green)
    post_cmap = ListedColormap(post_colors)
    post_labels = ['No Building', 'Building / No Damage', 'Minor Damage', 
                   'Major Damage', 'Destroyed', 'Unclassified']
    post_bounds = [0, 1, 2, 3, 4, 5, 6]
    post_norm = BoundaryNorm(post_bounds, post_cmap.N)
    
    # Random generator for reproducible results
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    # Determine the number of available samples
    total_available = len(results['pre_names'])
    num_to_visualize = min(num_samples, total_available)
    
    # Select random indices
    if num_samples >= total_available:
        selected_indices = np.arange(total_available)
    else:
        selected_indices = rng.choice(total_available, num_to_visualize, replace=False)
    
    for i, idx in enumerate(selected_indices):
        pre_img = results['pre_images'][idx]
        post_img = results['post_images'][idx]
        pre_mask = results['pre_predictions'][idx]
        post_mask = results['post_predictions'][idx]
        pre_name = results['pre_names'][idx]
        post_name = results['post_names'][idx]
        
        # Normalize images for display
        pre_img = pre_img.transpose(1, 2, 0)  # CHW -> HWC
        pre_img = (pre_img - pre_img.min()) / (pre_img.max() - pre_img.min() + 1e-8)
        
        post_img = post_img.transpose(1, 2, 0)  # CHW -> HWC
        post_img = (post_img - post_img.min()) / (post_img.max() - post_img.min() + 1e-8)
        
        # Create plot
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        
        # Show Pre-Disaster image and prediction
        axs[0, 0].imshow(pre_img)
        axs[0, 0].set_title(f"Pre-Disaster Image\n{pre_name}")
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(pre_mask, cmap=pre_cmap, norm=pre_norm)
        axs[0, 1].set_title("Pre-Disaster Mask")
        axs[0, 1].axis('off')
        
        # Show Post-Disaster image and prediction
        axs[1, 0].imshow(post_img)
        axs[1, 0].set_title(f"Post-Disaster Image\n{post_name}")
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(post_mask, cmap=post_cmap, norm=post_norm)
        axs[1, 1].set_title("Post-Disaster Mask")
        axs[1, 1].axis('off')
        
        # Create legend - only for post-disaster classification
        legend_elements = [Patch(facecolor=post_colors[j], label=label) for j, label in enumerate(post_labels)]
        
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  ncol=3, 
                  bbox_to_anchor=(0.5, -0.01),
                  title="Damage Classes")
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.suptitle('Building Damage Classification', fontsize=16, y=0.98)
        plt.savefig(os.path.join(save_dir, f"prediction_{i+1}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"{num_to_visualize} visualizations saved in '{save_dir}' directory.")
    
    # Create a summary graphic of class distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert all predictions to a large array
    all_pre_preds = np.concatenate(results['pre_predictions'])
    all_post_preds = np.concatenate(results['post_predictions'])
    
    # Count class frequencies
    pre_classes, pre_counts = np.unique(all_pre_preds, return_counts=True)
    post_classes, post_counts = np.unique(all_post_preds, return_counts=True)
    
    # Fill missing classes with 0
    full_pre_counts = np.zeros(2)  # Only 2 classes for pre-disaster
    full_post_counts = np.zeros(len(post_labels))
    
    for cls, count in zip(pre_classes, pre_counts):
        if 0 <= cls < 2:
            full_pre_counts[cls] = count
            
    for cls, count in zip(post_classes, post_counts):
        if 0 <= cls < len(post_labels):
            full_post_counts[cls] = count
    
    # Plot histograms
    ax1.bar(range(2), full_pre_counts, color=pre_colors)
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(['No Building', 'Building'], rotation=45, ha='right')
    ax1.set_title('Pre-Disaster Class Distribution')
    ax1.set_ylabel('Pixel Count')
    
    ax2.bar(range(len(post_labels)), full_post_counts, color=post_colors)
    ax2.set_xticks(range(len(post_labels)))
    ax2.set_xticklabels(post_labels, rotation=45, ha='right')
    ax2.set_title('Post-Disaster Class Distribution')
    ax2.set_ylabel('Pixel Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    print(f"Class distribution saved in '{save_dir}/class_distribution.png'.")