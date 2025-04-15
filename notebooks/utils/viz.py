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
    from matplotlib.colors import ListedColormap
    
    # Erstelle Ausgabeverzeichnis, falls es nicht existiert
    os.makedirs(save_dir, exist_ok=True)
    
    # Definiere Farbkarten und Labels
    pre_colors = ['#1a9850', '#d73027']  # Grün für "kein Gebäude", Rot für "Gebäude"
    pre_cmap = ListedColormap(pre_colors)
    pre_labels = ['Kein Gebäude', 'Gebäude']
    
    # Post-Disaster Farbkarte mit verschiedenen Farben für Schadensklassen
    post_colors = ['#1a9850',  # Kein Gebäude (Grün)
                  '#d73027',   # Gebäude, kein Schaden (Rot)
                  '#fc8d59',   # Leichter Schaden (Orange)
                  '#fee090',   # Mittlerer Schaden (Gelb)
                  '#91bfdb',   # Schwerer Schaden (Hellblau)
                  '#4575b4']   # Zerstört (Dunkelblau)
    post_cmap = ListedColormap(post_colors)
    post_labels = ['Kein Gebäude', 'Kein Schaden', 'Leichter Schaden', 
                   'Mittlerer Schaden', 'Schwerer Schaden', 'Zerstört']
    
    # Zufallsgenerator für reproduzierbare Ergebnisse
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    # Bestimme die Anzahl der vorhandenen Samples
    total_available = len(results['pre_names'])
    num_to_visualize = min(num_samples, total_available)
    
    # Wähle zufällige Indizes aus
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
        
        # Normalisiere Bilder für die Anzeige
        pre_img = pre_img.transpose(1, 2, 0)  # CHW -> HWC
        pre_img = (pre_img - pre_img.min()) / (pre_img.max() - pre_img.min() + 1e-8)
        
        post_img = post_img.transpose(1, 2, 0)  # CHW -> HWC
        post_img = (post_img - post_img.min()) / (post_img.max() - post_img.min() + 1e-8)
        
        # Erstelle Plot
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Zeige Pre-Disaster Bild und Vorhersage
        axs[0, 0].imshow(pre_img)
        axs[0, 0].set_title(f"Pre-Disaster Bild\n{pre_name}")
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(pre_mask, cmap=pre_cmap, vmin=0, vmax=len(pre_colors)-1)
        axs[0, 1].set_title("Pre-Disaster Vorhersage")
        axs[0, 1].axis('off')
        
        # Zeige Post-Disaster Bild und Vorhersage
        axs[1, 0].imshow(post_img)
        axs[1, 0].set_title(f"Post-Disaster Bild\n{post_name}")
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(post_mask, cmap=post_cmap, vmin=0, vmax=len(post_colors)-1)
        axs[1, 1].set_title("Post-Disaster Vorhersage")
        axs[1, 1].axis('off')
        
        # Erstelle Legenden
        axs[0, 2].axis('off')
        for j, (color, label) in enumerate(zip(pre_colors, pre_labels)):
            axs[0, 2].add_patch(plt.Rectangle((0.1, 0.9 - j*0.1), 0.1, 0.05, color=color))
            axs[0, 2].text(0.25, 0.9 - j*0.1, label, fontsize=12, va='center')
        axs[0, 2].set_title("Pre-Disaster Klassen")
        
        axs[1, 2].axis('off')
        for j, (color, label) in enumerate(zip(post_colors, post_labels)):
            axs[1, 2].add_patch(plt.Rectangle((0.1, 0.9 - j*0.1), 0.1, 0.05, color=color))
            axs[1, 2].text(0.25, 0.9 - j*0.1, label, fontsize=12, va='center')
        axs[1, 2].set_title("Post-Disaster Klassen")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"prediction_{i+1}.png"), dpi=150)
        plt.close(fig)
    
    print(f"{num_to_visualize} Visualisierungen wurden im Verzeichnis '{save_dir}' gespeichert.")
    
    # Erstelle zusätzlich eine Übersichtsgrafik der Klassenverteilung
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Konvertiere alle Vorhersagen zu einem großen Array
    all_pre_preds = np.concatenate(results['pre_predictions'])
    all_post_preds = np.concatenate(results['post_predictions'])
    
    # Zähle die Klassenhäufigkeiten
    pre_classes, pre_counts = np.unique(all_pre_preds, return_counts=True)
    post_classes, post_counts = np.unique(all_post_preds, return_counts=True)
    
    # Fülle fehlende Klassen mit 0 auf
    full_pre_counts = np.zeros(len(pre_labels))
    full_post_counts = np.zeros(len(post_labels))
    
    for cls, count in zip(pre_classes, pre_counts):
        if 0 <= cls < len(pre_labels):
            full_pre_counts[cls] = count
            
    for cls, count in zip(post_classes, post_counts):
        if 0 <= cls < len(post_labels):
            full_post_counts[cls] = count
    
    # Plotte Histogramme
    ax1.bar(range(len(pre_labels)), full_pre_counts, color=pre_colors)
    ax1.set_xticks(range(len(pre_labels)))
    ax1.set_xticklabels(pre_labels, rotation=45, ha='right')
    ax1.set_title('Pre-Disaster Klassenverteilung')
    ax1.set_ylabel('Anzahl Pixel')
    
    ax2.bar(range(len(post_labels)), full_post_counts, color=post_colors)
    ax2.set_xticks(range(len(post_labels)))
    ax2.set_xticklabels(post_labels, rotation=45, ha='right')
    ax2.set_title('Post-Disaster Klassenverteilung')
    ax2.set_ylabel('Anzahl Pixel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    print(f"Klassenverteilungen wurden in '{save_dir}/class_distribution.png' gespeichert.")

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import ListedColormap
    
    # Erstelle Ausgabeverzeichnis, falls es nicht existiert
    os.makedirs(save_dir, exist_ok=True)
    
    # Definiere Farbkarten und Labels
    pre_colors = ['#1a9850', '#d73027']  # Grün für "kein Gebäude", Rot für "Gebäude"
    pre_cmap = ListedColormap(pre_colors)
    pre_labels = ['Kein Gebäude', 'Gebäude']
    
    # Post-Disaster Farbkarte mit verschiedenen Farben für Schadensklassen
    post_colors = ['#1a9850',  # Kein Gebäude (Grün)
                  '#d73027',   # Gebäude, kein Schaden (Rot)
                  '#fc8d59',   # Leichter Schaden (Orange)
                  '#fee090',   # Mittlerer Schaden (Gelb)
                  '#91bfdb',   # Schwerer Schaden (Hellblau)
                  '#4575b4']   # Zerstört (Dunkelblau)
    post_cmap = ListedColormap(post_colors)
    post_labels = ['Kein Gebäude', 'Kein Schaden', 'Leichter Schaden', 
                   'Mittlerer Schaden', 'Schwerer Schaden', 'Zerstört']
    
    # Zufallsgenerator für reproduzierbare Ergebnisse
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    # Bestimme die Anzahl der vorhandenen Samples
    total_available = len(results['pre_names'])
    num_to_visualize = min(num_samples, total_available)
    
    # Wähle zufällige Indizes aus
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
        
        # Normalisiere Bilder für die Anzeige
        pre_img = pre_img.transpose(1, 2, 0)  # CHW -> HWC
        pre_img = (pre_img - pre_img.min()) / (pre_img.max() - pre_img.min() + 1e-8)
        
        post_img = post_img.transpose(1, 2, 0)  # CHW -> HWC
        post_img = (post_img - post_img.min()) / (post_img.max() - post_img.min() + 1e-8)
        
        # Erstelle Plot
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Zeige Pre-Disaster Bild und Vorhersage
        axs[0, 0].imshow(pre_img)
        axs[0, 0].set_title(f"Pre-Disaster Bild\n{pre_name}")
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(pre_mask, cmap=pre_cmap, vmin=0, vmax=len(pre_colors)-1)
        axs[0, 1].set_title("Pre-Disaster Vorhersage")
        axs[0, 1].axis('off')
        
        # Zeige Post-Disaster Bild und Vorhersage
        axs[1, 0].imshow(post_img)
        axs[1, 0].set_title(f"Post-Disaster Bild\n{post_name}")
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(post_mask, cmap=post_cmap, vmin=0, vmax=len(post_colors)-1)
        axs[1, 1].set_title("Post-Disaster Vorhersage")
        axs[1, 1].axis('off')
        
        # Erstelle Legenden
        axs[0, 2].axis('off')
        for j, (color, label) in enumerate(zip(pre_colors, pre_labels)):
            axs[0, 2].add_patch(plt.Rectangle((0.1, 0.9 - j*0.1), 0.1, 0.05, color=color))
            axs[0, 2].text(0.25, 0.9 - j*0.1, label, fontsize=12, va='center')
        axs[0, 2].set_title("Pre-Disaster Klassen")
        
        axs[1, 2].axis('off')
        for j, (color, label) in enumerate(zip(post_colors, post_labels)):
            axs[1, 2].add_patch(plt.Rectangle((0.1, 0.9 - j*0.1), 0.1, 0.05, color=color))
            axs[1, 2].text(0.25, 0.9 - j*0.1, label, fontsize=12, va='center')
        axs[1, 2].set_title("Post-Disaster Klassen")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"prediction_{i+1}.png"), dpi=150)
        plt.close(fig)
    
    print(f"{num_to_visualize} Visualisierungen wurden im Verzeichnis '{save_dir}' gespeichert.")
    
    # Erstelle zusätzlich eine Übersichtsgrafik der Klassenverteilung
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Konvertiere alle Vorhersagen zu einem großen Array
    all_pre_preds = np.concatenate(results['pre_predictions'])
    all_post_preds = np.concatenate(results['post_predictions'])
    
    # Zähle die Klassenhäufigkeiten
    pre_classes, pre_counts = np.unique(all_pre_preds, return_counts=True)
    post_classes, post_counts = np.unique(all_post_preds, return_counts=True)
    
    # Fülle fehlende Klassen mit 0 auf
    full_pre_counts = np.zeros(len(pre_labels))
    full_post_counts = np.zeros(len(post_labels))
    
    for cls, count in zip(pre_classes, pre_counts):
        if 0 <= cls < len(pre_labels):
            full_pre_counts[cls] = count
            
    for cls, count in zip(post_classes, post_counts):
        if 0 <= cls < len(post_labels):
            full_post_counts[cls] = count
    
    # Plotte Histogramme
    ax1.bar(range(len(pre_labels)), full_pre_counts, color=pre_colors)
    ax1.set_xticks(range(len(pre_labels)))
    ax1.set_xticklabels(pre_labels, rotation=45, ha='right')
    ax1.set_title('Pre-Disaster Klassenverteilung')
    ax1.set_ylabel('Anzahl Pixel')
    
    ax2.bar(range(len(post_labels)), full_post_counts, color=post_colors)
    ax2.set_xticks(range(len(post_labels)))
    ax2.set_xticklabels(post_labels, rotation=45, ha='right')
    ax2.set_title('Post-Disaster Klassenverteilung')
    ax2.set_ylabel('Anzahl Pixel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    print(f"Klassenverteilungen wurden in '{save_dir}/class_distribution.png' gespeichert.")