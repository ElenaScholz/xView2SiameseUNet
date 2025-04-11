
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
