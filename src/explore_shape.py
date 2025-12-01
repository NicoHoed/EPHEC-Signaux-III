import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from skimage import io
from src.preprocessing import pretraiter_image
import config
import cv2

# ---------------------------------------
#  CONFIGURATION
# ---------------------------------------
CONFIG = {
    'IMAGE_PATH': config.IMAGE_PATH_DEFAULT,
    'AIRE_MIN': config.AIRE_MIN,
    'AIRE_MAX': config.AIRE_MAX,
    'RATIO_MIN': config.RATIO_MIN,
    'RATIO_MAX': config.RATIO_MAX,
    'SEUIL_Y_PROXIMITE': config.SEUIL_Y_PROXIMITE
}

# ---------------------------------------
#  FakeRegion (compatible avec regionprops)
# ---------------------------------------

class FakeRegion:
    """
    Objet simple imitant les propriétés utiles de skimage.regionprops,
    mais créé à partir des contours OpenCV.
    """
    def __init__(self, contour, x, y, w, h, area):
        self.contour = contour
        self.bbox = (y, x, y + h, x + w)
        self.area = area

        # Calcul du centre via moments OpenCV
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + w/2, y + h/2

        self.centroid = (cy, cx)

        # Simulation des propriétés skimage
        self.extent = area / (w*h)
        self.solidity = 1.0
        self.euler_number = 0


# ---------------------------------------
#  Détection OpenCV
# ---------------------------------------

def detecter_regions_exploration(img_binaire, config):
    """
    Détection des touches via OpenCV.
    Sortie : liste de FakeRegion + infos Y.
    """

    # Conversion en uint8
    binary_uint8 = (img_binaire * 255).astype(np.uint8)

    # Détection des contours
    contours, _ = cv2.findContours(
        binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidats = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if not (config['AIRE_MIN'] <= area <= config['AIRE_MAX']):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        ratio = w / h
        if not (config['RATIO_MIN'] <= ratio <= config['RATIO_MAX']):
            continue

        r = FakeRegion(cnt, x, y, w, h, area)
        candidats.append(r)

    if not candidats:
        return [], 0, 0, 0

    # -------- Filtrage vertical (anti-trackpad) --------
    centres_y = [r.centroid[0] for r in candidats]
    moyenne_y = np.mean(centres_y)

    y_min_valid = moyenne_y - config['SEUIL_Y_PROXIMITE']
    y_max_valid = moyenne_y + config['SEUIL_Y_PROXIMITE']

    bonnes = [r for r in candidats if y_min_valid < r.centroid[0] < y_max_valid]

    return bonnes, moyenne_y, y_min_valid, y_max_valid


# ---------------------------------------
#  Analyse individuelle des touches
# ---------------------------------------

def analyser_features(region):
    """Retourne un dictionnaire de métriques d'analyse."""

    minr, minc, maxr, maxc = region.bbox
    h = maxr - minr
    w = maxc - minc
    cy, cx = region.centroid

    centroid_y_norm = (cy - minr) / h
    ratio_h_l = h / w

    return {
        "x": int(cx),
        "y": int(cy),
        "width": w,
        "height": h,
        "minc": int(minc),
        "minr": int(minr),
        "ratio_l_h": w / h,
        "ratio_h_l": ratio_h_l,
        "extent": region.extent,
        "solidity": region.solidity,
        "euler": region.euler_number,
        "centroid_norm": centroid_y_norm
    }


# ---------------------------------------
#  Interaction par clic
# ---------------------------------------

def on_click(event, ax, regions):
    if event.inaxes != ax:
        return

    x_click, y_click = event.xdata, event.ydata

    print(f"\nClic en ({int(x_click)}, {int(y_click)})")

    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        if minc <= x_click <= maxc and minr <= y_click <= maxr:

            stats = analyser_features(r)

            print("-" * 40)
            print("TOUCHE SÉLECTIONNÉE")
            print(f"  BBox (Min C/R) : ({stats['minc']}, {stats['minr']})")
            print(f"  Centre         : ({stats['x']}, {stats['y']})")
            print(f"  Taille         : {stats['width']} × {stats['height']} px")
            print(f"  Aire           : {r.area} px")
            print(f"  Ratio L/H      : {stats['ratio_l_h']:.2f}")
            print(f"  Ratio H/L      : {stats['ratio_h_l']:.2f}")
            print(f"  Extent         : {stats['extent']:.2f}")
            print(f"  Euler          : {stats['euler']}")
            print(f"  Centre Y rel   : {stats['centroid_norm']:.2f}")
            print("-" * 40)

            # Dessine un rectangle jaune
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor='yellow',
                linewidth=3,
            )
            ax.add_patch(rect)
            event.canvas.draw()
            return


# ---------------------------------------
#  Programme principal
# ---------------------------------------

def main():
    image_path = CONFIG['IMAGE_PATH']
    print(f"Chargement de {image_path}...")

    try:
        img = io.imread(image_path)
    except Exception as e:
        print("Erreur:", e)
        return

    print("Prétraitement...")
    img_bin, img_gris = pretraiter_image(img)

    print("Détection des touches...")
    regions, mean_y, y_min, y_max = detecter_regions_exploration(img_bin, CONFIG)

    if len(regions) == 0:
        print("❌ Aucune touche détectée.")
        return

    print(f"{len(regions)} touches détectées.")
    print(f"Centre Y moyen : {mean_y:.0f}")
    print(f"Zone Y acceptée : {y_min:.0f} → {y_max:.0f}")

    # --- AFFICHAGE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(img, cmap="gray")
    ax1.set_title(f"Touches détectées ({len(regions)})")

    # Lignes horizontales
    ax1.axhline(y=mean_y, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=y_min, color='red', linestyle='-', alpha=0.5)
    ax1.axhline(y=y_max, color='red', linestyle='-', alpha=0.5)

    # Dessin des rectangles des touches
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='#00FF00',
            linewidth=1,
        )
        ax1.add_patch(rect)

    # Masque inversé affiché
    ax2.imshow(np.invert(img_bin), cmap='gray')
    ax2.set_title("Masque (inversé) utilisé pour la détection")

    print("\nCliquez sur une touche dans la fenêtre graphique pour voir ses métriques.")
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, ax1, regions))

    plt.show()


if __name__ == "__main__":
    main()
