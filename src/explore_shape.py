import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from skimage import io, measure
from src.preprocessing import pretraiter_image
import config

# Ce script permet de faire des tests et de trouver avec quelles valeurs travailler pour une photo particuliere (pixels)
# Pour l'utiliser, il faut lancer le script avec la photo souhaitee dans le fichier config.py. Les instructions sont ensuite données dans script.

# --- RÉCUPÉRATION DES PARAMÈTRES ---
CONFIG = {
    'IMAGE_PATH': config.IMAGE_PATH_DEFAULT,
    'AIRE_MIN': config.AIRE_MIN,
    'AIRE_MAX': config.AIRE_MAX,
    'RATIO_MIN': config.RATIO_MIN,
    'RATIO_MAX': config.RATIO_MAX,
    'SEUIL_Y_PROXIMITE': config.SEUIL_Y_PROXIMITE
}


def detecter_regions_exploration(img_binaire, config):
    """
    Logique de détection pour l'exploration, utilisant le dictionnaire de configuration.
    """
    # Inversion nécessaire car measure.label cherche les zones True (Blanches)
    inversee = np.invert(img_binaire.astype(bool))
    
    label_image = measure.label(inversee)
    all_regions = measure.regionprops(label_image)
    
    candidats_initiaux = []
    
    # 1. Filtrage par dimensions physiques
    for r in all_regions:
        minr, minc, maxr, maxc = r.bbox
        hauteur = maxr - minr
        largeur = maxc - minc
        
        if hauteur == 0: continue
        
        aire = r.area
        ratio = largeur / hauteur

        # Utilisation du dictionnaire de configuration
        if (config['AIRE_MIN'] <= aire <= config['AIRE_MAX'] and 
            config['RATIO_MIN'] <= ratio <= config['RATIO_MAX']):
            candidats_initiaux.append(r)

    if not candidats_initiaux:
        return [], 0, 0, 0

    # 2. Filtrage Spatial (Anti-Trackpad)
    centres_y = [r.centroid[0] for r in candidats_initiaux]
    moyenne_y = np.mean(centres_y)
    
    # On définit la zone valide, en utilisant le seuil du dictionnaire
    y_min_valid = moyenne_y - config['SEUIL_Y_PROXIMITE']
    y_max_valid = moyenne_y + config['SEUIL_Y_PROXIMITE']
    
    bonnes_regions = []
    for r in candidats_initiaux:
        cy, _ = r.centroid
        if y_min_valid < cy < y_max_valid:
            bonnes_regions.append(r)

    return bonnes_regions, moyenne_y, y_min_valid, y_max_valid


def analyser_features(region):
    """Extraction des métriques pour l'analyse de layout (avec plus de détails)"""
    minr, minc, maxr, maxc = region.bbox
    hauteur = maxr - minr
    largeur = maxc - minc
    cy, cx = region.centroid
    
    # Centre relatif (0.0 haut, 1.0 bas)
    centroid_y_norm = (cy - minr) / hauteur 
    
    # Calcul du Ratio H/L pour Enter/Shift (à des fins de debug)
    ratio_h_l = hauteur / largeur 
    
    return {
        "x": int(cx),
        "y": int(cy),
        "width": largeur,
        "height": hauteur,
        "minc": int(minc),
        "minr": int(minr),
        "ratio_l_h": largeur / hauteur,
        "ratio_h_l": ratio_h_l, # Ajouté pour Enter debug
        "extent": region.extent,
        "solidity": region.solidity,
        "euler": region.euler_number,
        "centroid_norm": centroid_y_norm
    }


def on_click(event, ax, regions):
    """Logique de clic améliorée pour afficher plus de détails et les seuils de config."""
    if event.inaxes != ax: return
    x_click, y_click = event.xdata, event.ydata
    
    print(f"\nClic en ({int(x_click)}, {int(y_click)})")
    
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        if minc <= x_click <= maxc and minr <= y_click <= maxr:
            stats = analyser_features(r)
            
            # --- Affichage amélioré ---
            print("-" * 40)
            print(f"TOUCHE SÉLECTIONNÉE")
            print(f"   Coordonnées BBox (Min C/R) : ({stats['minc']}, {stats['minr']})")
            print(f"   Centre (X, Y) : ({stats['x']}, {stats['y']}) px")
            print(f"   Dimensions (L x H) : {stats['width']} x {stats['height']} px")
            print(f"   Aire                  : {r.area} px")
            # Utilise les seuils de config.py pour le debug
            print(f"   Ratio L/H (Shift)     : {stats['ratio_l_h']:.2f} (Seuil ISO: {config.THRESHOLD_SHIFT_RATIO_ISO})")
            print(f"   Ratio H/L (Enter)     : {stats['ratio_h_l']:.2f} (Seuil ISO: {config.THRESHOLD_ENTER_RATIO_H_L_ISO})")
            print(f"   Extent                : {stats['extent']:.2f}")
            print(f"   Euler                 : {stats['euler']} (Seuil Mac: {config.THRESHOLD_EULER_MAC})")
            print(f"   Centre Y Relatif      : {stats['centroid_norm']:.2f} (Seuil AZERTY: {config.THRESHOLD_TL_CENTER_Y_AZERTY})")
            
            # Détection basique pour info
            verdict = "Inconnu"
            if stats['ratio_l_h'] > 4.0: verdict = "Espace ?"
            # Utilise les seuils de config.py pour l'hypothèse
            elif stats['ratio_l_h'] > config.THRESHOLD_SHIFT_RATIO_ISO: verdict = "Shift/Enter/Cmd ANSI ?"
            elif stats['euler'] < config.THRESHOLD_EULER_WIN: verdict = "Command (Mac) ?"
            
            print(f"   -> Hypothèse          : {verdict}")
            print("-" * 40)
            
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='yellow', linewidth=3)
            ax.add_patch(rect)
            event.canvas.draw()
            return


def main():
    image_path = CONFIG['IMAGE_PATH']
    print(f"Chargement de {image_path}...")
    try: 
        img = io.imread(image_path)
    except Exception as e: 
        print(f"Erreur: {e}")
        return

    # Utilisation de la fonction importée du module src.preprocessing
    print("Prétraitement...")
    img_bin, img_gris = pretraiter_image(img)

    # Appel de la fonction d'exploration avec la configuration
    print("Détection des touches (Paramètres explorés)...")
    # Utilise la détection avec les paramètres de la config
    regions, mean_y, y_min, y_max = detecter_regions_exploration(img_bin, CONFIG)
    
    if len(regions) == 0:
        print("❌ Aucune touche trouvée. Vérifiez le seuillage.")
        return

    # Affichage des résultats en console
    print(f"{len(regions)} touches détectées.")
    print(f"Centre Y moyen : {mean_y:.0f} px")
    print(f"Zone acceptée : Y={y_min:.0f} à Y={y_max:.0f}")

    # --- Affichage Matplotlib (inchangé) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Vue Résultats
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f"Touches détectées ({len(regions)})")
    
    # Dessin des lignes de la zone de filtrage spatial
    ax1.axhline(y=mean_y, color='blue', linestyle='--', alpha=0.5, label='Centre Moyen')
    ax1.axhline(y=y_min, color='red', linestyle='-', alpha=0.5, label='Limite Haute')
    ax1.axhline(y=y_max, color='red', linestyle='-', alpha=0.5, label='Limite Basse')
    ax1.legend()

    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='#00FF00', linewidth=1)
        ax1.add_patch(rect)

    # Vue Masque
    ax2.imshow(np.invert(img_bin), cmap='gray') 
    ax2.set_title("Masque inversé (ce que voit l'algo)")

    print("\nINFO: Les lignes rouges horizontales montrent la zone de recherche.")
    print(f"   Si des touches sont en dehors, augmentez SEUIL_Y_PROXIMITE ({CONFIG['SEUIL_Y_PROXIMITE']}).")
    print("   Cliquez sur les touches (Espace, Cmd, etc.) pour vérifier leurs stats.")
    
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, ax1, regions))
    plt.show()

if __name__ == "__main__":
    main()