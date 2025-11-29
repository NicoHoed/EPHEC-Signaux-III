import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
import numpy as np

from src.preprocessing import pretraiter_image
from src.analysis import detecter_touches, identifier_zones_cles, classifier_clavier
import config

# --- CONFIGURATION ---
IMAGE_PATH = config.IMAGE_PATH_DEFAULT 

def main():
    print(f"Démarrage de l'analyse sur {IMAGE_PATH}...")
    
    try:
        img = io.imread(IMAGE_PATH)
    except FileNotFoundError:
        print(f"❌ Erreur: Image non trouvée dans {IMAGE_PATH}")
        return

    # 1. Prétraitement
    print("Prétraitement...")
    # On récupère l'image binaire (pour détection) ET grise (pour analyse fine)
    img_bin, img_gris = pretraiter_image(img)

    # 2. Détection
    print("Détection des touches...")
    # PASSAGE DES PARAMÈTRES DE CONFIGURATION EXPLICITEMENT
    touches, mean_y, y_min, y_max = detecter_touches(
        img_bin,
        aire_min=config.AIRE_MIN,
        aire_max=config.AIRE_MAX,
        ratio_max=config.RATIO_MAX,
        seuil_y=config.SEUIL_Y_PROXIMITE
    )

    # 3. Identification des zones
    print("Identification des zones clés...")
    rois = identifier_zones_cles(touches)
    
    if rois is None:
        print("Impossible de repérer la structure du clavier.")
        return

    # 4. Classification
    print("Classification du layout...")
    verdict, debug = classifier_clavier(rois, img_gris)

    # --- AFFICHAGE RÉSULTATS TERMINAL ---
    print("\n" + "="*30)
    print("RÉSULTATS DE L'ANALYSE")
    print("="*30)
    print(f"Format  : {verdict['ISO_ANSI']}")
    print(f"Système : {verdict['MAC_WIN']}")
    print(f"Langue  : {verdict['LAYOUT']}")
    print("-" * 30)
    print("Données techniques :")
    for k, v in debug.items():
        # Affichage des nouvelles métriques
        print(f"   - {k} : {v:.2f}") 
    print("="*30)

    # --- VISUALISATION GRAPHIQUE ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Analyse : {verdict['LAYOUT']} - {verdict['MAC_WIN']} - {verdict['ISO_ANSI']}")

    # Dessiner les zones de recherche Y (lignes rouges)
    ax.axhline(y=y_min, color='red', linestyle='--', alpha=0.3, label='Zone de filtrage')
    ax.axhline(y=y_max, color='red', linestyle='--', alpha=0.3)

    # Couleurs pour les touches identifiées
    colors = {
        "SPACE": "blue",
        "SHIFT": "orange",
        "TL_LETTER": "green",
        "OS_KEY": "magenta",
        "ENTER_KEY": "cyan"
    }

    # Dessiner toutes les touches en vert pâle
    for r in touches:
        rect = mpatches.Rectangle((r.bbox[1], r.bbox[0]), 
                                  r.bbox[3] - r.bbox[1], 
                                  r.bbox[2] - r.bbox[0],
                                  fill=False, edgecolor='#00FF00', linewidth=1, alpha=0.3)
        ax.add_patch(rect)

    # Dessiner les ROI en gras et couleur spécifique
    detected_patches = []
    for name, region in rois.items():
        if region and name != "h_ref": # Exclure h_ref au cas où il était dans le dictionnaire
            minr, minc, maxr, maxc = region.bbox
            color_code = colors.get(name, "yellow")
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=color_code, linewidth=3, label=name)
            ax.add_patch(rect)
            
            # Ajouter le label texte
            ax.text(minc, minr - 5, name, color=color_code, fontsize=8, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    print("\nAffichage de la fenêtre graphique...")
    plt.show()

if __name__ == "__main__":
    main()