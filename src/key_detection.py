import matplotlib.pyplot as plt
from skimage import measure, color
import numpy as np


def detecter_touches(img_binaire, aire_min=100, aire_max=5000, afficher=False):
    """
    Détecte les touches individuelles d’un clavier à partir d’une image binaire.

    Paramètres :
    ----------
    img_binaire : ndarray (booléen)
        Image binaire issue du prétraitement.
    aire_min : int, optionnel
        Aire minimale d’une touche (en pixels) pour éliminer le bruit.
    aire_max : int, optionnel
        Aire maximale d’une touche (en pixels) pour ignorer les grands objets.
    afficher : bool, optionnel
        Si True, affiche les rectangles englobants sur l’image.

    Retour :
    -------
    touches : list[dict]
        Liste de dictionnaires, chaque entrée contenant :
        {
            "bbox": (minr, minc, maxr, maxc),
            "aire": float,
            "ratio": float  # rapport largeur/hauteur (utile pour filtrer les touches)
        }
    """

    # Vérifier que l'image est bien 2D
    if img_binaire.ndim != 2:
        raise ValueError("L'image binaire doit être en 2D (une seule couche).")

    # Étiqueter les composantes connexes
    etiquettes = measure.label(img_binaire)
    regions = measure.regionprops(etiquettes)

    touches = []
    for region in regions:
        if aire_min < region.area < aire_max:
            minr, minc, maxr, maxc = region.bbox
            hauteur = maxr - minr
            largeur = maxc - minc
            ratio = largeur / hauteur if hauteur > 0 else 0

            # Filtrer un peu par forme (facultatif)
            if 0.5 < ratio < 3.5:  # touches approximativement rectangulaires
                touches.append(
                    {
                        "bbox": (minr, minc, maxr, maxc),
                        "aire": region.area,
                        "ratio": ratio,
                    }
                )

    # --- Affichage optionnel ---
    if afficher:
        img_affiche = img_binaire.astype(np.float32)

        # Conversion en RGB seulement si l'image est 2D
        img_rgb = color.gray2rgb(img_affiche)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_rgb, cmap="gray")

        for t in touches:
            minr, minc, maxr, maxc = t["bbox"]
            rect = plt.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                edgecolor="red",
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_patch(rect)

        ax.set_title(f"{len(touches)} touches détectées")
        ax.axis("off")
        plt.show()

    return touches
