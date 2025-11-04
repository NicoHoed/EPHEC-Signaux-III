import numpy as np
from skimage import measure


def detecter_touches(img_binaire, aire_min=5000, aire_max=10000, 
                     ratio_min=1.0, ratio_max=2.0, aire_large=30000, 
                     aire_max_large=80000, seuil_y=1000):
    """
    Détecte les rectangles noirs correspondant aux touches sur une image binaire.
    
    Paramètres :
    - img_binaire : image binaire prétraitée
    - aire_min : aire minimale pour une touche normale
    - aire_max : aire maximale pour une touche
    - ratio_min : ratio largeur/hauteur minimum pour une touche
    - ratio_max : ratio largeur/hauteur maximum pour une touche
    - aire_large : aire à partir de laquelle on considère une touche comme "large" (spacebar, Enter, etc.)
    - aire_max_large : aire maximale pour les touches larges (évite les très grandes régions)
    - seuil_y : seuil de position verticale pour filtrer les artefacts
    
    Retourne : liste de tuples (minr, minc, maxr, maxc) représentant les touches
    """
    
    # Étape 1 : Inverser l'image pour que les touches noires deviennent des objets
    inversee = np.invert(img_binaire.astype(bool))
    
    # Étape 2 : Labeliser les composants connectés
    labels = measure.label(inversee)
    regions = measure.regionprops(labels)
    
    # Étape 3 : Collecte initiale des rectangles selon les critères de taille et de ratio
    boites_touches = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        largeur = maxc - minc
        hauteur = maxr - minr
        aire = region.area
        ratio = largeur / hauteur if hauteur > 0 else 0
        
        # Cas 1 : touches normales
        if aire_min <= aire <= aire_max and ratio_min <= ratio <= ratio_max:
            boites_touches.append((minr, minc, maxr, maxc))
        
        # Cas 2 : touches larges (spacebar, Enter, Maj, etc.)
        elif aire_large <= aire <= aire_max_large and ratio > 0.1:
            boites_touches.append((minr, minc, maxr, maxc))
    
    # Étape 4 : Filtrage par position verticale pour éliminer artefacts et trackpad
    if boites_touches:
        centres_y = [(minr + maxr) / 2 for minr, _, maxr, _ in boites_touches]
        moyenne_y = np.mean(centres_y)
        boites_touches = [bbox for bbox in boites_touches
                         if abs((bbox[0] + bbox[2]) / 2 - moyenne_y) < seuil_y]
    
    return boites_touches