import numpy as np
import math
from skimage import measure, filters
import config 

def detecter_touches(img_binaire_inversee, aire_min=config.AIRE_MIN, aire_max=config.AIRE_MAX, ratio_max=config.RATIO_MAX, seuil_y=config.SEUIL_Y_PROXIMITE):
    """Extrait les régions candidates et filtre spatialement, utilisant les valeurs de config par défaut."""
    label_image = measure.label(img_binaire_inversee)
    regions = measure.regionprops(label_image)
    
    candidats = []
    for r in regions:
        if aire_min <= r.area <= aire_max:
            minr, minc, maxr, maxc = r.bbox
            h = maxr - minr
            w = maxc - minc
            if h > 0 and (w / h) <= ratio_max:
                candidats.append(r)
    
    if not candidats:
        return [], 0, 0, 0

    # Filtrage Spatial (Centré sur la masse du clavier)
    centres_y = [r.centroid[0] for r in candidats]
    moyenne_y = np.mean(centres_y)
    
    y_min = moyenne_y - seuil_y
    y_max = moyenne_y + seuil_y
    
    touches_finales = [r for r in candidats if y_min < r.centroid[0] < y_max]
    
    return touches_finales, moyenne_y, y_min, y_max

def identifier_zones_cles(touches):
    """
    Zoning V8 (Basé sur les rangées) :
    Utilise la hauteur médiane d'une touche (h_ref) pour naviguer ligne par ligne.
    """
    if not touches: return None

    # 1. Identifier la BARRE ESPACE
    spacebar = sorted(touches, key=lambda r: r.area, reverse=True)[0]
    cy_space, cx_space = spacebar.centroid
    w_space = spacebar.bbox[3] - spacebar.bbox[1]

    # 2. Calculer la "Hauteur de Référence" (h_ref)
    # On prend la médiane des hauteurs de toutes les touches (pour exclure l'espace ou les erreurs)
    hauteurs = [(r.bbox[2] - r.bbox[0]) for r in touches]
    h_ref = np.median(hauteurs)
    
    print(f"📍 Espace: Y={cy_space:.0f} | Hauteur standard touche (h_ref): {h_ref:.0f}px")

    # 3. Recherche par "Couloirs" horizontaux
    
    # A. TOUCHE OS (Même niveau Y que l'espace)
    # Zone : dy < 0.8 * h_ref (tolérance fine car même ligne)
    candidats_os = []
    for r in touches:
        if r == spacebar: continue
        cy, cx = r.centroid
        dy = cy - cy_space
        dx = cx - cx_space
        
        # Même ligne Y, et à gauche de l'espace
        if abs(dy) < (h_ref * 0.8) and -(w_space/2 + 250) < dx < -(w_space/2 * 0.1):
            candidats_os.append(r)
            
    # On prend le plus proche en X (le plus grand X parmi les négatifs)
    touche_os = sorted(candidats_os, key=lambda r: r.centroid[1])[-1] if candidats_os else None

    # B. SHIFT GAUCHE (Rangée +1)
    # Zone : Entre 0.5 et 1.8 hauteurs au-dessus de l'espace.
    # Cela exclut Caps Lock qui est à ~2.2 hauteurs au-dessus.
    candidats_shift = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space 
        
        # On cherche dans la "Rangée 1" au-dessus
        if -(h_ref * 2.0) < dy < -(h_ref * 0.5) and cx < cx_space:
            candidats_shift.append(r)
            
    # Le Shift est la plus grosse touche de cette rangée
    shift_left = sorted(candidats_shift, key=lambda r: r.area, reverse=True)[0] if candidats_shift else None

    # C. LETTRE HAUT-GAUCHE (Rangée +3 : Ligne QWERTY)
    # Zone : Entre 2.5 et 3.8 hauteurs au-dessus de l'espace.
    # Row 0 = Space, Row 1 = Shift, Row 2 = Caps/A, Row 3 = Tab/Q
    candidats_top = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space
        
        # On vise la 3ème rangée (Q/A)
        if -(h_ref * 3.8) < dy < -(h_ref * 2.5):
            candidats_top.append(r)
    
    top_left_key = None
    if candidats_top:
        # On trie de gauche à droite (X croissant)
        ligne_q_triee = sorted(candidats_top, key=lambda r: r.centroid[1])
        
        # Le premier est souvent TAB. Le deuxième est Q (ou A).
        # Comment distinguer Tab de Q ? Tab est rectangulaire (ratio > 1.2)
        premier = ligne_q_triee[0]
        ratio_premier = (premier.bbox[3] - premier.bbox[1]) / (premier.bbox[2] - premier.bbox[0])
        
        if len(ligne_q_triee) > 1 and ratio_premier > 1.2:
            # C'était Tab, on prend le suivant (le Q/A)
            top_left_key = ligne_q_triee[1]
            print("ℹ️  Touche 'Tab' détectée et ignorée, sélection de la suivante (Q/A).")
        else:
            # C'était probablement déjà Q/A (ou Tab non détecté)
            top_left_key = premier
    
    # Fallback : Si on n'a pas trouvé la ligne Q, on prend la touche la plus "Nord-Ouest" globale
    if not top_left_key and touches:
         top_left_key = sorted(touches, key=lambda r: r.centroid[0] + r.centroid[1])[0]

    return {
        "SPACE": spacebar,
        "SHIFT": shift_left,
        "TL_LETTER": top_left_key,
        "OS_KEY": touche_os
    }

def classifier_clavier(rois, img_gris):
    """Classification inchangée."""
    resultats = {"ISO_ANSI": "?", "MAC_WIN": "?", "LAYOUT": "?"}
    debug_info = {}

    # 1. ISO vs ANSI (Shift Gauche)
    if rois["SHIFT"]:
        minr, minc, maxr, maxc = rois["SHIFT"].bbox
        ratio = (maxc - minc) / (maxr - minr)
        debug_info["Shift_Ratio"] = ratio
        
        if ratio < 2.1: resultats["ISO_ANSI"] = "ISO (Europe)"
        else: resultats["ISO_ANSI"] = "ANSI (USA)"

    # 2. Mac vs Windows (Euler Local)
    if rois["OS_KEY"]:
        r = rois["OS_KEY"]
        minr, minc, maxr, maxc = r.bbox
        vignette = img_gris[minr:maxr, minc:maxc]
        thresh = filters.threshold_otsu(vignette)
        vignette_bin = vignette < thresh
        euler = measure.euler_number(vignette_bin, connectivity=2)
        debug_info["OS_Euler"] = euler
        
        if euler <= -1: resultats["MAC_WIN"] = "Mac OS"
        elif euler >= 1: resultats["MAC_WIN"] = "Windows/PC"
        else: resultats["MAC_WIN"] = "Incertain (Mac prob.)"

    # 3. AZERTY vs QWERTY (Lettre Haut-Gauche)
    if rois["TL_LETTER"]:
        r = rois["TL_LETTER"]
        minr, _, maxr, _ = r.bbox
        cy_norm = (r.centroid[0] - minr) / (maxr - minr)
        extent = r.extent
        debug_info["TL_CenterY"] = cy_norm
        debug_info["TL_Extent"] = extent
        
        if cy_norm > 0.53 and extent < 0.80:
            resultats["LAYOUT"] = "AZERTY"
        else:
            resultats["LAYOUT"] = "QWERTY"

    return resultats, debug_info