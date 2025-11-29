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
    Zoning mis à jour: Utilisation des multiplicateurs de config.ZONING_HR_MULTIPLIERS.
    Ajout de la détection ENTER_KEY.
    """
    if not touches: return None
    
    M = config.ZONING_HR_MULTIPLIERS # Raccourci pour le dictionnaire de multiplicateurs

    # 1. Identifier la BARRE ESPACE
    spacebar = sorted(touches, key=lambda r: r.area, reverse=True)[0]
    cy_space, cx_space = spacebar.centroid
    w_space = spacebar.bbox[3] - spacebar.bbox[1]

    # 2. Calculer la "Hauteur de Référence" (h_ref)
    hauteurs = [(r.bbox[2] - r.bbox[0]) for r in touches]
    h_ref = np.median(hauteurs)
    
    # 3. Recherche par "Couloirs" horizontaux
    # A. TOUCHE OS (Même niveau Y que l'espace)
    candidats_os = []
    for r in touches:
        if r == spacebar: continue
        cy, cx = r.centroid
        dy = cy - cy_space
        dx = cx - cx_space
        
        # Utilise M["OS_DY_TOLERANCE"] pour la tolérance verticale
        if abs(dy) < (h_ref * M["OS_DY_TOLERANCE"]) and -(w_space/2 + 250) < dx < -(w_space/2 * 0.1):
            candidats_os.append(r)
            
    touche_os = sorted(candidats_os, key=lambda r: r.centroid[1])[-1] if candidats_os else None

    # B. SHIFT GAUCHE (Rangée +1)
    candidats_shift = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space 
        
        # Utilise les multiplicateurs de config M["SHIFT_Y_MIN_HR"] et M["SHIFT_Y_MAX_HR"]
        if -(h_ref * M["SHIFT_Y_MAX_HR"]) < dy < -(h_ref * M["SHIFT_Y_MIN_HR"]) and cx < cx_space:
            candidats_shift.append(r)
            
    shift_left = sorted(candidats_shift, key=lambda r: r.area, reverse=True)[0] if candidats_shift else None

    # C. LETTRE HAUT-GAUCHE (Rangée Q/A)
    candidats_top = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space
        
        # Utilise les multiplicateurs de config M["TL_LETTER_Y_MIN_HR"] et M["TL_LETTER_Y_MAX_HR"]
        if -(h_ref * M["TL_LETTER_Y_MAX_HR"]) < dy < -(h_ref * M["TL_LETTER_Y_MIN_HR"]):
            candidats_top.append(r)
    
    top_left_key = None
    if candidats_top:
        ligne_q_triee = sorted(candidats_top, key=lambda r: r.centroid[1])
        premier = ligne_q_triee[0]
        ratio_premier = (premier.bbox[3] - premier.bbox[1]) / (premier.bbox[2] - premier.bbox[0])
        
        # Utilise config.THRESHOLD_TAB_RATIO
        if len(ligne_q_triee) > 1 and ratio_premier > config.THRESHOLD_TAB_RATIO:
            top_left_key = ligne_q_triee[1]
        else:
            top_left_key = premier
    
    # Fallback
    if not top_left_key and touches:
         top_left_key = sorted(touches, key=lambda r: r.centroid[0] + r.centroid[1])[0]

    # D. NOUVEAUTÉ: TOUCHE ENTER (Renforcement ISO/ANSI)
    y_enter_target = cy_space - (h_ref * M["ENTER_Y_TARGET_HR"])
    
    candidats_enter = []
    for r in touches:
        cy, cx = r.centroid
        
        # Utilise M["ENTER_Y_TOLERANCE_HR"]
        if abs(cy - y_enter_target) < (h_ref * M["ENTER_Y_TOLERANCE_HR"]):
            # On cherche loin à droite de la touche Espace pour isoler Enter
            if cx > (cx_space + 200): 
                candidats_enter.append(r)
                
    touche_enter = sorted(candidats_enter, key=lambda r: r.bbox[2] - r.bbox[0], reverse=True)[0] if candidats_enter else None

    # Renvoie toutes les ROIs
    return {
        "SPACE": spacebar,
        "SHIFT": shift_left,
        "TL_LETTER": top_left_key,
        "OS_KEY": touche_os,
        "ENTER_KEY": touche_enter
    }

def classifier_clavier(rois, img_gris):
    """Classification mise à jour avec les constantes de config et le renforcement ENTER."""
    resultats = {"ISO_ANSI": "?", "MAC_WIN": "?", "LAYOUT": "?"}
    debug_info = {}

    # 1. ISO vs ANSI (Shift Gauche - Verdict de base)
    if rois["SHIFT"]:
        minr, minc, maxr, maxc = rois["SHIFT"].bbox
        ratio_l_h = (maxc - minc) / (maxr - minr) # Ratio Largeur / Hauteur
        debug_info["Shift_Ratio"] = ratio_l_h
        
        # Utilise config.THRESHOLD_SHIFT_RATIO_ISO
        if ratio_l_h < config.THRESHOLD_SHIFT_RATIO_ISO: 
            resultats["ISO_ANSI"] = "ISO (Europe)"
        else: 
            resultats["ISO_ANSI"] = "ANSI (USA)"

    # 1.b. Renforcement ISO vs ANSI (Override/Confirmation: Touche Enter)
    if rois.get("ENTER_KEY"):
        r = rois["ENTER_KEY"]
        h = r.bbox[2] - r.bbox[0]
        w = r.bbox[3] - r.bbox[1]
        ratio_h_l = h / w # Ratio Hauteur / Largeur
        debug_info["Enter_Ratio_H_L"] = ratio_h_l
        
        # Utilise config.THRESHOLD_ENTER_RATIO_H_L_ANSI
        if ratio_h_l < config.THRESHOLD_ENTER_RATIO_H_L_ANSI:
             resultats["ISO_ANSI"] = "ANSI (USA) [Conf. Enter]"
        # Utilise config.THRESHOLD_ENTER_RATIO_H_L_ISO
        elif ratio_h_l > config.THRESHOLD_ENTER_RATIO_H_L_ISO:
             resultats["ISO_ANSI"] = "ISO (Europe) [Conf. Enter]"

    # 2. Mac vs Windows (Euler)
    if rois["OS_KEY"]:
        r = rois["OS_KEY"]
        minr, minc, maxr, maxc = r.bbox
        vignette = img_gris[minr:maxr, minc:maxc]
        thresh = filters.threshold_otsu(vignette)
        vignette_bin = vignette < thresh
        euler = measure.euler_number(vignette_bin, connectivity=2)
        debug_info["OS_Euler"] = euler
        
        # Utilise config.THRESHOLD_EULER_MAC/WIN
        if euler <= config.THRESHOLD_EULER_MAC: 
            resultats["MAC_WIN"] = "Mac OS"
        elif euler >= config.THRESHOLD_EULER_WIN: 
            resultats["MAC_WIN"] = "Windows/PC"
        else: 
            resultats["MAC_WIN"] = "Incertain (Mac prob.)"

    # 3. AZERTY vs QWERTY (Lettre Haut-Gauche)
    if rois["TL_LETTER"]:
        r = rois["TL_LETTER"]
        minr, _, maxr, _ = r.bbox
        cy_norm = (r.centroid[0] - minr) / (maxr - minr)
        extent = r.extent
        debug_info["TL_CenterY"] = cy_norm
        debug_info["TL_Extent"] = extent
        
        # Utilise config.THRESHOLD_TL_CENTER_Y_AZERTY/EXTENT_AZERTY
        if cy_norm > config.THRESHOLD_TL_CENTER_Y_AZERTY and extent < config.THRESHOLD_TL_EXTENT_AZERTY:
            resultats["LAYOUT"] = "AZERTY"
        else:
            resultats["LAYOUT"] = "QWERTY"

    return resultats, debug_info