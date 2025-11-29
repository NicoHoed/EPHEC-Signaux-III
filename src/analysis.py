import numpy as np
import pytesseract
from PIL import Image
from skimage import measure
import config 
from src.preprocessing import pretraiter_vignette_ocr

def detecter_touches(img_binaire, aire_min=config.AIRE_MIN, aire_max=config.AIRE_MAX):
    label_image = measure.label(img_binaire)
    regions = measure.regionprops(label_image)
    
    touches = []
    for r in regions:
        if aire_min <= r.area <= aire_max:
            minr, minc, maxr, maxc = r.bbox
            h, w = maxr - minr, maxc - minc
            ratio = w / h if h > 0 else 0
            # On accepte large pour ne rien rater
            if 0.3 < ratio < 6.0: 
                touches.append(r)
    return touches

def lire_touches_ocr(img_gris, touches):
    resultats = []
    # Tri spatial basique pour l'ordre de traitement
    touches_triees = sorted(touches, key=lambda r: (r.centroid[0], r.centroid[1]))

    for region in touches_triees:
        minr, minc, maxr, maxc = region.bbox
        vignette = img_gris[minr:maxr, minc:maxc]
        
        # 1. Tentative Normale
        roi = pretraiter_vignette_ocr(vignette, inversion=False)
        txt = pytesseract.image_to_string(Image.fromarray(roi), config=config.OCR_CONFIG).strip()
        
        # 2. Tentative Inversée (si échec)
        if not txt or not txt.isalnum():
            roi_inv = pretraiter_vignette_ocr(vignette, inversion=True)
            txt = pytesseract.image_to_string(Image.fromarray(roi_inv), config=config.OCR_CONFIG).strip()

        # Nettoyage : On ne garde que les majuscules A-Z et les chiffres 0-9
        # On garde les chiffres pour pouvoir identifier la "ligne des chiffres" et l'ignorer
        char_clean = "".join([c for c in txt if (c.isalpha() and c.isupper()) or c.isdigit()])
        
        if len(char_clean) >= 1: 
            resultats.append({
                "char": char_clean,
                "cx": region.centroid[1],
                "cy": region.centroid[0],
                "w": maxc - minc,
                "h": maxr - minr,
                "bbox": region.bbox
            })

    return resultats

def organiser_en_lignes(ocr_results, seuil_y=30):
    """ Groupe les touches par lignes physiques """
    if not ocr_results: return []

    tous = sorted(ocr_results, key=lambda k: k['cy'])
    lignes = []
    ligne_courante = [tous[0]]
    
    for i in range(1, len(tous)):
        touche = tous[i]
        derniere = ligne_courante[-1]
        
        if abs(touche['cy'] - derniere['cy']) < seuil_y:
            ligne_courante.append(touche)
        else:
            lignes.append(sorted(ligne_courante, key=lambda k: k['cx']))
            ligne_courante = [touche]
            
    lignes.append(sorted(ligne_courante, key=lambda k: k['cx']))
    return lignes

def filtrer_ligne(ligne):
    """
    Nettoie une ligne :
    1. Retire les touches trop larges (Tab, Caps, Shift) qui polluent le début/fin.
    2. Vérifie si c'est une ligne de chiffres.
    """
    if len(ligne) < 3: return "", False # Trop court

    # Calcul largeur médiane
    largeurs = [t['w'] for t in ligne]
    mediane_w = np.median(largeurs)
    
    touches_utiles = []
    chiffres_count = 0
    
    for i, t in enumerate(ligne):
        # Si la touche est > 1.6x la médiane, c'est une touche de fonction (Tab/Caps) -> Poubelle
        if t['w'] > 1.6 * mediane_w:
            continue
            
        touches_utiles.append(t['char'])
        if t['char'].isdigit():
            chiffres_count += 1
            
    # Si plus de 40% de chiffres, c'est la ligne des chiffres -> On l'ignore pour le layout
    if len(touches_utiles) > 0 and (chiffres_count / len(touches_utiles)) > 0.4:
        return "".join(touches_utiles), True # True = est une ligne de chiffres
        
    return "".join(touches_utiles), False

def determiner_layout(ocr_results, touches_geo, img_gris):
    infos = {"Layout": "Inconnu", "Format": "ANSI", "OS": "Inconnu", "Score": 0}
    
    lignes = organiser_en_lignes(ocr_results)
    
    score_azerty = 0
    score_qwerty = 0
    sequence_visuelle = ""
    
    # --- SCAN GLOBAL DES LIGNES ---
    # On ne présume pas de l'ordre. On cherche des motifs dans chaque ligne.
    
    for idx, ligne_raw in enumerate(lignes):
        seq, est_chiffres = filtrer_ligne(ligne_raw)
        
        prefix = "[NUM]" if est_chiffres else f"[L{idx}]"
        sequence_visuelle += f"{prefix} {seq} | "
        
        if est_chiffres: continue # On saute la ligne 12345
        if len(seq) < 3: continue # Trop court pour analyser
        
        # --- RECHERCHE DE SIGNATURES ---
        
        # 1. Signature Ligne TOP (QWERTY vs AZERTY)
        # AZERTY commence par A ou Z. QWERTY commence par Q.
        if 'A' in seq[:2] and 'Z' in seq[:4]: score_azerty += 10 # AZerty
        if 'Q' in seq[:2] and 'W' in seq[:4]: score_qwerty += 10 # QWerty
        
        # 2. Signature Ligne MID (ASDF vs QSDF)
        # AZERTY commence par Q, S. QWERTY commence par A, S.
        if 'Q' in seq[:2] and 'S' in seq[:3]: score_azerty += 10 # QSdf
        if 'A' in seq[:2] and 'S' in seq[:3]: score_qwerty += 10 # ASdf
        
        # Cas spécifique : le "M". 
        # Sur AZERTY, M est à la fin de la ligne MID (droite du L).
        # Sur QWERTY, M est sur la ligne BOT.
        if seq.endswith('M') or 'M' in seq[-2:]: 
            # C'est un indice fort AZERTY, MAIS seulement si on n'a pas vu de chiffres
            # (parfois 0 looks like O or D). On reste prudent.
            score_azerty += 5

        # 3. Signature Ligne BOT (WXCV vs ZXCV)
        # C'est souvent le Juge de Paix.
        # AZERTY : W, X, C, V
        # QWERTY : Z, X, C, V
        if 'W' in seq[:2] and 'X' in seq[:4]: score_azerty += 15 # Wxcv -> AZERTY (Indice Critique)
        if 'Z' in seq[:2] and 'X' in seq[:4]: score_qwerty += 15 # Zxcv -> QWERTY (Indice Critique)

    # --- DECISION LAYOUT ---
    if score_azerty > score_qwerty:
        infos["Layout"] = "AZERTY"
    elif score_qwerty > score_azerty:
        infos["Layout"] = "QWERTY" # (Simplification, QWERTZ possible si Z/Y inversés)
        
    infos["Score"] = max(score_azerty, score_qwerty)

    # --- DECISION FORMAT (ISO/ANSI) ---
    # On regarde la touche Entrée (grosse touche à droite)
    touches_droite = sorted(touches_geo, key=lambda r: r.centroid[1], reverse=True)[:5]
    for t in touches_droite:
        h, w = t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1]
        if t.area > 2000 and w > 0:
            if h/w > 1.1:
                infos["Format"] = "ISO (Europe)" # Haute
                break
            elif h/w < 0.9:
                infos["Format"] = "ANSI (USA)" # Large
                
    # --- DECISION OS ---
    full_text = "".join([x['char'] for x in ocr_results])
    if any(k in full_text for k in ["CMD", "MAND", "OPT"]): 
        infos["OS"] = "Mac OS"
    elif any(k in full_text for k in ["CTRL", "STRG", "WIN", "ALTGR"]): 
        infos["OS"] = "Windows"
    elif infos["Layout"] == "AZERTY": 
        infos["OS"] = "Windows (Probable)" # AZERTY Mac est rare hors France spécifique

    return infos, sequence_visuelle