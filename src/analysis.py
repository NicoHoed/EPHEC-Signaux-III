import numpy as np
from skimage import measure, filters
import config 
from src.ocr_classifier import classifier_layout_ocr


def detecter_touches(img_binaire_inversee, aire_min=None, aire_max=None, 
                     ratio_max=None, seuil_y=None):
    """
    Détection adaptative et robuste des touches.
    Calcule automatiquement les seuils si non fournis.
    """
    hauteur_img, largeur_img = img_binaire_inversee.shape
    resolution_totale = hauteur_img * largeur_img
    
    if aire_min is None:
        if config.AIRE_MIN_RATIO > 0:
            aire_min = int(resolution_totale * config.AIRE_MIN_RATIO)
        else:
            aire_min = config.AIRE_MIN
    
    if aire_max is None:
        if config.AIRE_MAX_RATIO > 0:
            aire_max = int(resolution_totale * config.AIRE_MAX_RATIO)
        else:
            aire_max = config.AIRE_MAX
    
    if ratio_max is None:
        ratio_max = config.RATIO_MAX
    
    if seuil_y is None:
        if config.SEUIL_Y_RATIO > 0:
            seuil_y = hauteur_img * config.SEUIL_Y_RATIO
        else:
            seuil_y = config.SEUIL_Y_PROXIMITE
    
    if config.DEBUG_MODE:
        print(f"   Seuils adaptatifs: aire=[{aire_min}, {aire_max}], ratio_max={ratio_max}, seuil_y={seuil_y:.0f}")
    
    label_image = measure.label(img_binaire_inversee)
    regions = measure.regionprops(label_image)
    
    candidats = []

    for r in regions:
        if aire_min <= r.area <= aire_max:
            minr, minc, maxr, maxc = r.bbox
            h = maxr - minr
            w = maxc - minc
            if h > 0:
                ratio = w / h
                solidite = r.solidity
                extent = r.extent

                patch = img_binaire_inversee[minr:maxr, minc:maxc]
                qualite = float(patch.var()) if patch.size > 0 else 0.0

                if (ratio <= ratio_max and
                    solidite > config.SOLIDITY_MIN and
                    extent > config.EXTENT_MIN and
                    qualite > getattr(config, "TOUCHE_QUALITE_MIN", 1e-4)):

                    r._qualite_touche = qualite
                    candidats.append(r)
    
    if not candidats:
        if config.DEBUG_MODE:
            print("   Aucun candidat trouvé après filtrage initial")
        return [], 0, 0, 0
    
    centres_y = np.array([r.centroid[0] for r in candidats])
    mediane_y = np.median(centres_y)
    ecart_type_y = np.std(centres_y)
    seuil_adaptatif = min(seuil_y, ecart_type_y * 2.5)
    
    y_min = mediane_y - seuil_adaptatif
    y_max = mediane_y + seuil_adaptatif
    
    touches_finales = [r for r in candidats if y_min < r.centroid[0] < y_max]
    
    if config.DEBUG_MODE:
        print(f"   Touches détectées: {len(touches_finales)}/{len(candidats)} candidats")
        print(f"   Zone Y valide: [{y_min:.0f}, {y_max:.0f}] (médiane={mediane_y:.0f})")
    
    return touches_finales, mediane_y, y_min, y_max


def identifier_zones_cles(touches):
    """
    Identification robuste des zones clés avec validation croisée.
    """
    if not touches or len(touches) < config.MIN_TOUCHES_DETECTEES:
        if config.DEBUG_MODE:
            print(f"   Échec: seulement {len(touches)} touches (minimum: {config.MIN_TOUCHES_DETECTEES})")
        return None
    
    M = config.ZONING_HR_MULTIPLIERS
    
    if config.DEBUG_MODE:
        print(f"   Début zoning avec {len(touches)} touches")
    
    # 1. BARRE ESPACE
    touches_triees_y = sorted(touches, key=lambda r: r.centroid[0], reverse=True)
    touches_bas = touches_triees_y[:15]
    
    if config.DEBUG_MODE:
        print(f"   Analyse des {len(touches_bas)} touches les plus basses pour trouver l'espace...")
    
    spacebar = None
    max_largeur = 0
    
    for i, candidate in enumerate(touches_bas):
        h = candidate.bbox[2] - candidate.bbox[0]
        w = candidate.bbox[3] - candidate.bbox[1]
        ratio = w / h if h > 0 else 0
        
        if config.DEBUG_MODE and i < 5:
            print(f"   Candidat #{i+1}: y={candidate.centroid[0]:.0f}, largeur={w:.0f}, ratio={ratio:.2f}")
        
        if w > max_largeur and w > 200 and ratio > 1.5:
            spacebar = candidate
            max_largeur = w
    
    if not spacebar:
        spacebar = max(touches_bas, key=lambda r: r.bbox[3] - r.bbox[1])
        if config.DEBUG_MODE:
            w = spacebar.bbox[3] - spacebar.bbox[1]
            print(f"   ⚠️ Fallback: touche la plus large du bas (w={w:.0f}px)")
    else:
        if config.DEBUG_MODE:
            print(f"   ✓ Espace identifié: y={spacebar.centroid[0]:.0f}, largeur={max_largeur:.0f}px")

    
    cy_space, cx_space = spacebar.centroid
    w_space = spacebar.bbox[3] - spacebar.bbox[1]
    
    if config.DEBUG_MODE:
        print(f"   ✓ Espace trouvé: centre=({cx_space:.0f}, {cy_space:.0f}), largeur={w_space:.0f}px")
    
    # 2. h_ref
    hauteurs = [(r.bbox[2] - r.bbox[0]) for r in touches]
    h_ref = np.percentile(hauteurs, 40)
    
    h_space = spacebar.bbox[2] - spacebar.bbox[0]
    if h_ref < 10 or h_ref > h_space * 2:
        hauteurs_filtrees = [h for h in hauteurs if 20 < h < 200]
        if hauteurs_filtrees:
            h_ref = np.median(hauteurs_filtrees)
            if config.DEBUG_MODE:
                print(f"   h_ref recalculé: {h_ref:.1f}px")
    
    if config.DEBUG_MODE:
        print(f"   h_ref: {h_ref:.1f}px (hauteur de référence)")
    
    # 3. SHIFT GAUCHE
    candidats_shift = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space
        
        if -(h_ref * M["SHIFT_Y_MAX_HR"]) < dy < -(h_ref * M["SHIFT_Y_MIN_HR"]):
            if cx < (cx_space - w_space * 0.2):
                h = r.bbox[2] - r.bbox[0]
                w = r.bbox[3] - r.bbox[1]
                ratio = w / h if h > 0 else 0
                
                if config.SHIFT_RATIO_MIN < ratio < config.SHIFT_RATIO_MAX:
                    candidats_shift.append(r)
    
    shift_left = sorted(candidats_shift, key=lambda r: r.area, reverse=True)[0] if candidats_shift else None
    
    if config.DEBUG_MODE:
        if shift_left:
            print(f"   ✓ Shift trouvé: {len(candidats_shift)} candidats")
        else:
            print(f"   ✗ Shift NON trouvé ({len(candidats_shift)} candidats)")
    
    # 4. LETTRE HAUT-GAUCHE
    candidats_top = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space
        
        if -(h_ref * M["TL_LETTER_Y_MAX_HR"]) < dy < -(h_ref * M["TL_LETTER_Y_MIN_HR"]):
            h = r.bbox[2] - r.bbox[0]
            w = r.bbox[3] - r.bbox[1]
            ratio = w / h if h > 0 else 0
            
            if config.TL_LETTER_RATIO_MIN < ratio < config.TL_LETTER_RATIO_MAX:
                candidats_top.append(r)
    
    top_left_key = None
    if candidats_top:
        ligne_triee = sorted(candidats_top, key=lambda r: r.centroid[1])
        premier = ligne_triee[0]
        ratio_premier = (premier.bbox[3] - premier.bbox[1]) / (premier.bbox[2] - premier.bbox[0])
        
        if len(ligne_triee) > 1 and ratio_premier > config.THRESHOLD_TAB_RATIO:
            top_left_key = ligne_triee[1]
        else:
            top_left_key = premier
    
    if not top_left_key and touches:
        top_left_key = sorted(touches, key=lambda r: r.centroid[0] + r.centroid[1])[0]
    
    if config.DEBUG_MODE:
        if top_left_key:
            print(f"   ✓ TL_LETTER trouvé: {len(candidats_top)} candidats")
        else:
            print(f"   ✗ TL_LETTER NON trouvé")
    
    # 5. TOUCHE OS
    candidats_os = []
    for r in touches:
        cy, cx = r.centroid
        dy = abs(cy - cy_space)
        dx = cx - cx_space
        
        if dy < (h_ref * M["OS_DY_TOLERANCE"]):
            if -(w_space * 0.7) < dx < -(w_space * 0.05):
                h = r.bbox[2] - r.bbox[0]
                w = r.bbox[3] - r.bbox[1]
                ratio = w / h if h > 0 else 0
                
                if config.OS_KEY_RATIO_MIN < ratio < config.OS_KEY_RATIO_MAX:
                    candidats_os.append(r)
    
    touche_os = sorted(candidats_os, key=lambda r: r.centroid[1], reverse=True)[0] if candidats_os else None
    
    if config.DEBUG_MODE:
        if touche_os:
            print(f"   ✓ OS_KEY trouvé: {len(candidats_os)} candidats")
        else:
            print(f"   ✗ OS_KEY NON trouvé ({len(candidats_os)} candidats)")
    
    # 6. ENTER
    y_enter_target = cy_space - (h_ref * M["ENTER_Y_TARGET_HR"])
    candidats_enter = []
    
    for r in touches:
        cy, cx = r.centroid
        
        if abs(cy - y_enter_target) < (h_ref * M["ENTER_Y_TOLERANCE_HR"]):
            if cx > (cx_space + w_space * 0.3):
                candidats_enter.append(r)
    
    touche_enter = sorted(candidats_enter, key=lambda r: r.bbox[2] - r.bbox[0], reverse=True)[0] if candidats_enter else None
    
    if config.DEBUG_MODE:
        if touche_enter:
            print(f"   ✓ ENTER trouvé: {len(candidats_enter)} candidats")
        else:
            print(f"   ⚠️ ENTER NON trouvé ({len(candidats_enter)} candidats)")
    
    zones_trouvees = sum([
        spacebar is not None,
        shift_left is not None,
        top_left_key is not None,
        touche_os is not None
    ])

    if config.DEBUG_MODE:
        print(f"\n Bilan zoning: {zones_trouvees}/4 zones critiques trouvées")
        print(f" - SPACE: {'✓' if spacebar else '✗'}")
        print(f" - SHIFT: {'✓' if shift_left else '✗'}")
        print(f" - TL_LETTER: {'✓' if top_left_key else '✗'}")
        print(f" - OS_KEY: {'✓' if touche_os else '✗'}")
        print(f" - ENTER: {'✓' if touche_enter else '✗'}")

    if zones_trouvees == 2:
        if config.DEBUG_MODE:
            print(" Mode tolérant: élargissement des tolérances pour tenter de récupérer une zone manquante.")

        M_tol = dict(M)
        M_tol["SHIFT_Y_MIN_HR"] = max(0.3, M["SHIFT_Y_MIN_HR"] * 0.8)
        M_tol["SHIFT_Y_MAX_HR"] = M["SHIFT_Y_MAX_HR"] * 1.2
        M_tol["OS_DY_TOLERANCE"] = M["OS_DY_TOLERANCE"] * 1.3

        candidats_shift = []
        candidats_os = []

        for r in touches:
            cy, cx = r.centroid
            dy = cy - cy_space
            dx = cx - cx_space

            if -(h_ref * M_tol["SHIFT_Y_MAX_HR"]) < dy < -(h_ref * M_tol["SHIFT_Y_MIN_HR"]):
                if cx < (cx_space - w_space * 0.2):
                    h = r.bbox[2] - r.bbox[0]
                    w = r.bbox[3] - r.bbox[1]
                    ratio = w / h if h > 0 else 0
                    if config.SHIFT_RATIO_MIN < ratio < config.SHIFT_RATIO_MAX:
                        candidats_shift.append(r)

            if abs(dy) < (h_ref * M_tol["OS_DY_TOLERANCE"]):
                if -(w_space * 0.8) < dx < -(w_space * 0.05):
                    h = r.bbox[2] - r.bbox[0]
                    w = r.bbox[3] - r.bbox[1]
                    ratio = w / h if h > 0 else 0
                    if config.OS_KEY_RATIO_MIN < ratio < config.OS_KEY_RATIO_MAX:
                        candidats_os.append(r)

        if not shift_left and candidats_shift:
            shift_left = sorted(candidats_shift, key=lambda r: r.area, reverse=True)[0]
            if config.DEBUG_MODE:
                print(" Mode tolérant: SHIFT récupéré.")

        if not touche_os and candidats_os:
            touche_os = sorted(candidats_os, key=lambda r: r.centroid[1], reverse=True)[0]
            if config.DEBUG_MODE:
                print(" Mode tolérant: OS_KEY récupérée.")

        zones_trouvees = sum([
            spacebar is not None,
            shift_left is not None,
            top_left_key is not None,
            touche_os is not None
        ])
        if config.DEBUG_MODE:
            print(f" Après mode tolérant: {zones_trouvees}/4 zones critiques.")

    
    return {
        "SPACE": spacebar,
        "SHIFT": shift_left,
        "TL_LETTER": top_left_key,
        "OS_KEY": touche_os,
        "ENTER_KEY": touche_enter,
        "h_ref": h_ref
    }


def classifier_clavier(rois, img_gris, touches):
    """
    Classification avec validation multi-critères.
    """
    resultats = {"ISO_ANSI": "?", "MAC_WIN": "?", "LAYOUT": "?"}
    debug_info = {}
    
    # 1. ISO vs ANSI (Shift Gauche)
    if rois.get("SHIFT"):
        minr, minc, maxr, maxc = rois["SHIFT"].bbox
        h = maxr - minr
        w = maxc - minc
        ratio_l_h = w / h if h > 0 else 0
        debug_info["Shift_Ratio"] = ratio_l_h
        
        if ratio_l_h < config.THRESHOLD_SHIFT_RATIO_ISO:
            resultats["ISO_ANSI"] = "ISO (Europe)"
        else:
            resultats["ISO_ANSI"] = "ANSI (USA)"
    
    # 1b. Renforcement avec Enter
    if rois.get("ENTER_KEY"):
        r = rois["ENTER_KEY"]
        h = r.bbox[2] - r.bbox[0]
        w = r.bbox[3] - r.bbox[1]
        ratio_h_l = h / w if w > 0 else 0
        debug_info["Enter_Ratio_H_L"] = ratio_h_l
        
        if ratio_h_l < config.THRESHOLD_ENTER_RATIO_H_L_ANSI:
            resultats["ISO_ANSI"] = "ANSI (USA) [Conf. Enter]"
        elif ratio_h_l > config.THRESHOLD_ENTER_RATIO_H_L_ISO:
            resultats["ISO_ANSI"] = "ISO (Europe) [Conf. Enter]"
    
    # 2. Mac vs Windows (Euler)
    if rois.get("OS_KEY"):
        r = rois["OS_KEY"]
        minr, minc, maxr, maxc = r.bbox
        vignette = img_gris[minr:maxr, minc:maxc]
        
        try:
            thresh = filters.threshold_otsu(vignette)
            vignette_bin = vignette < thresh
            euler = measure.euler_number(vignette_bin, connectivity=2)
            debug_info["OS_Euler"] = euler
            
            if euler <= config.THRESHOLD_EULER_MAC:
                resultats["MAC_WIN"] = "Mac OS"
            elif euler >= config.THRESHOLD_EULER_WIN:
                resultats["MAC_WIN"] = "Windows/PC"
            else:
                resultats["MAC_WIN"] = "Incertain (prob. Mac)"
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"   Erreur calcul Euler: {e}")
            resultats["MAC_WIN"] = "Erreur détection"
    
    # 3. Classification Layout (OCR + Géométrie)
    if config.DEBUG_MODE:
        print("\n === Classification du Layout ===")

    layout_ocr, confiance_ocr, details_ocr = classifier_layout_ocr(rois, touches, img_gris, iso_ansi_info=resultats["ISO_ANSI"])

    if config.DEBUG_MODE:
        print(f" Résultat OCR: {layout_ocr} (confiance: {confiance_ocr:.0%})")

    layout_geo = None

    # Géométrie sur TL_LETTER comme avant
    if rois.get("TL_LETTER"):
        r = rois["TL_LETTER"]
        minr, _, maxr, _ = r.bbox
        h = maxr - minr
        cy_norm = (r.centroid[0] - minr) / h if h > 0 else 0
        extent = r.extent
        debug_info["TL_CenterY"] = cy_norm
        debug_info["TL_Extent"] = extent

        if config.DEBUG_MODE:
            print(f" TL_LETTER: CenterY={cy_norm:.3f} (seuil AZERTY: {config.THRESHOLD_TL_CENTER_Y_AZERTY})")
            print(f" TL_LETTER: Extent={extent:.3f} (seuil AZERTY: {config.THRESHOLD_TL_EXTENT_AZERTY})")

        if (cy_norm > config.THRESHOLD_TL_CENTER_Y_AZERTY and
                extent < config.THRESHOLD_TL_EXTENT_AZERTY):
            layout_geo = "AZERTY"
        else:
            layout_geo = "QWERTY/QWERTZ"

    # Fusion OCR + Géométrie
    if layout_ocr and confiance_ocr >= 0.35:
        base_layout = details_ocr.get('base_layout', layout_ocr.split('_')[0])
        region = details_ocr.get('region', '?')
        
        resultats["LAYOUT"] = f"{base_layout}_{region} [OCR:{confiance_ocr:.0%}]"
        
        if 'sequences' in details_ocr:
            debug_info["Layout_Seq_Row1"] = ''.join(details_ocr['sequences'].get('row1', []))
            debug_info["Layout_Seq_Row2"] = ''.join(details_ocr['sequences'].get('row2', []))
        
        debug_info["Layout_Confiance"] = confiance_ocr
        debug_info["Layout_Method"] = "OCR"
    else:
        # Fallback géométrique pur
        if config.DEBUG_MODE:
            print(" OCR insuffisant, fallback géométrique...")
        if layout_geo:
            if layout_geo == "AZERTY":
                resultats["LAYOUT"] = "AZERTY [Geo]"
                if config.DEBUG_MODE:
                    print(" → Verdict géométrique: AZERTY")
            elif layout_geo == "QWERTY/QWERTZ":
                resultats["LAYOUT"] = "QWERTY/QWERTZ [Geo]"
                if config.DEBUG_MODE:
                    print(" → Verdict géométrique: QWERTY/QWERTZ")
        
        debug_info["Layout_Method"] = "Geometric"
        debug_info["Layout_Confiance"] = 0.5

    if layout_ocr:
        debug_info["Layout_OCR_Tentative"] = f"{layout_ocr}({confiance_ocr:.0%})"

    return resultats, debug_info
