import numpy as np
import cv2
import config
from skimage import filters, measure

# -----------------------------------------------------
#  FakeRegion : compatible avec regionprops
# -----------------------------------------------------

class FakeRegion:
    def __init__(self, contour, x, y, w, h, area):
        self.contour = contour
        self.bbox = (y, x, y + h, x + w)
        self.area = area

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + w/2, y + h/2

        self.centroid = (cy, cx)
        self.extent = area / (w*h)
        self.solidity = 1.0
        self.euler_number = 0


# -----------------------------------------------------
#  DÉTECTION DES TOUCHES – VERSION OPENCV
# -----------------------------------------------------

def detecter_touches(img_binaire,
                     aire_min=config.AIRE_MIN,
                     aire_max=config.AIRE_MAX,
                     ratio_max=config.RATIO_MAX,
                     seuil_y=config.SEUIL_Y_PROXIMITE):
    
    binary_uint8 = (img_binaire * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    touches = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (aire_min <= area <= aire_max):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        ratio = w / h
        if ratio > ratio_max:
            continue

        touches.append(FakeRegion(cnt, x, y, w, h, area))

    if not touches:
        return [], 0, 0, 0

    centres_y = [r.centroid[0] for r in touches]
    moyenne_y = np.mean(centres_y)

    y_min = moyenne_y - seuil_y
    y_max = moyenne_y + seuil_y

    touches_filtrees = [r for r in touches if y_min < r.centroid[0] < y_max]

    return touches_filtrees, moyenne_y, y_min, y_max


# -----------------------------------------------------
#  IDENTIFICATION DES ZONES
# -----------------------------------------------------

def identifier_zones_cles(touches):
    if not touches:
        return None

    M = config.ZONING_HR_MULTIPLIERS

    # 1. Barre espace
    spacebar = sorted(touches, key=lambda r: r.area, reverse=True)[0]
    cy_space, cx_space = spacebar.centroid
    w_space = spacebar.bbox[3] - spacebar.bbox[1]

    # 2. Hauteur de référence
    hauteurs = [(r.bbox[2] - r.bbox[0]) for r in touches]
    h_ref = np.median(hauteurs)

    # 3. OS KEY
    candidats_os = []
    for r in touches:
        if r == spacebar:
            continue
        cy, cx = r.centroid
        dy = cy - cy_space
        dx = cx - cx_space
        if abs(dy) < (h_ref * M["OS_DY_TOLERANCE"]) and -(w_space/2 + 250) < dx < -(w_space/2 * 0.1):
            candidats_os.append(r)
    os_key = sorted(candidats_os, key=lambda r: r.centroid[1])[-2] if candidats_os else None

    # 4. SHIFT
    candidats_shift = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space
        if -(h_ref * M["SHIFT_Y_MAX_HR"]) < dy < -(h_ref * M["SHIFT_Y_MIN_HR"]) and cx < cx_space:
            candidats_shift.append(r)
    shift_left = sorted(candidats_shift, key=lambda r: r.area, reverse=True)[0] if candidats_shift else None

    # 5. TL LETTER
    candidats_top = []
    for r in touches:
        cy, cx = r.centroid
        dy = cy - cy_space
        if -(h_ref * M["TL_LETTER_Y_MAX_HR"]) < dy < -(h_ref * M["TL_LETTER_Y_MIN_HR"]):
            candidats_top.append(r)

    top_left_key = None
    if candidats_top:
        ligne = sorted(candidats_top, key=lambda r: r.centroid[1])
        premier = ligne[0]
        ratio_premier = (premier.bbox[3] - premier.bbox[1]) / (premier.bbox[2] - premier.bbox[0])
        if len(ligne) > 1 and ratio_premier > config.THRESHOLD_TAB_RATIO:
            top_left_key = ligne[1]
        else:
            top_left_key = premier

    # 6. ENTER
    y_enter_target = cy_space - (h_ref * M["ENTER_Y_TARGET_HR"])
    candidats_enter = []
    for r in touches:
        cy, cx = r.centroid
        if abs(cy - y_enter_target) < (h_ref * M["ENTER_Y_TOLERANCE_HR"]) and cx > (cx_space + 200):
            candidats_enter.append(r)
    enter_key = sorted(candidats_enter, key=lambda r: r.bbox[2] - r.bbox[0], reverse=True)[0] if candidats_enter else None

    return {
        "SPACE": spacebar,
        "SHIFT": shift_left,
        "TL_LETTER": top_left_key,
        "OS_KEY": os_key,
        "ENTER_KEY": enter_key
    }


# -----------------------------------------------------
#  CLASSIFICATION
# -----------------------------------------------------

def classifier_clavier(rois, img_gris):
    resultats = {"ISO_ANSI": "?", "MAC_WIN": "?", "LAYOUT": "?"}
    debug = {}

    # 1. ISO vs ANSI via Shift
    if rois["SHIFT"]:
        minr, minc, maxr, maxc = rois["SHIFT"].bbox
        ratio = (maxc - minc) / (maxr - minr)
        debug["Shift_Ratio"] = ratio
        if ratio < config.THRESHOLD_SHIFT_RATIO_ISO:
            resultats["ISO_ANSI"] = "ISO (Europe)"
        else:
            resultats["ISO_ANSI"] = "ANSI (USA)"

    # 2. ENTER reinforce
    if rois["ENTER_KEY"]:
        r = rois["ENTER_KEY"]
        h = r.bbox[2] - r.bbox[0]
        w = r.bbox[3] - r.bbox[1]
        ratio = h / w
        debug["Enter_Ratio_H_L"] = ratio
        if ratio < config.THRESHOLD_ENTER_RATIO_H_L_ANSI:
            resultats["ISO_ANSI"] = "ANSI (USA) [Enter]"
        elif ratio > config.THRESHOLD_ENTER_RATIO_H_L_ISO:
            resultats["ISO_ANSI"] = "ISO (Europe) [Enter]"

    # 3. OS Key classification
    if rois["OS_KEY"]:
        r = rois["OS_KEY"]
        minr, minc, maxr, maxc = r.bbox
        vignette = img_gris[minr:maxr, minc:maxc]
        thresh = filters.threshold_otsu(vignette)
        bin_os = vignette < thresh
        euler = measure.euler_number(bin_os, connectivity=2)
        debug["OS_Euler"] = euler
        if euler <= config.THRESHOLD_EULER_MAC:
            resultats["MAC_WIN"] = "Mac OS"
        elif euler >= config.THRESHOLD_EULER_WIN:
            resultats["MAC_WIN"] = "Windows/PC"
        else:
            resultats["MAC_WIN"] = "Incertain"

    # 4. AZERTY / QWERTY via TL Letter
    if rois["TL_LETTER"]:
        r = rois["TL_LETTER"]
        minr, _, maxr, _ = r.bbox
        cy_norm = (r.centroid[0] - minr) / (maxr - minr)
        extent = r.extent
        debug["TL_CenterY"] = cy_norm
        debug["TL_Extent"] = extent
        if cy_norm > config.THRESHOLD_TL_CENTER_Y_AZERTY and extent < config.THRESHOLD_TL_EXTENT_AZERTY:
            resultats["LAYOUT"] = "AZERTY"
        else:
            resultats["LAYOUT"] = "QWERTY"

    return resultats, debug
