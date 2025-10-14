import numpy as np
from skimage import measure


def detect_touch_rectangles(binary_img, min_area=500, max_area=100000):
    """
    Détecte les rectangles noirs correspondant aux touches sur une image binaire.
    """
    # Inverser l'image pour que les touches noires deviennent des objets
    inverted = np.invert(binary_img.astype(bool))

    # Labeliser les composants connectés
    labels = measure.label(inverted)
    regions = measure.regionprops(labels)

    touch_boxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        area = region.area
        if min_area <= area <= max_area:
            touch_boxes.append((minr, minc, maxr, maxc))

    return touch_boxes
