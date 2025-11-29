import numpy as np
from PIL import Image
from skimage import filters, exposure, transform, util
from skimage.morphology import disk, binary_opening, binary_closing

def charger_image_safe(path):
    """Charge l'image avec PIL pour éviter les bugs Numpy/Skimage récents."""
    pil_img = Image.open(path).convert('RGB')
    return np.array(pil_img)

def pretraiter_image(img_path_or_array):
    # 1. Chargement
    if isinstance(img_path_or_array, str):
        img = charger_image_safe(img_path_or_array)
    else:
        img = img_path_or_array
    
    # 2. Conversion Gris
    gris = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    gris = gris.astype(np.uint8)
    
    # 3. AJOUT LÉGER: Débruitage doux (optionnel, peut être désactivé)
    # Décommenter si tu veux tester:
    # gris = filters.gaussian(gris, sigma=0.5, preserve_range=True).astype(np.uint8)
    
    # 4. Contraste (TON CODE ORIGINAL)
    p2, p98 = np.percentile(gris, (2, 98))
    gris_ocr = exposure.rescale_intensity(gris, in_range=(p2, p98))
    
    # 5. Binarisation (TON CODE ORIGINAL - Otsu global)
    try:
        seuil = filters.threshold_otsu(gris)
        binaire = gris > seuil
    except:
        binaire = gris > 127
    
    # 6. AUTO-CORRECTION INTELLIGENTE DU MASQUE (TON CODE ORIGINAL)
    if np.sum(binaire) > (binaire.size * 0.5):
        binaire = np.invert(binaire)
    
    # 7. Nettoyage (TON CODE ORIGINAL)
    binaire = binary_opening(binaire, footprint=disk(2))
    binaire = binary_closing(binaire, footprint=disk(3))
    
    return binaire, gris_ocr

def pretraiter_vignette_ocr(vignette_roi, inversion=False):
    if vignette_roi.size == 0: return vignette_roi
    
    img = vignette_roi.copy()
    
    if inversion:
        img = util.invert(img)
    
    # Upscale
    img = transform.rescale(img, scale=3, order=1, preserve_range=True, anti_aliasing=False)
    img = img.astype(np.uint8)
    
    # Padding
    h, w = img.shape
    pad = 20
    canvas = np.ones((h + 2*pad, w + 2*pad), dtype=np.uint8) * 255
    
    # Inversion locale pour Tesseract (toujours Noir sur Blanc)
    if np.median(img) < 127:
        img = util.invert(img)
    
    canvas[pad:pad+h, pad:pad+w] = img
    
    return canvas
