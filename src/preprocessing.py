import numpy as np
from skimage import color, filters, exposure, morphology
from scipy.ndimage import median_filter
import config

def pretraiter_image(img):
    """
    Chaîne de traitement adaptative améliorée.
    Retourne:
    - nettoyee: image binaire nettoyée (True = Touche, False = Fond)
    - gris: image en niveaux de gris (pour analyse fine ultérieure)
    """
    # 1. Conversion en niveaux de gris
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        gris = color.rgb2gray(img)
    else:
        gris = img.astype(float) / 255.0 if img.max() > 1 else img

    if config.DEBUG_MODE:
        print(f" Dimensions image: {gris.shape}")
        print(f" Luminosité moyenne: {np.mean(gris):.2f}")

    # 2. Détection de la luminosité globale et ajustement adaptatif
    luminosite_moyenne = np.mean(gris)
    if config.PREPROCESS_MODE == "adaptive":
        if luminosite_moyenne < config.LUMINOSITY_THRESHOLD_DARK:
            # Image sombre: éclaircir
            gris = exposure.adjust_gamma(gris, gamma=config.GAMMA_DARK)
            if config.DEBUG_MODE:
                print(f" Correction gamma (sombre): {config.GAMMA_DARK}")
        elif luminosite_moyenne > config.LUMINOSITY_THRESHOLD_BRIGHT:
            # Image claire: assombrir légèrement
            gris = exposure.adjust_gamma(gris, gamma=config.GAMMA_BRIGHT)
            if config.DEBUG_MODE:
                print(f" Correction gamma (claire): {config.GAMMA_BRIGHT}")

    # 2b. LÉGER SHARPEN pour aider les bords (utile sur images un peu floues)
    # On applique un filtre Laplacien très léger
    laplace = filters.laplace(gris)
    gris_sharp = np.clip(gris + 0.3 * laplace, 0, 1)

    # 3. Réduction du bruit adaptative
    filtree = filters.gaussian(gris_sharp, sigma=1.5)

    # 4. Égalisation adaptative par zones (CLAHE)
    filtree = exposure.equalize_adapthist(filtree, clip_limit=0.03)

    # 5. Binarisation multi-seuils (combinaison Otsu + Local)
    seuil_otsu = filters.threshold_otsu(filtree)

    # Seuil local avec taille de bloc adaptative
    block_size = max(51, int(min(filtree.shape) / 20))
    if block_size % 2 == 0:  # Doit être impair
        block_size += 1

    try:
        seuil_local = filters.threshold_local(filtree, block_size=block_size)
        # Combinaison des deux approches (ET logique)
        binaire_otsu = filtree > seuil_otsu
        binaire_local = filtree > seuil_local
        binaire = np.logical_and(binaire_otsu, binaire_local)
    except Exception:
        # Fallback sur Otsu seul si le seuil local échoue
        if config.DEBUG_MODE:
            print(" Avertissement: seuil local échoué, utilisation d'Otsu seul")
        binaire = filtree > seuil_otsu

    # 6. Nettoyage morphologique adaptatif
    kernel_size = max(2, int(min(gris.shape) / 500))
    kernel = morphology.disk(kernel_size)
    if config.DEBUG_MODE:
        print(f" Taille kernel morphologique: {kernel_size}")

    nettoyee = morphology.binary_opening(binaire, kernel)
    nettoyee = morphology.binary_closing(nettoyee, kernel)

    # Suppression des petits objets et petits trous
    min_size = int(gris.shape[0] * gris.shape[1] * 0.0001)  # 0.01% de l'image
    nettoyee = morphology.remove_small_objects(nettoyee, min_size=min_size)
    nettoyee = morphology.remove_small_holes(nettoyee, area_threshold=200)

    # 7. Inversion finale
    nettoyee = np.invert(nettoyee)

    return nettoyee, gris_sharp

def pretraiter_image_basic(img):
    """
    Version basique du prétraitement (pour compatibilité).
    """
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        gris = color.rgb2gray(img)
    else:
        gris = img
    
    filtree = median_filter(gris, size=3)
    filtree = exposure.equalize_hist(filtree)
    seuil = filters.threshold_otsu(filtree)
    binaire = filtree > seuil
    
    from scipy.ndimage import binary_opening, binary_closing
    nettoyee = binary_opening(binaire, iterations=2)
    nettoyee = binary_closing(nettoyee, iterations=3)
    nettoyee = np.invert(nettoyee)
    
    return nettoyee, gris
