import cv2
import numpy as np
from skimage import color

def pretraiter_image(img, debug=False):
    """
    Prétraitement avec OpenCV :
    - Conversion gris
    - Filtre bilateral (anti-bruit)
    - Equalisation CLAHE
    - Seuillage adaptatif
    - Nettoyage morphologique
    Retourne:
        - masque binaire nettoyé
        - image grayscale (pour analyse fine)
    """

    # ----------------------------
    # 1. Conversion en niveaux de gris
    # ----------------------------
    if len(img.shape) == 3:
        img_rgb = img[..., :3]  # retirer canal alpha si besoin
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ----------------------------
    # 2. Denoising (bilateral filter)
    # ----------------------------
    gray_filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # ----------------------------
    # 3. Amélioration contraste CLAHE
    # ----------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_filtered)

    # ----------------------------
    # 4. Seuillage adaptatif → très robuste
    # ----------------------------
    bin_img = cv2.adaptiveThreshold(
        gray_clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=2
    )

    # ----------------------------
    # 5. Morphologie (open/close)
    # ----------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_clean = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Conversion → bool (compatible avec ton detecter_touches)
    bin_clean_bool = bin_clean.astype(bool)

    # ----------------------------
    # 6. Debug (affichage)
    # ----------------------------
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Gray"); ax[0].axis("off")
        ax[1].imshow(gray_clahe, cmap='gray'); ax[1].set_title("CLAHE"); ax[1].axis("off")
        ax[2].imshow(bin_img, cmap='gray'); ax[2].set_title("Binaire brut"); ax[2].axis("off")
        ax[3].imshow(bin_clean, cmap='gray'); ax[3].set_title("Nettoyée"); ax[3].axis("off")
        plt.show()

    return bin_clean_bool, gray
