import cv2
import numpy as np
from skimage import color

def pretraiter_image(img, debug=False):
    # 1. Gris
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Flou pour lisser les lettres
    gray_blur = cv2.GaussianBlur(gray, (7,7), 0)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_blur)

    # 4. Adaptative threshold
    bin_adapt = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,  # fenêtre grande = meilleure séparation
        5
    )

    # 5. Morph CLOSE fort pour remplir les touches
    ker5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(bin_adapt, cv2.MORPH_CLOSE, ker5, iterations=3)

    # 6. Morph OPEN pour retirer les lettres internes
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, ker5, iterations=2)

    # 7. Nouvelle fermeture pour lisser les rectangles
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, ker5, iterations=3)

    # 8. En booléen
    bin_bool = clean.astype(bool)

    # DEBUG
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(gray, cmap='gray'); ax[0].set_title("Gray"); ax[0].axis("off")
        ax[1].imshow(gray_clahe, cmap='gray'); ax[1].set_title("CLAHE"); ax[1].axis("off")
        ax[2].imshow(bin_adapt, cmap='gray'); ax[2].set_title("Adaptive"); ax[2].axis("off")
        ax[3].imshow(clean, cmap='gray'); ax[3].set_title("Final Clean"); ax[3].axis("off")
        plt.show()

    return bin_bool, gray
