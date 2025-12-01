"""
Prétraitement Brute Force : Seuillage Adaptatif
"""
import cv2
import numpy as np

def extract_smart_roi(image, debug=False):
    """
    Zone fixe large. On ne prend pas de risques.
    """
    h, w = image.shape[:2]
    # Zone 15% à 60% (Cœur du clavier)
    roi = image[int(h*0.15):int(h*0.60), :]
    
    # Normalisation Hauteur 600px
    target_height = 600
    ratio = target_height / roi.shape[0]
    target_width = int(roi.shape[1] * ratio)
    
    return cv2.resize(roi, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def preprocess_for_text_ocr(image):
    """
    Retourne l'image binaire brute.
    On ne filtre plus les contours (trop risqué pour les textes fins).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Augmentation du contraste locale (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # SEUILLAGE ADAPTATIF (Le plus robuste)
    # Il calcule le seuil pour chaque petite zone de l'image.
    # Block Size 21, C=10 (Ajusté pour réduire le bruit)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        21, 10
    )
    
    # Nettoyage léger (bruit de sel/poivre) mais sans toucher aux lettres
    # Denoising
    final = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return final