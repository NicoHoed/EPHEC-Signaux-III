"""
Module de prétraitement simplifié et agressif
"""
import cv2
import numpy as np

def preprocess_for_text_ocr(image):
    """
    Prétraitement universel pour clavier :
    1. Grayscale
    2. Threshold agressif (Noir et Blanc pur)
    3. Inversion automatique (pour avoir toujours Texte Noir sur Fond Blanc)
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 2. Upscaling (Agrandir l'image aide l'OCR)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Augmentation locale du contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. Thresholding (Binarisation)
    # Otsu trouve le seuil optimal automatiquement
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Gestion Fond Noir vs Fond Blanc
    # On compte les pixels blancs
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    
    # Si plus de 50% de l'image est noire, c'est probablement un clavier noir avec texte blanc
    # Tesseract veut du texte NOIR sur fond BLANC. Donc on inverse.
    if white_pixels < (total_pixels * 0.5):
        binary = cv2.bitwise_not(binary)
        
    # 6. Dilatation légère pour épaissir les caractères fins
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1) # Erode sur fond blanc = épaissir le noir

    return binary