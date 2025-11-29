"""
Module de prétraitement d'images pour la reconnaissance de layout clavier
"""
import cv2
import numpy as np
from skimage import exposure


def preprocess_version_a(image):
    """
    Prétraitement Version A - Éclairage Normal (AMÉLIORÉ)
    Pipeline: Grayscale → CLAHE → Gaussian Blur → Otsu Threshold
    
    Args:
        image: Image numpy array (BGR)
        
    Returns:
        Image prétraitée (binaire)
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CLAHE plus agressif pour augmenter le contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Débruitage avec filtre gaussien
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Binarisation avec Otsu
    _, binary = cv2.threshold(blurred, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def preprocess_version_b(image):
    """
    Prétraitement Version B - Éclairage Sombre
    Pipeline: Grayscale → Gamma Correction → CLAHE → Bilateral Filter → Adaptive Threshold
    
    Args:
        image: Image numpy array (BGR)
        
    Returns:
        Image prétraitée (binaire)
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Correction gamma pour éclaircir
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)
    
    # CLAHE plus agressif
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gamma_corrected)
    
    # Filtre bilatéral (préserve les bords)
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(bilateral, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    return binary


def preprocess_version_c(image):
    """
    Prétraitement Version C - Éclairage Clair/Reflets (OPTIMISÉ)
    Pipeline: Grayscale → Normalization → Adaptive Threshold DOUX
    Cette version fonctionne le mieux selon les tests !
    
    Args:
        image: Image numpy array (BGR)
        
    Returns:
        Image prétraitée (binaire)
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalisation de l'intensité
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # CLAHE modéré pour améliorer le contraste sans trop nettoyer
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    
    # Léger débruitage qui préserve les détails
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Binarisation adaptative DOUCE (fenêtre large, seuil bas)
    binary = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 8)
    
    return binary


def preprocess_multipass(image, save_debug=False, debug_path=None, filename=None):
    """
    Applique les 3 versions de prétraitement
    PRIORITÉ à la version C (bright) qui fonctionne le mieux
    
    Args:
        image: Image numpy array (BGR)
        save_debug: Si True, sauvegarde les versions intermédiaires
        debug_path: Chemin pour sauvegarder les images debug
        filename: Nom de base du fichier
        
    Returns:
        Liste des 3 versions prétraitées (avec version C en priorité)
    """
    versions = []
    
    # Version C (BRIGHT) - LA MEILLEURE - on la met 3 fois pour qu'elle gagne le vote !
    version_c = preprocess_version_c(image)
    versions.append(('C_bright_1', version_c))
    versions.append(('C_bright_2', version_c))
    versions.append(('C_bright_3', version_c))
    
    # Version B (plus douce)
    version_b = preprocess_version_b(image)
    versions.append(('B_dark', version_b))
    
    # Version A
    version_a = preprocess_version_a(image)
    versions.append(('A_normal', version_a))
    
    # Sauvegarde pour debug si demandé
    if save_debug and debug_path and filename:
        from pathlib import Path
        base_name = Path(filename).stem
        
        # Sauvegarder seulement les versions uniques
        debug_filename_c = f"{base_name}_C_bright.png"
        debug_full_path_c = Path(debug_path) / debug_filename_c
        cv2.imwrite(str(debug_full_path_c), version_c)
        
        debug_filename_b = f"{base_name}_B_dark.png"
        debug_full_path_b = Path(debug_path) / debug_filename_b
        cv2.imwrite(str(debug_full_path_b), version_b)
        
        debug_filename_a = f"{base_name}_A_normal.png"
        debug_full_path_a = Path(debug_path) / debug_filename_a
        cv2.imwrite(str(debug_full_path_a), version_a)
    
    return versions


def enhance_for_ocr(binary_image):
    """
    Post-traitement DOUX pour améliorer l'OCR sans sur-nettoyer
    
    Args:
        binary_image: Image binaire
        
    Returns:
        Image légèrement améliorée (pas trop nettoyée)
    """
    # Pas de dilatation/érosion agressive
    # Juste un léger nettoyage du bruit avec opening
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned


def invert_if_needed(image):
    """
    Inverse l'image si le texte est blanc sur fond noir
    (Tesseract préfère le texte noir sur fond blanc)
    
    Args:
        image: Image binaire
        
    Returns:
        Image éventuellement inversée
    """
    # Calcule la moyenne des pixels
    mean_value = np.mean(image)
    
    # Si la moyenne > 127, plus de pixels blancs que noirs
    # Cela signifie probablement fond blanc, texte noir (OK)
    # Si moyenne < 127, probablement fond noir, texte blanc (à inverser)
    if mean_value < 127:
        return cv2.bitwise_not(image)
    
    return image