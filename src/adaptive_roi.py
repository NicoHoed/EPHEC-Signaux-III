"""
Extraction adaptative de ROI pour différents types de claviers
"""
import cv2
import numpy as np


def detect_keyboard_type(image):
    """
    Détecte le type de clavier basé sur les dimensions
    
    Args:
        image: Image du clavier
        
    Returns:
        Type de clavier détecté
    """
    height, width = image.shape[:2]
    ratio = width / height
    
    # Full keyboard (avec pavé numérique): ratio > 3.5
    if ratio > 3.5:
        return 'full'
    # TKL (Ten Key Less - sans pavé numérique): 2.8 < ratio < 3.5
    elif 2.8 < ratio <= 3.5:
        return 'tkl'
    # Compact (60-75%): 2.0 < ratio < 2.8
    elif 2.0 < ratio <= 2.8:
        return 'compact'
    # Autres layouts
    else:
        return 'unknown'


def extract_adaptive_roi(image, verbose=False):
    """
    Extrait la ROI de manière adaptative selon le type de clavier
    
    Args:
        image: Image du clavier normalisé
        verbose: Afficher les infos de détection
        
    Returns:
        ROI extraite
    """
    height, width = image.shape[:2]
    keyboard_type = detect_keyboard_type(image)
    
    if verbose:
        print(f"   Type détecté: {keyboard_type} (ratio: {width/height:.2f})")
    
    # Paramètres selon le type
    if keyboard_type == 'full':
        # Clavier complet avec pavé numérique
        y_start_pct = 0.30
        y_end_pct = 0.45
        x_end_pct = 0.55  # Moins large car pavé numérique à droite
    elif keyboard_type == 'tkl':
        # TKL - Ten Key Less
        y_start_pct = 0.30
        y_end_pct = 0.45
        x_end_pct = 0.65
    elif keyboard_type == 'compact':
        # Clavier compact (60-75%)
        y_start_pct = 0.28
        y_end_pct = 0.47
        x_end_pct = 0.75
    else:
        # Fallback - paramètres génériques
        y_start_pct = 0.30
        y_end_pct = 0.45
        x_end_pct = 0.70
    
    y_start = int(height * y_start_pct)
    y_end = int(height * y_end_pct)
    x_end = int(width * x_end_pct)
    
    if verbose:
        print(f"   ROI: y[{y_start_pct*100:.0f}%-{y_end_pct*100:.0f}%], x[0-{x_end_pct*100:.0f}%]")
    
    roi = image[y_start:y_end, 0:x_end]
    return roi


def extract_multiple_roi_candidates(image):
    """
    Extrait plusieurs ROI candidates (si la première échoue)
    
    Args:
        image: Image du clavier normalisé
        
    Returns:
        Liste de ROI candidates
    """
    height, width = image.shape[:2]
    
    candidates = []
    
    # Candidate 1: Rangée standard (lettres)
    y1_start = int(height * 0.30)
    y1_end = int(height * 0.45)
    roi1 = image[y1_start:y1_end, 0:int(width * 0.70)]
    candidates.append(('standard', roi1))
    
    # Candidate 2: Un peu plus haut (au cas où)
    y2_start = int(height * 0.25)
    y2_end = int(height * 0.40)
    roi2 = image[y2_start:y2_end, 0:int(width * 0.70)]
    candidates.append(('higher', roi2))
    
    # Candidate 3: Un peu plus bas
    y3_start = int(height * 0.35)
    y3_end = int(height * 0.50)
    roi3 = image[y3_start:y3_end, 0:int(width * 0.70)]
    candidates.append(('lower', roi3))
    
    return candidates


def find_best_roi_with_text(image, preprocessor_func, ocr_func):
    """
    Trouve la meilleure ROI en testant plusieurs candidates
    
    Args:
        image: Image du clavier normalisé
        preprocessor_func: Fonction de prétraitement
        ocr_func: Fonction OCR
        
    Returns:
        Meilleure ROI et son score OCR
    """
    candidates = extract_multiple_roi_candidates(image)
    
    best_roi = None
    best_score = 0
    best_text = ""
    
    for name, roi in candidates:
        # Prétraiter
        preprocessed = preprocessor_func(roi)
        
        # OCR rapide
        text, confidence = ocr_func(preprocessed)
        
        # Calculer un score (longueur + confiance)
        score = len(text) * (confidence / 100)
        
        if score > best_score:
            best_score = score
            best_roi = roi
            best_text = text
    
    return best_roi, best_text, best_score