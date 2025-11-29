"""
Prétraitement avancé avec détection automatique des touches
"""
import cv2
import numpy as np


def detect_key_regions(image, debug=False):
    """
    Détecte automatiquement les régions de touches du clavier
    
    Args:
        image: Image du clavier
        debug: Si True, retourne aussi les images de debug
        
    Returns:
        Liste de rectangles (x, y, w, h) des touches détectées
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Détection de contours
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilatation pour fermer les contours
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Trouver les contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours pour garder seulement ceux qui ressemblent à des touches
    height, width = image.shape[:2]
    min_key_area = (width * height) * 0.0005  # Au moins 0.05% de l'image
    max_key_area = (width * height) * 0.05    # Au max 5% de l'image
    
    key_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_key_area < area < max_key_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Vérifier le ratio (touches carrées/rectangulaires)
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.5:
                key_regions.append((x, y, w, h))
    
    # Trier par position (haut à bas, gauche à droite)
    key_regions.sort(key=lambda r: (r[1], r[0]))
    
    return key_regions


def extract_top_row_keys(image, num_keys=6):
    """
    Extrait les N premières touches de la rangée supérieure
    
    Args:
        image: Image du clavier
        num_keys: Nombre de touches à extraire
        
    Returns:
        Liste d'images des touches extraites
    """
    key_regions = detect_key_regions(image)
    
    if not key_regions:
        return []
    
    # Trouver la première rangée (touches avec Y similaire)
    if len(key_regions) == 0:
        return []
    
    # Grouper les touches par rangée (tolerance de ±20% de hauteur)
    first_key_y = key_regions[0][1]
    first_row_keys = []
    
    for x, y, w, h in key_regions:
        # Si la touche est à peu près à la même hauteur que la première
        if abs(y - first_key_y) < h * 0.5:
            first_row_keys.append((x, y, w, h))
    
    # Trier les touches de gauche à droite
    first_row_keys.sort(key=lambda r: r[0])
    
    # Prendre les N premières
    top_keys = first_row_keys[:num_keys]
    
    # Extraire les images des touches
    key_images = []
    for x, y, w, h in top_keys:
        key_img = image[y:y+h, x:x+w]
        key_images.append(key_img)
    
    return key_images


def create_combined_key_strip(key_images):
    """
    Combine les images des touches en une seule bande horizontale
    
    Args:
        key_images: Liste d'images de touches
        
    Returns:
        Image combinée
    """
    if not key_images:
        return None
    
    # Normaliser la hauteur de toutes les touches
    max_height = max(img.shape[0] for img in key_images)
    
    resized_keys = []
    for img in key_images:
        h, w = img.shape[:2]
        if h < max_height:
            # Redimensionner pour avoir la même hauteur
            ratio = max_height / h
            new_w = int(w * ratio)
            resized = cv2.resize(img, (new_w, max_height))
            resized_keys.append(resized)
        else:
            resized_keys.append(img)
    
    # Concaténer horizontalement
    combined = np.hstack(resized_keys)
    
    return combined


def extract_smart_roi(image):
    """
    Extraction intelligente de la ROI en détectant les touches
    
    Args:
        image: Image du clavier complet
        
    Returns:
        ROI optimisée contenant les premières touches
    """
    # Tenter la détection automatique
    key_images = extract_top_row_keys(image, num_keys=6)
    
    if key_images and len(key_images) >= 4:
        # Succès ! Combiner les touches
        combined = create_combined_key_strip(key_images)
        if combined is not None:
            return combined
    
    # Fallback sur la méthode classique
    height, width = image.shape[:2]
    y_start = int(height * 0.20)
    y_end = int(height * 0.35)
    x_end = int(width * 0.70)
    
    roi = image[y_start:y_end, 0:x_end]
    return roi


def preprocess_for_text_ocr(image):
    """
    Prétraitement spécialisé pour la reconnaissance de texte sur touches de clavier
    
    Args:
        image: Image de la ROI
        
    Returns:
        Image prétraitée optimale pour OCR
    """
    # Conversion en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Égalisation d'histogramme adaptative TRÈS agressive
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Augmenter encore le contraste
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
    
    # Débruitage
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Binarisation adaptative avec fenêtre plus large
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 10
    )
    
    # Closing pour connecter les parties de lettres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Inversion si nécessaire (texte noir sur fond blanc)
    if np.mean(closed) < 127:
        closed = cv2.bitwise_not(closed)
    
    return closed