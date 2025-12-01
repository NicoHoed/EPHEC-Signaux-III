"""
Fonctions utilitaires pour le détecteur de layout clavier
"""
import os
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

def create_output_dirs(base_output_path):
    output_path = Path(base_output_path)
    processed_path = output_path / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    return output_path, processed_path

def load_image(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None: return None
        return img
    except Exception:
        return None

def save_image(image, save_path, filename):
    try:
        full_path = Path(save_path) / filename
        cv2.imwrite(str(full_path), image)
    except Exception:
        pass

def normalize_resolution(image, target_width=1800):
    """
    Augmente la résolution pour que l'OCR voie mieux les petits détails.
    Passage de 1200 à 1800px.
    """
    height, width = image.shape[:2]
    ratio = target_width / width
    new_height = int(height * ratio)
    return cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LANCZOS4)

def extract_roi(image, roi_type="top_row", adaptive=False, verbose=False):
    """
    Extrait une zone large et sûre (Safe Zone) plutôt qu'une zone trop précise.
    On vise le quart haut-gauche où se trouvent les touches critiques (A, Q, Z, W).
    """
    height, width = image.shape[:2]
    
    # Stratégie "SHOTGUN": On prend une zone large en haut à gauche
    # La ligne QWERTY se situe généralement entre 20% et 45% de la hauteur
    # Les touches Q/A W/Z se trouvent dans les premiers 30% de la largeur
    
    y_start = int(height * 0.20)  # On commence assez haut (sous les touches F1-F12)
    y_end = int(height * 0.50)    # On descend jusqu'au milieu du clavier (ligne ASDF)
    x_end = int(width * 0.60)     # On prend la moitié gauche (suffisant pour QWERT/AZERT)
    
    roi = image[y_start:y_end, 0:x_end]
    
    return roi

def generate_report(results, output_path):
    # (Code inchangé pour le rapport, simplifié pour l'exemple)
    pass
    
def print_summary(report):
    # (Code inchangé)
    pass

def get_image_files(input_path):
    input_path = Path(input_path)
    image_files = list(input_path.glob("*.png"))
    image_files.extend(input_path.glob("*.PNG"))
    return sorted(image_files)