"""
--------------------------------------------------------------------------------
File: src/preprocessing.py
Author:
    Nicolas HOEDENAEKEN
    Théo MERTENS
    Baris OZCELIK
    Khassan AKTAMIROV
    
Description: 
    Module de traitement d'image utilisant OpenCV.
    Fournit plusieurs méthodes (Adaptive Threshold, LAB Channel, Inversion)
    pour préparer l'image avant l'OCR, afin de gérer les reflets, 
    les ombres et les claviers à fort contraste (touches noires/lettres blanches).
--------------------------------------------------------------------------------
"""

import cv2
import numpy as np

def method_adaptive_threshold(img):
    """Méthode 1: Contraste local agressif (bon pour les reflets)"""
    # Upscale pour aider l'OCR
    img_upscaled = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY)
    
    # Denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Inversion (Texte blanc sur noir devient Noir sur Blanc)
    gray = cv2.bitwise_not(gray)
    
    # Seuil adaptatif
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, # Fenêtre large
        C=10
    )
    return binary

def method_lab_channel(img):
    """Méthode 2: Canal de Luminance (bon pour les claviers colorés)"""
    img_upscaled = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    
    # Threshold simple sur la luminance
    _, binary = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Heuristique : Si l'image est majoritairement noire, on inverse
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
        
    return binary

def method_simple_inversion(img):
    """Méthode 3: Simple inversion (Fallback)"""
    img_upscaled = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(gray)

def get_processed_images(img):
    """Retourne une liste de tuples (nom_methode, image_traitée)"""
    return [
        ("Adaptive", method_adaptive_threshold(img)),
        ("LAB", method_lab_channel(img)),
        ("Inverted", method_simple_inversion(img))
    ]