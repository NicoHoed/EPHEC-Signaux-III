import os

# --- Configuration Globale des Chemins ---
# Image par défaut pour les tests rapides
IMAGE_PATH_DEFAULT = 'data/inputs/ISO-WIN-AZERTY-9.png' 

# --- Paramètres Tesseract OCR ---
# Configuration pour lire 1 seul caractère (PSM 10) ou un mot court (PSM 8/7)
# On whitelist uniquement les majuscules pour éviter de lire des symboles bizarres
OCR_CONFIG = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
OCR_CONFIG_OS = r'--oem 3 --psm 7' # Pour lire des mots comme "CTRL", "CMD", "ALT"

# --- Paramètres de Détection de Touches (V1) ---
AIRE_MIN = 100
AIRE_MAX = 500000
RATIO_MIN = 0.5      
RATIO_MAX = 8.0      
SEUIL_Y_PROXIMITE = 1000

# --- Seuils de Classification (Géométrique V1) ---
THRESHOLD_ENTER_RATIO_H_L_ISO = 1.2 # H/L > 1.2 => Enter ISO (L inversé)