# --- Configuration Globale des Chemins ---
# Chemin vers l'image à analyser par défaut (pour main.py et explore_shape.py)
IMAGE_PATH_DEFAULT = 'data/inputs/ANSI-WIN-QWERTZ-1.png' 

# --- Paramètres de Détection de Touche ---
# Ces valeurs sont utilisées dans src/analysis.py pour filtrer les régions détectées.

# 1. Filtres sur l'Aire (en pixels)
# Ajuster pour inclure les touches de fonction (min) et la barre espace (max)
AIRE_MIN = 2000      
AIRE_MAX = 400000    

# 2. Filtres sur le Ratio Largeur/Hauteur (L/H)
# Ajuster pour inclure les touches carrées (min) et la barre espace (max)
RATIO_MIN = 0.5      
RATIO_MAX = 8.0      

# 3. Filtrage Spatial (Anti-Trackpad et centrage clavier)
# Distance max verticale autorisée par rapport au centre Y moyen du clavier.
SEUIL_Y_PROXIMITE = 1000

# ----------------------------------------------------
# --- Paramètres de Classification et de Zoning ---
# ----------------------------------------------------

# 1. ISO vs ANSI (Shift Gauche)
# Ratio Largeur/Hauteur (L/H) de la touche Shift gauche.
THRESHOLD_SHIFT_RATIO_ISO = 2.1 # L/H < 2.1 => ISO (petit)

# 2. Renforcement ISO/ANSI (Touche Entrée)
# Ratio Hauteur/Largeur (H/L) de la touche Enter.
THRESHOLD_ENTER_RATIO_H_L_ISO = 1.2    # H/L > 1.2 => Enter ISO (Grand 'L' inversé)
THRESHOLD_ENTER_RATIO_H_L_ANSI = 1.0   # H/L < 1.0 => Enter ANSI (Rectangle)

# 3. Mac vs Windows (Nombre d'Euler sur touche OS)
THRESHOLD_EULER_MAC = -1
THRESHOLD_EULER_WIN = 1

# 4. AZERTY vs QWERTY (Touche Lettre Haut-Gauche - TL_LETTER: Q/A)
THRESHOLD_TL_CENTER_Y_AZERTY = 0.53
THRESHOLD_TL_EXTENT_AZERTY = 0.80

# --- Paramètres de Zoning ---
# Ce dictionnaire contient les coefficients multiplicatifs pour la hauteur de référence
# utilisés pour définir les couloirs de recherche verticaux (Y-axis) des touches clés.
ZONING_HR_MULTIPLIERS = {
    # Tolérances (facteur pour h_ref)
    "OS_DY_TOLERANCE": 0.8,         # Tolérance dY par rapport à l'Espace pour la touche OS
    
    # Range Y (multiples de h_ref au-dessus de l'espace)
    "SHIFT_Y_MIN_HR": 0.5,          # Bord inférieur du Shift
    "SHIFT_Y_MAX_HR": 2.0,          # Bord supérieur du Shift
    "TL_LETTER_Y_MIN_HR": 2.5,      # Bord inférieur de la rangée Q/A
    "TL_LETTER_Y_MAX_HR": 3.8,      # Bord supérieur de la rangée Q/A
    
    # Nouveau: Paramètres de position pour la touche ENTER
    "ENTER_Y_TARGET_HR": 2.5,       # Position Y cible (en h_ref) du centre de l'Enter
    "ENTER_Y_TOLERANCE_HR": 1.5,    # Tolérance +/- pour le centre de l'Enter
}

# 5. Seuil de ratio pour distinguer la touche Tab du Q/A
THRESHOLD_TAB_RATIO = 1.2