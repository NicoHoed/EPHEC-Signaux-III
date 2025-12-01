"""
--------------------------------------------------------------------------------
File: src/engine.py
Author:
    Nicolas HOEDENAEKEN
    Théo MERTENS
    Baris OZCELIK
    Khassan AKTAMIROV
    
Description: 
    Cœur logique de l'application. Contient :
    1. Le pipeline OCR (EasyOCR) avec gestion multi-images.
    2. L'algorithme de Clustering (K-Means) pour identifier les rangées physiques
       indépendamment de l'angle de la photo.
    3. Le système de Scoring pondéré pour classifier le layout (AZERTY/QWERTY...).
--------------------------------------------------------------------------------
"""

import easyocr
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import warnings

# Mapping pour corriger les erreurs fréquentes d'OCR
OCR_CORRECTIONS = {
    '0': 'O', '1': 'I', '5': 'S', '2': 'Z', '4': 'A', '8': 'B', 
    '|': 'I', '$': 'S', '€': 'E', '(': 'C', '[': 'C', '{': 'C'
}

# --- DÉFINITION INTELLIGENTE DES LAYOUTS ---
# On ne définit plus des lignes entières rigides, mais des "Marqueurs forts"
LAYOUT_RULES = {
    "AZERTY": {
        # Si je trouve ces lettres en HAUT, c'est +++ pour AZERTY
        "TOP":  {'A', 'Z', 'E', 'R', 'T'}, 
        # Si je trouve ces lettres au MILIEU, c'est +++ pour AZERTY
        "MID":  {'Q', 'S', 'D', 'F', 'G', 'M'}, # M est au milieu en AZERTY (souvent)
        # Si je trouve ces lettres en BAS, c'est +++ pour AZERTY
        "BOT":  {'W', 'X', 'C', 'V'}
    },
    "QWERTY": {
        "TOP":  {'Q', 'W', 'E', 'R', 'T', 'Y'},
        "MID":  {'A', 'S', 'D', 'F', 'G'},
        "BOT":  {'Z', 'X', 'C', 'V'}
    },
    "QWERTZ": {
        "TOP":  {'Q', 'W', 'E', 'R', 'T', 'Z'},
        "MID":  {'A', 'S', 'D', 'F', 'G'},
        "BOT":  {'Y', 'X', 'C', 'V'}
    }
}

def clean_char(text):
    text = text.upper().strip()
    # On nettoie les caractères non alphanumériques sauf s'ils ressemblent à des lettres
    text = ''.join(filter(str.isalnum, text))
    if len(text) != 1: return ""
    return OCR_CORRECTIONS.get(text, text)

def run_ocr_pipeline(reader, processed_images):
    char_data = {}

    for method_name, img in processed_images:
        # On utilise une allowlist large
        try:
            results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        except Exception:
            continue
        
        for (bbox, text, conf) in results:
            if conf < 0.3: continue # 0.3 = tolerance
            
            char = clean_char(text)
            if not char or not char.isalpha(): continue

            # Centre Y
            y_center = (bbox[0][1] + bbox[2][1]) / 2

            if char not in char_data:
                char_data[char] = {'y_sum': 0, 'count': 0}
            
            char_data[char]['y_sum'] += y_center
            char_data[char]['count'] += 1

    # On garde les lettres vues au moins 1 fois (pour maximiser les chances)
    validated_chars = {}
    for char, data in char_data.items():
        avg_y = data['y_sum'] / data['count']
        validated_chars[char] = avg_y

    return validated_chars

def cluster_rows(validated_chars):
    """
    K-MEANS 1D : Pour séparer 3 niveaux de hauteur.
    """
    if len(validated_chars) < 4:
        return None

    y_coords = np.array(list(validated_chars.values())).reshape(-1, 1)
    
    # On tente de trouver 3 rangées. Si ça échoue (trop peu de points), on tente 2.
    n_clusters = 3
    if len(y_coords) < 3: n_clusters = len(y_coords)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(y_coords)
        centers = kmeans.cluster_centers_.flatten()
        
        # On trie les centres pour savoir qui est HAUT (petit Y), MILIEU, BAS (grand Y)
        sorted_indices = np.argsort(centers)
        
        # Mapping du cluster ID vers le Row ID (0=Haut, 1=Milieu, 2=Bas)
        map_cluster = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        
        char_rows = {}
        chars_list = list(validated_chars.keys())
        for i, char in enumerate(chars_list):
            original_label = labels[i]
            row_id = map_cluster[original_label]
            char_rows[char] = row_id
            
        return char_rows
    except Exception as e:
        print(f"Erreur clustering: {e}")
        return None

def score_layout(char_to_row_map):
    """
    Nouveau système de scoring pondéré.
    """
    scores = {k: 0 for k in LAYOUT_RULES.keys()}
    
    # On compte combien de lettres pertinentes on a trouvées au total
    relevant_chars_found = 0

    for char, detected_row in char_to_row_map.items():
        # 0=TOP, 1=MID, 2=BOT
        
        relevant_chars_found += 1
        
        for layout_name, rules in LAYOUT_RULES.items():
            
            # --- RÈGLES POSITIVES (Bonus) ---
            if detected_row == 0 and char in rules["TOP"]:
                scores[layout_name] += 10 # Gros bonus si bonne lettre en haut
                # Bonus Spécial AZERTY/QWERTY
                if char in ['A', 'Q', 'Z', 'W']: scores[layout_name] += 15 

            elif detected_row == 1 and char in rules["MID"]:
                scores[layout_name] += 10
                if char in ['A', 'Q', 'M']: scores[layout_name] += 15

            elif detected_row == 2 and char in rules["BOT"]:
                scores[layout_name] += 10
                if char in ['W', 'Z', 'Y', 'M']: scores[layout_name] += 15

            # --- RÈGLES NÉGATIVES (Malus / Contradiction) ---
            # Si je vois un 'Q' en haut, ce N'EST PAS un AZERTY
            if detected_row == 0 and char == 'Q':
                scores['AZERTY'] -= 50
            
            # Si je vois un 'A' en haut, ce N'EST PAS un QWERTY/QWERTZ
            if detected_row == 0 and char == 'A':
                scores['QWERTY'] -= 50
                scores['QWERTZ'] -= 50

            # Si je vois un 'Z' en haut, ce N'EST PAS un QWERTY (Z est en bas)
            if detected_row == 0 and char == 'Z':
                scores['QWERTY'] -= 50
                # C'est soit AZERTY soit QWERTZ

            # Si je vois un 'Y' en bas, c'est probablement QWERTZ
            if detected_row == 2 and char == 'Y':
                scores['QWERTZ'] += 40
                scores['QWERTY'] -= 20 # Y est en haut en QWERTY

    # Normalisation et décision
    if relevant_chars_found == 0:
        return "INCONNU", 0, scores

    best_layout = max(scores, key=scores.get)
    max_score = scores[best_layout]
    
    # Confiance relative (écart avec le deuxième meilleur)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        margin = sorted_scores[0] - sorted_scores[1]
        # Si la marge est grande (>50 pts), confiance max
        confidence = min(100, max(0, 50 + margin)) 
    else:
        confidence = 0

    # Si le score est négatif ou très bas, on doute
    if max_score <= 0:
        return "INCERTAIN", 0, scores

    return best_layout, confidence, scores