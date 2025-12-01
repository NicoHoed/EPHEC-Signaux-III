"""
Classification Spatiale (Zoned Classification)
"""
import re

def classify_layout_zoned(text_left, text_center, text_right, verbose=False):
    """
    Utilise la position (Gauche/Centre/Droite) pour pondérer les lettres.
    """
    
    scores = {'AZERTY': 0, 'QWERTY': 0, 'QWERTZ': 0}
    
    # --- ANALYSE DE LA ZONE GAUCHE (CRITIQUE) ---
    # C'est là que tout se joue : Q, W, A, Z sont tous à gauche.
    
    # AZERTY commence par A et Z
    if 'A' in text_left: scores['AZERTY'] += 60  # A à gauche = AZERTY quasi sûr
    if 'Z' in text_left: scores['AZERTY'] += 40  # Z à gauche = AZERTY (ou QWERTZ bas)
    
    # QWERTY commence par Q et W
    if 'Q' in text_left: 
        scores['QWERTY'] += 50
        scores['QWERTZ'] += 50
    if 'W' in text_left: 
        scores['QWERTY'] += 50
        scores['QWERTZ'] += 50

    # --- ANALYSE DE LA ZONE CENTRE ---
    
    # En QWERTY, le A est au milieu (ligne ASDF)
    if 'A' in text_center: 
        scores['QWERTY'] += 30
        scores['QWERTZ'] += 30
        scores['AZERTY'] -= 20 # Si A est au milieu, ce n'est pas AZERTY

    # En QWERTY/Z, le Z peut être en bas à gauche (Zone Left) ou milieu (Zone Center limit)
    # En AZERTY, le Z est toujours en haut à gauche.
    
    # Le Y est crucial pour QWERTY vs QWERTZ
    # QWERTY : Y au milieu-droit (Zone Center ou Right)
    # QWERTZ : Z au milieu-droit
    # AZERTY : Y au milieu-droit
    
    if 'Z' in text_center or 'Z' in text_right:
        scores['QWERTZ'] += 60 # Z au milieu/droite = QWERTZ (Touche 6)
        scores['QWERTY'] -= 30 # Impossible en QWERTY (Z est à gauche)
        scores['AZERTY'] -= 30 # Impossible en AZERTY (Z est à gauche)
        
    if 'Y' in text_left:
         # Y à gauche est très rare (QWERTZ bas gauche), souvent erreur de lecture
         pass
         
    # --- RÈGLES D'EXCLUSION SPATIALE ---
    
    # Conflit A : Si j'ai A à gauche ET A au centre -> Priorité Gauche (AZERTY)
    if 'A' in text_left and 'A' in text_center:
        scores['AZERTY'] += 20
        
    # Conflit Z : Si Z à gauche -> AZERTY ou QWERTY. 
    # Mais si Q est aussi à gauche -> QWERTY/Z.
    if 'Z' in text_left and 'Q' in text_left:
        scores['QWERTY'] += 50
        scores['QWERTZ'] += 50
        scores['AZERTY'] -= 50 # AZERTY n'a pas de Q à gauche (il est au centre sur la ligne A)
        
    # --- DÉPARTAGE FINAL ---
    
    best_layout = max(scores, key=scores.get)
    best_score = scores[best_layout]
    
    # Normalisation 100
    best_score = min(best_score, 100)
    
    if best_score < 30:
        return 'UNKNOWN', 0, scores
        
    return best_layout, best_score, scores