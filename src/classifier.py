"""
Module de classification de layout clavier
"""


# Patterns de référence pour chaque layout
LAYOUT_PATTERNS = {
    'QWERTY': {
        'full': ['QWERTY', 'QWERT', 'QWER'],
        'key_positions': {
            0: 'Q',  # Position 0 doit être Q
            1: 'W',  # Position 1 doit être W
            5: 'Y'   # Position 5 doit être Y (différencie de QWERTZ)
        }
    },
    'QWERTZ': {
        'full': ['QWERTZ', 'QWERT', 'QWER'],
        'key_positions': {
            0: 'Q',  # Position 0 doit être Q
            1: 'W',  # Position 1 doit être W
            5: 'Z'   # Position 5 doit être Z (différencie de QWERTY)
        }
    },
    'AZERTY': {
        'full': ['AZERTY', 'AZERT', 'AZER'],
        'key_positions': {
            0: 'A',  # Position 0 doit être A (différencie de QWERTY/QWERTZ)
            1: 'Z',  # Position 1 doit être Z
            5: 'Y'   # Position 5 doit être Y
        }
    }
}


def calculate_pattern_score(detected_text, layout_name):
    """
    Calcule un score de correspondance pour un layout donné
    
    Args:
        detected_text: Texte détecté par OCR
        layout_name: Nom du layout à tester
        
    Returns:
        Score de correspondance (0-100)
    """
    if not detected_text:
        return 0
    
    pattern = LAYOUT_PATTERNS[layout_name]
    score = 0
    
    # Test 1: Correspondance avec les patterns complets
    for full_pattern in pattern['full']:
        if full_pattern in detected_text:
            # Match exact
            if detected_text.startswith(full_pattern):
                score += 100
                break
            else:
                score += 80
                break
        elif detected_text.startswith(full_pattern[:4]):
            # Match partiel (4+ caractères)
            score += 60
            break
        elif detected_text.startswith(full_pattern[:3]):
            # Match court (3 caractères)
            score += 40
            break
    
    # Test 2: Vérification des positions clés
    key_positions = pattern['key_positions']
    position_score = 0
    positions_checked = 0
    
    for pos, expected_char in key_positions.items():
        if pos < len(detected_text):
            positions_checked += 1
            if detected_text[pos] == expected_char:
                position_score += 20
    
    # Ajouter le score de position (moyenne)
    if positions_checked > 0:
        score += position_score
    
    # Bonus si toutes les positions clés sont correctes
    if positions_checked == len(key_positions) and position_score == len(key_positions) * 20:
        score += 10
    
    return min(score, 100)  # Cap à 100


def classify_layout(detected_text, ocr_confidence, verbose=False):
    """
    Classifie le layout du clavier
    
    Args:
        detected_text: Texte détecté par OCR
        ocr_confidence: Confiance de l'OCR (0-100)
        verbose: Si True, affiche les détails
        
    Returns:
        Tuple (layout_name, final_confidence, scores_detail)
    """
    if not detected_text or len(detected_text) < 3:
        return 'UNKNOWN', 0, {}
    
    # Calculer les scores pour chaque layout
    scores = {}
    for layout_name in LAYOUT_PATTERNS.keys():
        score = calculate_pattern_score(detected_text, layout_name)
        scores[layout_name] = score
    
    if verbose:
        print(f"  📊 Scores de correspondance:")
        for layout, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"     {layout}: {score}")
    
    # Trouver le meilleur score
    best_layout = max(scores, key=scores.get)
    best_score = scores[best_layout]
    
    # Si le score est trop faible, retourner UNKNOWN
    if best_score < 40:
        return 'UNKNOWN', 0, scores
    
    # Calculer la confiance finale (moyenne pondérée)
    # 60% score de pattern, 40% confiance OCR
    final_confidence = int((best_score * 0.6) + (ocr_confidence * 0.4))
    
    return best_layout, final_confidence, scores


def analyze_ambiguous_cases(detected_text, scores):
    """
    Analyse les cas ambigus où plusieurs layouts ont des scores proches
    
    Args:
        detected_text: Texte détecté
        scores: Dictionnaire des scores
        
    Returns:
        Analyse textuelle
    """
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_scores) < 2:
        return "Pas assez de données pour comparaison"
    
    best_layout, best_score = sorted_scores[0]
    second_layout, second_score = sorted_scores[1]
    
    diff = best_score - second_score
    
    if diff < 20:
        return f"Ambiguïté détectée entre {best_layout} et {second_layout} (diff: {diff})"
    else:
        return f"Classification claire: {best_layout} domine"


def get_discriminating_keys(layout1, layout2):
    """
    Retourne les touches discriminantes entre deux layouts
    
    Args:
        layout1: Premier layout
        layout2: Deuxième layout
        
    Returns:
        Dictionnaire des différences
    """
    if layout1 not in LAYOUT_PATTERNS or layout2 not in LAYOUT_PATTERNS:
        return {}
    
    keys1 = LAYOUT_PATTERNS[layout1]['key_positions']
    keys2 = LAYOUT_PATTERNS[layout2]['key_positions']
    
    differences = {}
    for pos in set(keys1.keys()) | set(keys2.keys()):
        char1 = keys1.get(pos, '?')
        char2 = keys2.get(pos, '?')
        if char1 != char2:
            differences[pos] = {'layout1': char1, 'layout2': char2}
    
    return differences