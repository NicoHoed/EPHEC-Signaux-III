"""
Moteur OCR pour la reconnaissance de caractères sur les touches de clavier
"""
import pytesseract
import re
from collections import Counter


# Configuration de Tesseract (à ajuster selon ton installation)
# Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def clean_ocr_result(text):
    """
    Nettoie et corrige les erreurs fréquentes de l'OCR
    """
    if not text: return ""
    
    text = text.upper()
    
    # 1. Remplacements des confusions fréquentes (Leet speak inverse)
    replacements = {
        '0': 'O', '1': 'I', '5': 'S', '2': 'Z', 
        '8': 'B', '|': 'I', '[': '', ']': '', 
        '{': '', '}': '', '(': '', ')': '',
        '$': 'S', '€': 'E'
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
        
    # 2. Garder uniquement les lettres A-Z
    text = re.sub(r'[^A-Z]', '', text)
    
    return text


def ocr_multiconfig(image):
    # On utilise PSM 11 (Sparse Text) qui est très permissif
    configs = ['--psm 11 --oem 3'] 
    
    results = []
    # Pas de whitelist restrictive ! Laissez passer les symboles pour le fallback
    custom_config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ[]" 
    
    for config in configs:
        full_config = f"{config} {custom_config}"
        try:
            text = pytesseract.image_to_string(image, config=full_config)
            # Nettoyage minimal : on garde tout pour l'instant
            clean = text.replace('\n', ' ').strip()
            if len(clean) > 2:
                results.append(clean)
        except:
            pass
            
    return results

def ocr_with_config(image, config):
    # Simplification : on prend la config telle quelle
    try:
        text = pytesseract.image_to_string(image, config=config)
        return clean_ocr_result(text)
    except Exception:
        return ""


def process_image_multipass(preprocessed_versions):
    """
    Traite toutes les versions prétraitées avec toutes les configs OCR
    
    Args:
        preprocessed_versions: Liste de tuples (nom, image)
        
    Returns:
        Liste de tous les résultats OCR
    """
    all_results = []
    
    for version_name, image in preprocessed_versions:
        # OCR avec configurations multiples
        ocr_results = ocr_multiconfig(image)
        all_results.extend(ocr_results)
    
    return all_results


def vote_on_results(ocr_results):
    """
    Système de vote pour déterminer le meilleur résultat
    
    Args:
        ocr_results: Liste de chaînes OCR
        
    Returns:
        Tuple (résultat_majoritaire, nombre_votes, total_votes)
    """
    if not ocr_results:
        return "", 0, 0
    
    # Filtrer les résultats trop courts (bruit)
    valid_results = [r for r in ocr_results if len(r) >= 3]
    
    if not valid_results:
        return "", 0, 0
    
    # Compter les occurrences
    counter = Counter(valid_results)
    most_common = counter.most_common(1)[0]
    
    best_result = most_common[0]
    votes = most_common[1]
    total_votes = len(valid_results)
    
    return best_result, votes, total_votes


def extract_key_sequence(text, max_length=6):
    """
    Extrait une séquence de touches depuis le texte OCR
    Cherche les 6 premières lettres consécutives
    
    Args:
        text: Texte OCR nettoyé
        max_length: Longueur maximale de la séquence
        
    Returns:
        Séquence de touches (6 caractères max)
    """
    if not text:
        return ""
    
    # Prendre les N premiers caractères
    return text[:max_length]


def get_best_ocr_result(preprocessed_versions, verbose=False):
    """
    Fonction principale pour obtenir le meilleur résultat OCR
    
    Args:
        preprocessed_versions: Liste des versions prétraitées
        verbose: Si True, affiche les détails
        
    Returns:
        Tuple (meilleur_résultat, confiance, tous_résultats)
    """
    # Obtenir tous les résultats OCR
    all_ocr_results = process_image_multipass(preprocessed_versions)
    
    if verbose:
        print(f"  🔍 Résultats OCR bruts ({len(all_ocr_results)}): {all_ocr_results}")
    
    # Vote pour déterminer le meilleur résultat
    best_result, votes, total_votes = vote_on_results(all_ocr_results)
    
    # Calculer la confiance
    confidence = (votes / total_votes * 100) if total_votes > 0 else 0
    
    # Extraire la séquence de touches clés
    key_sequence = extract_key_sequence(best_result)
    
    if verbose:
        print(f"  🗳️  Meilleur résultat: '{key_sequence}' (votes: {votes}/{total_votes}, confiance: {confidence:.1f}%)")
    
    return key_sequence, confidence, all_ocr_results