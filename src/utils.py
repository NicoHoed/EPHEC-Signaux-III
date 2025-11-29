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
    """
    Crée les dossiers de sortie nécessaires
    
    Args:
        base_output_path: Chemin de base pour les sorties
    """
    output_path = Path(base_output_path)
    processed_path = output_path / "processed"
    
    output_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    return output_path, processed_path


def load_image(image_path):
    """
    Charge une image depuis un chemin
    
    Args:
        image_path: Chemin vers l'image
        
    Returns:
        Image numpy array ou None si erreur
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Impossible de charger: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {image_path}: {e}")
        return None


def save_image(image, save_path, filename):
    """
    Sauvegarde une image
    
    Args:
        image: Image numpy array
        save_path: Dossier de destination
        filename: Nom du fichier
    """
    try:
        full_path = Path(save_path) / filename
        cv2.imwrite(str(full_path), image)
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde de {filename}: {e}")


def normalize_resolution(image, target_width=1200):
    """
    Normalise la résolution de l'image
    
    Args:
        image: Image numpy array
        target_width: Largeur cible en pixels
        
    Returns:
        Image redimensionnée
    """
    height, width = image.shape[:2]
    if width == target_width:
        return image
    
    ratio = target_width / width
    new_height = int(height * ratio)
    
    resized = cv2.resize(image, (target_width, new_height), 
                        interpolation=cv2.INTER_LANCZOS4)
    return resized


def extract_roi(image, roi_type="top_row", adaptive=True, verbose=False):
    """
    Extrait la région d'intérêt de l'image
    
    Args:
        image: Image numpy array
        roi_type: Type de ROI à extraire
        adaptive: Si True, adapte selon les dimensions du clavier
        verbose: Afficher les infos
        
    Returns:
        ROI extraite
    """
    height, width = image.shape[:2]
    
    if roi_type == "top_row":
        if adaptive:
            # Détection adaptative du type de clavier
            ratio = width / height
            
            if verbose:
                print(f"   Ratio w/h: {ratio:.2f}")
            
            # Ajuster selon le ratio
            if ratio > 3.5:  # Clavier complet
                y_start_pct, y_end_pct, x_end_pct = 0.30, 0.45, 0.55
                ktype = "Full (avec pavé num)"
            elif 2.8 < ratio <= 3.5:  # TKL
                y_start_pct, y_end_pct, x_end_pct = 0.30, 0.45, 0.65
                ktype = "TKL (sans pavé num)"
            elif 2.0 < ratio <= 2.8:  # Compact
                y_start_pct, y_end_pct, x_end_pct = 0.28, 0.47, 0.75
                ktype = "Compact (60-75%)"
            else:  # Autre
                y_start_pct, y_end_pct, x_end_pct = 0.30, 0.45, 0.70
                ktype = "Standard"
            
            if verbose:
                print(f"   Type: {ktype}")
                print(f"   Zone: y[{y_start_pct*100:.0f}%-{y_end_pct*100:.0f}%], x[0-{x_end_pct*100:.0f}%]")
        else:
            # Mode classique (non-adaptatif)
            y_start_pct, y_end_pct, x_end_pct = 0.30, 0.45, 0.70
        
        y_start = int(height * y_start_pct)
        y_end = int(height * y_end_pct)
        x_end = int(width * x_end_pct)
        
        roi = image[y_start:y_end, 0:x_end]
        return roi
    
    return image


def calculate_confidence(votes, total_votes):
    """
    Calcule le score de confiance basé sur les votes
    
    Args:
        votes: Nombre de votes pour le résultat majoritaire
        total_votes: Nombre total de votes
        
    Returns:
        Score de confiance en pourcentage
    """
    if total_votes == 0:
        return 0
    return int((votes / total_votes) * 100)


def generate_report(results, output_path):
    """
    Génère un rapport JSON des résultats
    
    Args:
        results: Liste des résultats de classification
        output_path: Chemin du dossier de sortie
    """
    total = len(results)
    successful = sum(1 for r in results if r['detected_layout'] != 'UNKNOWN')
    failed = total - successful
    accuracy = (successful / total * 100) if total > 0 else 0
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": total,
            "successful": successful,
            "failed": failed,
            "accuracy": f"{accuracy:.2f}%"
        },
        "results": results
    }
    
    report_path = Path(output_path) / "report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Rapport sauvegardé: {report_path}")
    return report


def print_summary(report):
    """
    Affiche un résumé des résultats
    
    Args:
        report: Dictionnaire du rapport
    """
    summary = report['summary']
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    print(f"Total d'images traitées: {summary['total_images']}")
    print(f"✅ Succès: {summary['successful']}")
    print(f"❌ Échecs: {summary['failed']}")
    print(f"🎯 Précision: {summary['accuracy']}")
    print("="*60)
    
    # Distribution des layouts
    layouts = {}
    for result in report['results']:
        layout = result['detected_layout']
        layouts[layout] = layouts.get(layout, 0) + 1
    
    print("\n📈 Distribution des layouts détectés:")
    for layout, count in sorted(layouts.items()):
        print(f"  {layout}: {count}")
    print()


def get_image_files(input_path):
    """
    Récupère tous les fichiers PNG du dossier d'entrée
    
    Args:
        input_path: Chemin du dossier d'entrée
        
    Returns:
        Liste des chemins d'images
    """
    input_path = Path(input_path)
    image_files = list(input_path.glob("*.png"))
    image_files.extend(input_path.glob("*.PNG"))
    
    return sorted(image_files)