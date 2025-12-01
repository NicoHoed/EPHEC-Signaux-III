#!/usr/bin/env python3
"""
Détecteur de Layout Clavier - VERSION VALIDATION
Exécute le traitement par lots et compare avec la vérité terrain (noms de fichiers).
"""
import argparse
import time
from pathlib import Path
from collections import Counter
import cv2  # <--- L'import manquant était ici !

# Imports des modules personnalisés
from src import utils, ocr_engine, classifier, advanced_preprocessing

def parse_ground_truth(filename):
    """
    Extrait le vrai layout depuis le nom du fichier.
    Format attendu: FORMAT-OS-LAYOUT-X.png (ex: ANSI-WIN-AZERTY-1.png)
    """
    try:
        parts = filename.split('-')
        if len(parts) >= 3:
            layout_tag = parts[2].upper()
            if 'QWERTY' in layout_tag: return 'QWERTY'
            if 'QWERTZ' in layout_tag: return 'QWERTZ'
            if 'AZERTY' in layout_tag: return 'AZERTY'
            return layout_tag
        return 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'

def process_single_image_v2(image_path, output_path, processed_path, 
                            save_debug=False, verbose=False, use_smart_roi=True):
    filename = image_path.name
    
    # 1. Chargement & ROI (Inchangé)
    image = utils.load_image(image_path)
    if image is None: return {'filename': filename, 'detected_layout': 'ERROR'}
    
    normalized = utils.normalize_resolution(image)
    roi = advanced_preprocessing.extract_smart_roi(normalized) # Utilise la version "Large"
    
    # 2. Prétraitement (Binaire)
    binary = advanced_preprocessing.preprocess_for_text_ocr(roi)
    
    # 3. ZONING (Découpage Spatial) [Cite: 1]
    # On découpe l'image en 3 tiers verticaux
    h, w = binary.shape[:2]
    w_3 = w // 3
    
    # Zone GAUCHE (Contient Q, W, A, Z majeurs)
    zone_left = binary[:, 0:w_3]
    # Zone CENTRE (Contient T, Y, U, H, J...)
    zone_center = binary[:, w_3:2*w_3]
    # Zone DROITE (Contient M, P, L...)
    zone_right = binary[:, 2*w_3:]
    
    if save_debug:
        utils.save_image(zone_left, processed_path, f"{Path(filename).stem}_L.png")
        utils.save_image(zone_center, processed_path, f"{Path(filename).stem}_C.png")
    
    # 4. OCR PAR ZONE (Avec inversion automatique intégrée si besoin)
    # On définit une petite fonction locale pour tester Normal + Inversé
    def smart_read(img_zone):
        # Version noire
        res_a = ocr_engine.get_best_ocr_result([('A', img_zone)])[2]
        # Version blanche (inversée)
        res_b = ocr_engine.get_best_ocr_result([('B', cv2.bitwise_not(img_zone))])[2]
        # On combine tout
        return " ".join(res_a + res_b)

    text_left = smart_read(zone_left)
    text_center = smart_read(zone_center)
    text_right = smart_read(zone_right)
    
    full_text_debug = f"L[{text_left}] C[{text_center}] R[{text_right}]"

    # 5. Classification Spatiale
    layout, confidence, scores = classifier.classify_layout_zoned(
        text_left, text_center, text_right, verbose=verbose
    )
    
    return {
        'filename': filename,
        'detected_layout': layout,
        'detected_chars': full_text_debug[:25], 
        'confidence': confidence,
        'processing_time': 0
    }

def main():
    parser = argparse.ArgumentParser(description='Détecteur Clavier - Mode Batch & Validation')
    parser.add_argument('--input', type=str, default='data/inputs', help='Dossier images')
    parser.add_argument('--output', type=str, default='data/outputs', help='Dossier résultats')
    parser.add_argument('--save-debug', action='store_true', help='Sauvegarder images debug')
    parser.add_argument('--verbose', action='store_true', help='Logs détaillés')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎹 DÉTECTEUR CLAVIER - BATCH PROCESSING & VALIDATION")
    print("="*70)
    
    output_path, processed_path = utils.create_output_dirs(args.output)
    image_files = utils.get_image_files(args.input)
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans {args.input}")
        return

    stats = {'total': 0, 'success': 0, 'failed': 0, 'errors': 0}
    confusion_matrix = Counter()

    print(f"📂 Traitement de {len(image_files)} images...\n")
    print(f"{'FICHIER':<30} | {'VRAI':<10} | {'DÉTECTÉ':<10} | {'TXT LU':<15} | {'RÉSULTAT'}")
    print("-" * 90)

    for img_path in image_files:
        stats['total'] += 1
        true_layout = parse_ground_truth(img_path.name)
        
        res = process_single_image_v2(
            img_path, output_path, processed_path, 
            save_debug=args.save_debug, verbose=args.verbose
        )
        
        if res.get('detected_layout') == 'ERROR':
            stats['errors'] += 1
            print(f"{res['filename']:<30} | {true_layout:<10} | ERROR      | -              | ⚠️  ERREUR")
            continue

        detected = res['detected_layout']
        text_read = res['detected_chars'][:15].replace('\n', '')
        
        is_success = (detected == true_layout)
        status_icon = "✅ OK" if is_success else "❌ KO"
        if detected == 'UNKNOWN': status_icon = "❓ UNK"
        
        if is_success: stats['success'] += 1
        else: stats['failed'] += 1
            
        confusion_matrix[(true_layout, detected)] += 1
        
        print(f"{res['filename']:<30} | {true_layout:<10} | {detected:<10} | {text_read:<15} | {status_icon}")

    print("\n" + "="*70)
    print("📊 RAPPORT DE PERFORMANCE")
    print("="*70)
    
    accuracy = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    print(f"Total images : {stats['total']}")
    print(f"✅ Corrects   : {stats['success']}")
    print(f"❌ Incorrects : {stats['failed']}")
    print(f"🎯 PRÉCISION  : {accuracy:.2f}%")
    
    print("\n🔍 ANALYSE DES ERREURS (Vrai -> Détecté) :")
    for (truth, pred), count in confusion_matrix.most_common():
        if truth != pred:
            print(f"   • {truth} confondu avec {pred} : {count} fois")
            
    print(f"\n📂 Résultats détaillés sauvegardés dans : {output_path}")

if __name__ == "__main__":
    main()