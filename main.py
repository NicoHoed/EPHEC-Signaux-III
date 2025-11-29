#!/usr/bin/env python3
"""
Détecteur de Layout Clavier - VERSION 2 AMÉLIORÉE
Avec détection automatique des touches et prétraitement avancé
"""
import argparse
import time
from pathlib import Path

# Imports des modules personnalisés
from src import utils, ocr_engine, classifier, advanced_preprocessing


def process_single_image_v2(image_path, output_path, processed_path, 
                            save_debug=False, verbose=False, use_smart_roi=True):
    """
    Traite une seule image avec la version 2 améliorée
    
    Args:
        image_path: Chemin de l'image
        output_path: Dossier de sortie
        processed_path: Dossier pour images prétraitées
        save_debug: Si True, sauvegarde les images intermédiaires
        verbose: Si True, affiche les détails
        use_smart_roi: Si True, utilise la détection intelligente de ROI
        
    Returns:
        Dictionnaire avec les résultats
    """
    filename = image_path.name
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🖼️  Traitement: {filename}")
        print(f"{'='*60}")
    else:
        print(f"🖼️  {filename}...", end=" ", flush=True)
    
    # 1. Chargement de l'image
    image = utils.load_image(image_path)
    if image is None:
        result = {
            'filename': filename,
            'detected_layout': 'ERROR',
            'confidence': 0,
            'detected_chars': '',
            'processing_time': 0,
            'error': 'Failed to load image'
        }
        if not verbose:
            print("❌ ERREUR")
        return result
    
    # 2. Normalisation de la résolution
    if verbose:
        print("📐 Normalisation de la résolution...")
    normalized = utils.normalize_resolution(image)
    
    # 3. Extraction de la ROI (intelligente ou classique)
    if verbose:
        print(f"🔍 Extraction de la zone d'intérêt (mode: {'intelligent' if use_smart_roi else 'classique'})...")
    
    if use_smart_roi:
        roi = advanced_preprocessing.extract_smart_roi(normalized)
    else:
        roi = utils.extract_roi(normalized, roi_type="top_row")
    
    if save_debug:
        utils.save_image(roi, processed_path, f"{Path(filename).stem}_roi.png")
    
    # 4. Prétraitement avancé
    if verbose:
        print("🎨 Prétraitement avancé...")
    
    preprocessed_final = advanced_preprocessing.preprocess_for_text_ocr(roi)
    
    if save_debug:
        utils.save_image(preprocessed_final, processed_path, 
                        f"{Path(filename).stem}_preprocessed.png")
    
    # 5. OCR avec configurations multiples
    if verbose:
        print("🔤 Reconnaissance OCR...")
    
    # Créer des "versions" pour compatibilité avec ocr_engine
    versions = [
        ('advanced', preprocessed_final),
        ('advanced', preprocessed_final),  # Dupliquer pour avoir plus de votes
        ('advanced', preprocessed_final),
    ]
    
    detected_text, ocr_confidence, all_ocr = ocr_engine.get_best_ocr_result(
        versions,
        verbose=verbose
    )
    
    # 6. Classification du layout
    if verbose:
        print("🎯 Classification du layout...")
    layout, final_confidence, scores = classifier.classify_layout(
        detected_text,
        ocr_confidence,
        verbose=verbose
    )
    
    # Temps de traitement
    processing_time = time.time() - start_time
    
    # Résultat
    result = {
        'filename': filename,
        'detected_layout': layout,
        'confidence': final_confidence,
        'detected_chars': detected_text,
        'processing_time': f"{processing_time:.2f}s",
        'ocr_confidence': int(ocr_confidence),
        'pattern_scores': scores
    }
    
    if verbose:
        print(f"\n✅ Résultat: {layout} (confiance: {final_confidence}%)")
        print(f"⏱️  Temps: {processing_time:.2f}s")
    else:
        # Affichage compact
        emoji = "✅" if layout != "UNKNOWN" else "❓"
        print(f"{emoji} {layout} ({final_confidence}%) - '{detected_text}'")
    
    return result


def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(
        description='Détecteur de Layout Clavier V2 - QWERTY/QWERTZ/AZERTY (Amélioré)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/inputs',
        help='Dossier contenant les images PNG (défaut: data/inputs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/outputs',
        help='Dossier de sortie pour les résultats (défaut: data/outputs)'
    )
    parser.add_argument(
        '--save-debug',
        action='store_true',
        help='Sauvegarder les images prétraitées pour débogage'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Afficher les détails du traitement'
    )
    parser.add_argument(
        '--no-smart-roi',
        action='store_true',
        help='Désactiver la détection intelligente de ROI'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=int,
        default=60,
        help='Seuil de confiance minimal (défaut: 60%%)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*60)
    print("🎹 DÉTECTEUR DE LAYOUT CLAVIER V2 (Amélioré)")
    print("="*60)
    
    # Création des dossiers de sortie
    output_path, processed_path = utils.create_output_dirs(args.output)
    
    # Récupération des fichiers images
    image_files = utils.get_image_files(args.input)
    
    if not image_files:
        print(f"\n❌ Aucune image PNG trouvée dans: {args.input}")
        return
    
    print(f"\n📁 Dossier d'entrée: {args.input}")
    print(f"📁 Dossier de sortie: {args.output}")
    print(f"🖼️  Nombre d'images: {len(image_files)}")
    print(f"🧠 ROI intelligente: {'Activée' if not args.no_smart_roi else 'Désactivée'}")
    
    if args.save_debug:
        print(f"🔧 Mode debug: images prétraitées seront sauvegardées")
    
    print(f"\n🚀 Démarrage du traitement...\n")
    
    # Traitement de toutes les images
    all_results = []
    
    for image_path in image_files:
        result = process_single_image_v2(
            image_path,
            output_path,
            processed_path,
            save_debug=args.save_debug,
            verbose=args.verbose,
            use_smart_roi=not args.no_smart_roi
        )
        all_results.append(result)
    
    # Génération du rapport
    print(f"\n📝 Génération du rapport...")
    report = utils.generate_report(all_results, output_path)
    
    # Affichage du résumé
    utils.print_summary(report)
    
    # Statistiques supplémentaires
    low_confidence = [r for r in all_results 
                     if r['detected_layout'] != 'UNKNOWN' 
                     and r['confidence'] < args.confidence_threshold]
    
    if low_confidence:
        print(f"\n⚠️  {len(low_confidence)} image(s) avec confiance < {args.confidence_threshold}%:")
        for result in low_confidence:
            print(f"   - {result['filename']}: {result['detected_layout']} ({result['confidence']}%)")
    
    print("\n✨ Traitement terminé!")
    print(f"📊 Voir le rapport complet: {output_path / 'report.json'}")
    
    if args.save_debug:
        print(f"🔧 Images debug: {processed_path}")
    
    print()


if __name__ == "__main__":
    main()