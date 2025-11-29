#!/usr/bin/env python3
"""
Test rapide sur UNE image avec visualisation complète
"""
import sys
import cv2
from pathlib import Path

# Import direct depuis les modules
from src.utils import load_image, normalize_resolution, extract_roi
from src.preprocessing import preprocess_multipass, invert_if_needed, enhance_for_ocr
from src.ocr_engine import process_image_multipass, vote_on_results, extract_key_sequence
from src.classifier import classify_layout


def quick_test(image_path):
    """
    Test complet sur une image avec affichage détaillé
    """
    print("\n" + "="*60)
    print(f"🧪 TEST RAPIDE: {Path(image_path).name}")
    print("="*60)
    
    # 1. Charger
    print("\n1️⃣ Chargement...")
    image = load_image(image_path)
    if image is None:
        print("❌ Erreur de chargement")
        return
    
    h, w = image.shape[:2]
    print(f"   Dimensions: {w}x{h}")
    
    # 2. Normaliser
    print("\n2️⃣ Normalisation...")
    normalized = normalize_resolution(image, target_width=1200)
    
    # 3. Extraire ROI
    print("\n3️⃣ Extraction ROI (rangée de lettres)...")
    roi = extract_roi(normalized, roi_type="top_row")
    roi_h, roi_w = roi.shape[:2]
    print(f"   ROI: {roi_w}x{roi_h}")
    
    # Sauvegarder la ROI
    cv2.imwrite("test_roi.png", roi)
    print(f"   💾 Sauvegardé: test_roi.png")
    
    # 4. Prétraiter
    print("\n4️⃣ Prétraitement (version C optimisée)...")
    preprocessed_versions = preprocess_multipass(roi)
    
    # Sauvegarder la version C
    version_c = preprocessed_versions[0][1]  # Première version = C
    cv2.imwrite("test_preprocessed.png", version_c)
    print(f"   💾 Sauvegardé: test_preprocessed.png")
    
    # 5. Post-traitement
    print("\n5️⃣ Post-traitement...")
    enhanced_versions = []
    for version_name, version_img in preprocessed_versions:
        inverted = invert_if_needed(version_img)
        enhanced = enhance_for_ocr(inverted)
        enhanced_versions.append((version_name, enhanced))
    
    # Sauvegarder la version finale
    final = enhanced_versions[0][1]
    cv2.imwrite("test_final.png", final)
    print(f"   💾 Sauvegardé: test_final.png")
    
    # 6. OCR
    print("\n6️⃣ Reconnaissance OCR...")
    all_ocr_results = process_image_multipass(enhanced_versions)
    
    print(f"   🔍 Résultats OCR bruts ({len(all_ocr_results)}): {all_ocr_results}")
    
    # Vote
    best_result, votes, total_votes = vote_on_results(all_ocr_results)
    ocr_confidence = (votes / total_votes * 100) if total_votes > 0 else 0
    
    # Extraire séquence
    detected_text = extract_key_sequence(best_result)
    
    print(f"\n   📝 Texte détecté: '{detected_text}'")
    print(f"   📊 Confiance OCR: {ocr_confidence:.1f}% (votes: {votes}/{total_votes})")
    
    # 7. Classification
    print("\n7️⃣ Classification...")
    layout, final_confidence, scores = classify_layout(
        detected_text,
        ocr_confidence,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"🎯 RÉSULTAT FINAL")
    print(f"{'='*60}")
    print(f"Layout détecté: {layout}")
    print(f"Confiance: {final_confidence}%")
    print(f"Caractères: '{detected_text}'")
    print(f"Scores: {scores}")
    print(f"{'='*60}")
    
    print(f"\n📁 Fichiers générés:")
    print(f"   - test_roi.png          → ROI extraite")
    print(f"   - test_preprocessed.png → Version C (optimisée)")
    print(f"   - test_final.png        → Envoyé à l'OCR")
    print()
    
    return layout, final_confidence, detected_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Usage: python quick_test.py <chemin_image.png>")
        print("\nExemple:")
        print("  python quick_test.py data/inputs/ISO-WIN-AZERTY-1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"\n❌ Fichier introuvable: {image_path}")
        sys.exit(1)
    
    quick_test(image_path)