#!/usr/bin/env python3
"""
Script de diagnostic visuel pour comprendre ce que voit l'OCR
"""
import cv2
import sys
from pathlib import Path

# Imports des modules personnalisés
from src import utils, preprocessing


def visualize_processing(image_path, save_path="debug_output"):
    """
    Visualise toutes les étapes de traitement pour une image
    
    Args:
        image_path: Chemin de l'image à analyser
        save_path: Dossier où sauvegarder les visualisations
    """
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    
    filename = Path(image_path).stem
    print(f"\n{'='*60}")
    print(f"🔍 Analyse de: {Path(image_path).name}")
    print(f"{'='*60}\n")
    
    # 1. Chargement
    print("1️⃣  Chargement de l'image...")
    image = utils.load_image(image_path)
    if image is None:
        print("❌ Impossible de charger l'image")
        return
    
    height, width = image.shape[:2]
    print(f"   Dimensions: {width}x{height}")
    
    # Sauvegarder l'original
    cv2.imwrite(str(save_path / f"{filename}_0_original.png"), image)
    
    # 2. Normalisation
    print("\n2️⃣  Normalisation à 1200px de largeur...")
    normalized = utils.normalize_resolution(image, target_width=1200)
    new_height, new_width = normalized.shape[:2]
    print(f"   Nouvelles dimensions: {new_width}x{new_height}")
    cv2.imwrite(str(save_path / f"{filename}_1_normalized.png"), normalized)
    
    # 3. Extraction ROI avec marqueurs visuels
    print("\n3️⃣  Extraction de la ROI (Zone d'Intérêt)...")
    
    # Créer une copie pour dessiner les zones
    marked_image = normalized.copy()
    h, w = normalized.shape[:2]
    
    # Marquer la zone ROI
    y_start = int(h * 0.20)
    y_end = int(h * 0.35)
    x_end = int(w * 0.70)
    
    # Dessiner un rectangle rouge sur la zone ROI
    cv2.rectangle(marked_image, (0, y_start), (x_end, y_end), (0, 0, 255), 3)
    cv2.putText(marked_image, "ROI", (10, y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    print(f"   Zone ROI: x[0:{x_end}], y[{y_start}:{y_end}]")
    print(f"   Soit: {(y_start/h)*100:.1f}% - {(y_end/h)*100:.1f}% de hauteur")
    cv2.imwrite(str(save_path / f"{filename}_2_roi_marked.png"), marked_image)
    
    # Extraire la ROI
    roi = utils.extract_roi(normalized, roi_type="top_row")
    roi_height, roi_width = roi.shape[:2]
    print(f"   ROI extraite: {roi_width}x{roi_height}")
    cv2.imwrite(str(save_path / f"{filename}_3_roi_extracted.png"), roi)
    
    # 4. Prétraitements
    print("\n4️⃣  Application des 3 versions de prétraitement...")
    
    print("   📊 Version A (Normal)...")
    version_a = preprocessing.preprocess_version_a(roi)
    cv2.imwrite(str(save_path / f"{filename}_4a_preprocess_normal.png"), version_a)
    
    print("   📊 Version B (Sombre)...")
    version_b = preprocessing.preprocess_version_b(roi)
    cv2.imwrite(str(save_path / f"{filename}_4b_preprocess_dark.png"), version_b)
    
    print("   📊 Version C (Clair)...")
    version_c = preprocessing.preprocess_version_c(roi)
    cv2.imwrite(str(save_path / f"{filename}_4c_preprocess_bright.png"), version_c)
    
    # 5. Post-traitement
    print("\n5️⃣  Post-traitement (inversion + enhancement)...")
    
    for version_name, version_img in [('A', version_a), ('B', version_b), ('C', version_c)]:
        inverted = preprocessing.invert_if_needed(version_img)
        enhanced = preprocessing.enhance_for_ocr(inverted)
        cv2.imwrite(str(save_path / f"{filename}_5{version_name}_final.png"), enhanced)
    
    print(f"\n✅ Toutes les visualisations sauvegardées dans: {save_path}/")
    print(f"\n📝 Vérifie les images suivantes:")
    print(f"   - {filename}_2_roi_marked.png   → Vérifier si la zone ROI capture bien la première rangée")
    print(f"   - {filename}_3_roi_extracted.png → Voir ce qui est extrait")
    print(f"   - {filename}_5A_final.png        → Version finale envoyée à l'OCR")
    print()


def main():
    """
    Fonction principale
    """
    if len(sys.argv) < 2:
        print("\n❌ Usage: python debug_visualizer.py <chemin_image.png>")
        print("\nExemple:")
        print("  python debug_visualizer.py data/inputs/ISO-WIN-AZERTY-1.png")
        print("\nOu pour analyser plusieurs images:")
        print("  python debug_visualizer.py data/inputs/*.png")
        return
    
    image_paths = sys.argv[1:]
    
    print("\n" + "="*60)
    print("🔬 VISUALISATEUR DE DÉBOGAGE")
    print("="*60)
    
    for image_path in image_paths:
        visualize_processing(image_path)
        
        if len(image_paths) > 1:
            print("\n" + "-"*60 + "\n")
    
    print("✨ Analyse terminée!")
    print(f"📁 Voir les résultats dans: debug_output/\n")


if __name__ == "__main__":
    main()