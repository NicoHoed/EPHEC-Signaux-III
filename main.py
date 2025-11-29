import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
import config
from src.preprocessing import pretraiter_image
from src.analysis import detecter_touches, lire_touches_ocr, determiner_layout

def main():
    image_path = config.IMAGE_PATH_DEFAULT
    print(f"Chargement de {image_path}...")

    # 1. Prétraitement et Chargement
    # img_bin : Masque noir/blanc pour détecter les formes
    # img_gris : Image grise propre pour l'OCR
    img_bin, img_gris = pretraiter_image(image_path)
    
    # Pour l'affichage final (couleur)
    try:
        img_display = io.imread(image_path)
    except Exception:
        print("Erreur de chargement pour l'affichage.")
        return

    # 3. Détection des formes (V1)
    print("Détection des touches...")
    touches = detecter_touches(img_bin)
    print(f"{len(touches)} touches potentielles détectées.")

    # 4. OCR (V2)
    print("Lecture OCR (Tesseract)...")
    ocr_results = lire_touches_ocr(img_gris, touches)
    print(f"{len(ocr_results)} caractères identifiés avec succès.")

    # 5. Classification
    infos, sequence_lue = determiner_layout(ocr_results, touches, img_gris)
    
    print("-" * 30)
    print(f"RÉSULTAT V2 : {infos}")
    print(f"Séquence de lettres vue : {sequence_lue}")
    print("-" * 30)

    # 6. Visualisation
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # CORRECTION ICI : On affiche l'image chargée, pas le chemin !
    ax.imshow(img_display)
    
    # Dessiner les boites
    for r in touches:
        # Boite verte pour tout ce qui est détecté comme "forme de touche"
        rect = mpatches.Rectangle((r.bbox[1], r.bbox[0]), r.bbox[3]-r.bbox[1], r.bbox[2]-r.bbox[0],
                                  fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(rect)

    # Ecrire les lettres reconnues
    for item in ocr_results:
        # Boite rouge pour ce qui est lu par OCR
        rect = mpatches.Rectangle((item['bbox'][1], item['bbox'][0]), 
                                  item['bbox'][3]-item['bbox'][1], item['bbox'][2]-item['bbox'][0],
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # Texte en jaune
        ax.text(item['cx'], item['cy'], item['char'], 
                color='yellow', fontsize=14, fontweight='bold', ha='center', va='center')

    ax.set_title(f"Detection V2: {infos['Layout']} - {infos['Format']} ({infos['OS']})")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()