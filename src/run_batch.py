import os
import csv
import glob
from skimage import io
import pandas as pd
import config

from src.preprocessing import pretraiter_image
from src.analysis import detecter_touches, identifier_zones_cles, classifier_clavier

# Ce script permet de faire des tests sur un lot de photos.

# --- CONFIGURATION ---
INPUT_DIR = 'data/inputs'
OUTPUT_FILE = 'data/outputs/rapport_analyse.csv'
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tif']

def analyser_image(image_path):
    """Exécute le pipeline complet sur une image et retourne un dictionnaire de résultats."""
    nom_fichier = os.path.basename(image_path)
    resultat = {
        "Fichier": nom_fichier,
        "Statut": "OK",
        "Format": "N/A",
        "OS": "N/A",
        "Langue": "N/A",
        "Shift_Ratio": 0.0,
        "OS_Euler": 0,
        "TL_Extent": 0.0,
        "NB_Touches": 0,
        "Enter_Ratio_H_L": 0.0 # NOUVEAU
    }

    try:
        # 1. Chargement
        img = io.imread(image_path)
        
        # 2. Prétraitement
        img_bin, img_gris = pretraiter_image(img)
        
        # 3. Détection

        touches, _, _, _ = detecter_touches(
            img_bin,
            aire_min=config.AIRE_MIN,
            aire_max=config.AIRE_MAX,
            ratio_max=config.RATIO_MAX,
            seuil_y=config.SEUIL_Y_PROXIMITE
        )
        resultat["NB_Touches"] = len(touches)
        
        if len(touches) < 10:
            resultat["Statut"] = "ÉCHEC (Pas assez de touches)"
            return resultat

        # 4. Zoning
        rois = identifier_zones_cles(touches)
        
        # Vérification minimale de la présence des zones clés
        if rois is None or not rois.get("SPACE") or not rois.get("SHIFT") or not rois.get("OS_KEY"):
            resultat["Statut"] = "ÉCHEC (ROIs clés manquantes)"
            return resultat



        # 5. Classification
        verdict, debug = classifier_clavier(rois, img_gris)
        
        # Remplissage des résultats
        resultat["Format"] = verdict.get("ISO_ANSI", "?")
        resultat["OS"] = verdict.get("MAC_WIN", "?")
        resultat["Langue"] = verdict.get("LAYOUT", "?")
        
        # Données techniques (si disponibles dans debug)
        resultat["Shift_Ratio"] = round(debug.get("Shift_Ratio", 0), 2)
        resultat["OS_Euler"] = debug.get("OS_Euler", 0)
        resultat["TL_Extent"] = round(debug.get("TL_Extent", 0), 2)
        resultat["Enter_Ratio_H_L"] = round(debug.get("Enter_Ratio_H_L", 0), 2) # NOUVEAU

    except Exception as e:
        resultat["Statut"] = f"ERREUR ({str(e)})"
    
    return resultat

def main():
    # 1. Récupération des fichiers
    fichiers = []
    for ext in EXTENSIONS:
        fichiers.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not fichiers:
        print(f"❌ Aucun fichier trouvé dans {INPUT_DIR}")
        return

    print(f"Lancement du traitement par lots sur {len(fichiers)} images...\n")
    
    resultats_globaux = []

    # 2. Boucle de traitement
    for i, fichier in enumerate(fichiers):
        # Enlever \r pour un affichage propre
        print(f"[{i+1}/{len(fichiers)}] Traitement de {os.path.basename(fichier)}...          ") 
        donnees = analyser_image(fichier)
        resultats_globaux.append(donnees)
        
        # Petit feedback visuel immédiat
        status_icon = "✅" if "OK" in donnees["Statut"] else "⚠️"
        print(f"{status_icon} {donnees['Fichier']:<25} -> {donnees['Format']} | {donnees['OS']} | {donnees['Langue']} | Statut: {donnees['Statut']}")

    # 3. Sauvegarde CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # NOUVEAU: Ajout de Enter_Ratio_H_L dans les colonnes CSV
    colonnes = ["Fichier", "Statut", "Format", "OS", "Langue", "NB_Touches", "Shift_Ratio", "Enter_Ratio_H_L", "OS_Euler", "TL_Extent"]
    
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=colonnes)
            writer.writeheader()
            for data in resultats_globaux:
                # On filtre pour n'écrire que les colonnes définies
                clean_data = {k: data.get(k, "N/A") for k in colonnes}
                writer.writerow(clean_data)
        
        print(f"\nRapport complet généré : {OUTPUT_FILE}")
        
    except IOError as e:
        print(f"\n❌ Erreur d'écriture du fichier CSV : {e}")

if __name__ == "__main__":
    main()