import numpy as np
from sklearn.cluster import DBSCAN


def reconstruire_grille(boites_touches, seuil_y=50):
    """
    Reconstruit la grille des touches à partir des bounding boxes.
    Organise les touches en lignes et colonnes selon leur position spatiale.

    Paramètres :
    - boites_touches : liste de tuples (minr, minc, maxr, maxc)
    - seuil_y : seuil de proximité verticale pour regrouper les touches en lignes

    Retourne : liste de listes contenant les touches organisées par ligne/colonne
    """

    if not boites_touches:
        return []

    # Étape 1 : Calculer le centre Y de chaque touche
    centres_y = [(minr + maxr) / 2 for minr, _, maxr, _ in boites_touches]

    # Étape 2 : Regrouper les touches en lignes selon leur position Y (clustering DBSCAN)
    centres_y_array = np.array(centres_y).reshape(-1, 1)
    clustering = DBSCAN(eps=seuil_y, min_samples=1).fit(centres_y_array)
    labels_lignes = clustering.labels_

    # Étape 3 : Organiser les touches par ligne
    lignes = {}
    for idx, (bbox, label_ligne) in enumerate(zip(boites_touches, labels_lignes)):
        if label_ligne not in lignes:
            lignes[label_ligne] = []
        lignes[label_ligne].append(bbox)

    # Étape 4 : Trier les lignes par position Y et les touches dans chaque ligne par position X
    lignes_ordonnees = []
    for label_ligne in sorted(lignes.keys()):
        touches_ligne = lignes[label_ligne]

        # Trier les touches de la ligne par coordonnée X (colonne)
        touches_ligne_triees = sorted(touches_ligne, key=lambda bbox: bbox[1])  # bbox[1] = minc (colonne)

        # Créer une structure pour chaque touche avec ses informations
        lignes_ordonnees.append([
            {
                "bbox": bbox,
                "char": None
            }
            for bbox in touches_ligne_triees
        ])

    return lignes_ordonnees


def afficher_grille(grille):
    """
    Affiche la structure de la grille (utile pour débogage).

    Paramètres :
    - grille : résultat de reconstruire_grille()
    """
    for num_ligne, ligne in enumerate(grille):
        print(f"Ligne {num_ligne} : {len(ligne)} touches")
        for num_col, touche in enumerate(ligne):
            minr, minc, maxr, maxc = touche["bbox"]
            largeur = maxc - minc
            hauteur = maxr - minr
            print(f"  [{num_col}] bbox=({minr}, {minc}, {maxr}, {maxc}), dim=({largeur}x{hauteur})")
