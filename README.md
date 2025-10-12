# Détection du layout d’un clavier physique

## **Objectif**

Développer un système capable d’identifier le type de clavier utilisé (AZERTY français-Belgique, AZERTY français-France, QWERTY US, QWERTY UK, etc.) sur base d’une photo du clavier.

## **MVP**

Traitement d’une image d’un clavier physique ; détection automatique de la disposition des touches ; identification du layout (au minimum : QWERTY ou AZERTY) avec affichage du résultat à l’utilisateur.

---

## Plan de projet : Détection du layout de clavier physique

### **1. Phase de préparation des données**

**Objectif :** constituer un jeu d’images variées de claviers.

- **Collecte d’échantillons :**

    - Différents layouts : `AZERTY FR`, `AZERTY BE`, `QWERTY US`, `QWERTY UK`, etc.
    - Conditions variables : angles, luminosité, distance, contraste.
    - Sauvegarder au format `.png` ou `.jpg`, résolution moyenne (1000–2000 px de large).

- **Annotation manuelle :**

    - Créer un tableau `dataset.csv` avec deux colonnes : `filename`, `layout`.
    - Optionnel : marquer manuellement les coordonnées approximatives d’une ligne de touches (pour valider les algorithmes).


---

### **2. Prétraitement d’image**

**Objectif :** obtenir une image claire, binarisée et prête pour l’analyse des touches.

Étapes :

1. **Conversion en niveaux de gris** → `img_gray = rgb2gray(img)`
2. **Réduction de bruit** : filtre moyenneur ou médian.
3. **Amélioration du contraste** : histogramme égalisé.
4. **Seuillage automatique** (Otsu ou manuel) → `img_bin`.
5. **Morphologie** :

    - `ouverture` pour enlever les petits points.
    - `fermeture` pour combler les trous dans les touches.

>À ce stade : on obtient une image “noir et blanc” où chaque touche est une forme blanche isolée.

---

### **3. Détection des touches individuelles**

**Objectif :** repérer les touches et extraire leur géométrie.

- **Contours** : appliquer un algorithme de recherche de composantes connexes ou de contours (implémenté à la main si besoin).
- **Filtrage des régions** :

    - Supprimer les très petites ou très grandes (bruit, bords du clavier).
    - Garder les rectangles approximatifs.
- **Approximation rectangulaire** : calculer le rectangle englobant chaque touche.
- **Visualisation** : afficher les bounding boxes pour vérification.

---

### **4. Reconstruction de la grille**

**Objectif :** organiser les touches en lignes et colonnes.

- Trier les touches selon leur coordonnée `y` pour regrouper par ligne.
- Calculer la moyenne de `y` dans chaque groupe → lignes du clavier.
- Trier ensuite dans chaque ligne par coordonnée `x`.
- Représenter la structure comme une matrice de symboles (touches).

---

### **5. Détection du layout**

**Objectif :** comparer la structure à des patterns connus.

#### Méthode simple :

- Identifier les **3 premières touches** de la ligne supérieure de lettres :

    - Si c’est `A-Z-E` → **AZERTY**
    - Si c’est `Q-W-E` → **QWERTY**

- Pour cela, tu peux :

    - Croper les images des premières touches détectées.
    - Appliquer une **corrélation** ou **template matching** avec des modèles de lettres (`A`, `Q`, `Z`, `W`) pré-enregistrés (créés à la main ou extraits d’autres images).
    - Mesurer la corrélation maximale → identifier la lettre dominante.

>Même si la reconnaissance n’est pas parfaite, une corrélation simple entre images de caractères peut suffire pour différencier AZERTY/QWERTY.

---

### **6. Validation et affichage du résultat**

**Objectif :** montrer à l’utilisateur la détection.

- Superposer le label détecté sur l’image :

```
Layout détecté : AZERTY (FR)
```

- Afficher les touches détectées (rectangle + index de ligne/colonne).
- Évaluer la précision du système sur le petit dataset.

---

### **7. Améliorations possibles**

- Reconnaissance plus fine (France vs Belgique) : comparer les symboles `²`, `é`, etc.
- Compensation d’inclinaison (correction de perspective avec transformation affine).
- Gestion d’éclairage inégal (filtrage adaptatif, seuillage local).
- Interface utilisateur simple avec `matplotlib` ou `tkinter`.
