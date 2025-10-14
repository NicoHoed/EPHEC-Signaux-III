# Détection du layout d’un clavier physique

## **Objectif**

Développer un système capable d’identifier le type de clavier utilisé (AZERTY français-Belgique, AZERTY français-France, QWERTY US, QWERTY UK, etc.) sur base d’une photo du clavier.

## **MVP**

Traitement d’une image d’un clavier physique ; détection automatique de la disposition des touches ; identification du layout (au minimum : QWERTY ou AZERTY) avec affichage du résultat à l’utilisateur.

## Plan de projet : Détection du layout de clavier physique

### **1. Phase de préparation des données**

**Objectif :** constituer un jeu d’images variées de claviers.

* **Collecte d’échantillons :**

  * Différents layouts : `AZERTY FR`, `AZERTY BE`, `QWERTY US`, `QWERTY UK`, etc.
  * Conditions variables : angles, luminosité, distance, contraste.
  * Sauvegarder au format `.png` ou `.jpg`, résolution moyenne (1000–2000 px de large).

* **Annotation manuelle :**

  * Créer un tableau `dataset.csv` avec deux colonnes : `filename`, `layout`.
  * Optionnel : marquer manuellement les coordonnées approximatives d’une ligne de touches (pour valider les algorithmes).

### **2. Prétraitement d’image**

**Objectif :** obtenir une image claire, binarisée et prête pour l’analyse des touches.

Étapes :

1. **Conversion en niveaux de gris** → `img_gray = rgb2gray(img)`
2. **Réduction de bruit** : filtre moyenneur ou médian.
3. **Amélioration du contraste** : histogramme égalisé.
4. **Seuillage automatique** (Otsu ou manuel) → `img_bin`.
5. **Morphologie** :

   * `ouverture` pour enlever les petits points.
   * `fermeture` pour combler les trous dans les touches.

> À ce stade : on obtient une image “noir et blanc” où chaque touche est un rectangle noir isolé et les caractères à l’intérieur apparaissent en blanc.

### **3. Détection des touches entières**

**Objectif :** repérer uniquement les rectangles noirs correspondant aux touches du clavier.

* **Inversion de l’image** : les touches noires deviennent les objets à détecter.
* **Labelisation des composants connexes** pour extraire les régions.
* **Filtrage par aire** :

  * éliminer les petites régions (bruits, lettres)
  * éliminer les très grandes régions (trackpad ou artefacts).
* **Filtrage par ratio largeur/hauteur** : détecter les touches régulières et accepter séparément les touches larges atypiques (spacebar, Enter, Maj droite).
* **Filtrage par position verticale** : éliminer les composants en dehors de la zone du clavier.
* **Résultat** : une liste de rectangles englobants représentant les touches entières.

> À ce stade : tous les objets détectés correspondent à de vraies touches du clavier. Les lettres à l’intérieur ne sont pas encore analysées.

### **4. Visualisation**

**Objectif :** vérifier la détection.

* Afficher l’image binaire avec des rectangles rouges superposés sur chaque touche détectée.
* Compter le nombre de touches détectées pour chaque ligne.

### **5. Reconstruction de la grille** (étape suivante)

**Objectif :** organiser les touches détectées en lignes et colonnes.

* Trier les touches par coordonnée verticale (`y`) pour regrouper par ligne.
* Calculer la moyenne de `y` dans chaque groupe pour définir la ligne du clavier.
* Trier ensuite les touches de chaque ligne par coordonnée horizontale (`x`).
* Représenter la structure sous forme de matrice (initialement remplie de `None` ou symboles à remplir après reconnaissance des caractères).

> Cette étape permet de déterminer **le nombre de touches par ligne** et la disposition générale du clavier.

### **6. Détection du layout (après reconnaissance des caractères)**

**Objectif :** comparer la structure à des patterns connus.

* Identifier les premières touches de la ligne de lettres pour différencier AZERTY/QWERTY.
* Méthodes possibles :

  * Template matching sur les caractères.
  * Corrélation simple avec des modèles prédéfinis (`A`, `Q`, `Z`, `W`).

### **7. Validation et affichage du résultat**

* Superposer le label détecté sur l’image :

```
Layout détecté : AZERTY (FR)
```

* Afficher les touches détectées (rectangles + index ligne/colonne).
* Évaluer la précision du système sur le dataset.

```mermaid
flowchart TD
    A["Image du clavier - photo RGB/JPG"] --> B["Prétraitement d'image
- Gris
- Filtre médian
- Equalisation
- Seuillage
- Morphologie"]

    B --> C["Détection des touches
- Inversion image
- Composants connectés
- Filtrage aire / ratio / position"]

    C --> D["Visualisation / Vérification
- Rectangles rouges sur touches
- Comptage touches par ligne"]

    D --> E["Reconstruction de la grille
- Regroupement par ligne Y
- Tri par colonne X
- Matrice image_based"]

    E --> F["Détection du layout
- Comparaison avec modèles AZERTY/QWERTY"]

```
