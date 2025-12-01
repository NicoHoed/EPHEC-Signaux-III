# KeyDetect - Détecteur de Layout Clavier

**KeyDetect** est un outil de vision par ordinateur capable d'identifier le layout d'un clavier (AZERTY, QWERTY, QWERTZ) à partir d'une simple photo.

Contrairement aux approches classiques basées sur des zones fixes, ce projet utilise le **Clustering Géométrique (K-Means)** pour comprendre la structure du clavier, ce qui le rend robuste aux photos prises de biais, aux rotations légères et aux différents facteurs de forme.

## Fonctionnalités

* **Robustesse visuelle :** Utilise 3 méthodes de prétraitement simultanées (Adaptive Threshold, LAB Channel, Inversion) pour gérer les reflets et les claviers noirs/blancs.
* **Intelligence Géométrique :** Utilise l'algorithme K-Means pour regrouper dynamiquement les touches par rangées (Haut/Milieu/Bas), indépendamment de l'angle de la photo.
* **Moteur OCR puissant :** Basé sur `EasyOCR` pour une lecture précise des caractères.
* **Mode Benchmark (GUI) :** Interface graphique pour valider automatiquement un lot d'images et calculer le taux de précision.

## Structure du Projet

```
KeyDetect/
├── data/
│   └── inputs/          # Placez vos images de test ici (.png)
├── src/
│   ├── __init__.py
│   ├── engine.py        # Cerveau : Pipeline OCR, Clustering & Scoring
│   └── preprocessing.py # Traitement d'image (OpenCV)
├── main.py              # Script console (analyse fichier par fichier)
├── gui_benchmark.py     # Interface graphique (analyse de masse & stats)
└── requirements.txt     # Dépendances
```

## Installation

Ce projet nécessite **Python 3.11** (pour la stabilité de l'interface graphique).

### 1. Pré-requis Système

  * **Windows :** Avoir Python installé.
  * **Linux :** `sudo apt-get install python3-tk`
  * **macOS (Important) :** Python via Homebrew pose souvent problème avec l'interface graphique (`tkinter`).
      * *Recommandé :* Installez **Python 3.11** via l'installateur officiel sur [python.org](https://www.python.org/downloads/macos/).

### 2. Installation de l'environnement

Ouvrez votre terminal à la racine du projet :

#### Sur macOS / Linux

```bash
# Créer un nouvel environnement virtuel (en pointant vers Python 3.11)
# Si installé via l'installeur officiel :
/usr/local/bin/python3.11 -m venv venv
# Sinon :
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

#### Sur Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### Préparation des données

Placez vos photos de claviers dans le dossier `data/inputs`.
*Pour le benchmark automatique, nommez vos fichiers ainsi :* `TYPE-OS-LAYOUT-X.png` (ex: `ISO-WIN-AZERTY-1.png`). Le script cherche "AZERTY", "QWERTY" ou "QWERTZ" dans le nom.

### Mode Console (Analyse simple)

Pour voir le détail du processus (rangées détectées, scores détaillés) :

```bash
python main.py
```

### Mode Interface Graphique (Benchmark)

Pour lancer une validation de masse et voir les statistiques de réussite :

```bash
python gui_benchmark.py
```

1.  Cliquez sur **"Charger l'OCR"** (patientez quelques secondes).
2.  Cliquez sur **"LANCER L'ANALYSE"**.

## Comment ça marche ?

1.  **Shotgun Preprocessing :** L'image est dupliquée et traitée avec 3 filtres différents pour maximiser la lisibilité.
2.  **OCR Pipeline :** EasyOCR scanne les 3 versions. Seules les lettres détectées avec une confiance suffisante sont conservées.
3.  **Clustering K-Means :** L'algorithme analyse la position Y (verticale) de toutes les lettres trouvées pour identifier mathématiquement 3 clusters (les 3 rangées du clavier).
4.  **Scoring Pondéré :** Le moteur vérifie la présence de lettres clés (A, Z, Q, W, M...) dans les clusters identifiés. Des points sont attribués ou retirés  pour déterminer le layout final avec un indice de confiance.

## Auteur

Nicolas HOEDENAEKEN
Théo MERTENS
Baris OZCELIK
Khassan AKTAMIROV

Projet réalisé dans le cadre du cours de Signaux III.
