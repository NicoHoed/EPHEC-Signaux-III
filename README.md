# ⌨️ Reconnaissance automatique du layout de clavier

Ce projet permet d’analyser une photo de clavier physique et de détecter automatiquement :

- Le format (ISO / ANSI)

- Le système (Mac / Windows)

- Le layout (AZERTY, QWERTY, etc.)

L’analyse inclut également une visualisation graphique des touches détectées.

## Installation (Linux / macOS)

Depuis la racine du projet, exécutez :


```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optionnel, nécessaire pour l'affichage graphique :
sudo apt install python3-tk
```

## Configuration de l’image à analyser

Ouvrez le fichier :

config.py

Modifiez la ligne suivante pour indiquer le chemin de l’image à analyser :


```bash
IMAGE_PATH_DEFAULT = "data/inputs/nom_de_votre_image.jpg"
```

Vous trouverez plusieurs images d’exemple dans :

data/inputs/

## Exécution du programme
Toujours dans l’environnement virtuel :

```bash
python main.py
```
