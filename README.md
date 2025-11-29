# 🎹 Détecteur de Layout Clavier

Programme Python de reconnaissance automatique de layout clavier (QWERTY, QWERTZ, AZERTY) à partir de photos PNG.

## 📋 Prérequis

### 1. Python et Environnement Virtuel
- Python 3.8 ou supérieur
- Un environnement virtuel (venv) déjà créé

### 2. Tesseract OCR
**⚠️ IMPORTANT** : Installer Tesseract OCR sur votre système :

#### Windows
1. Télécharger l'installeur : https://github.com/UB-Mannheim/tesseract/wiki
2. Installer (par défaut dans `C:\Program Files\Tesseract-OCR`)
3. Si installé ailleurs, modifier dans `src/ocr_engine.py` :
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Chemin\Vers\tesseract.exe'
```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
```

Vérifier l'installation :
```bash
tesseract --version
```

## 🚀 Installation

### 1. Activer l'environnement virtuel

**Linux/macOS :**
```bash
source venv/bin/activate
```

**Windows :**
```bash
venv\Scripts\activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
keyboard_layout_detector/
│
├── venv/                      # Environnement virtuel
├── data/
│   ├── inputs/               # 📥 Mettre vos images PNG ici
│   └── outputs/              # 📤 Résultats générés
│       ├── processed/        # Images debug (si --save-debug)
│       └── report.json       # Rapport détaillé
│
├── src/                      # Code source
│   ├── __init__.py
│   ├── utils.py              # Fonctions utilitaires
│   ├── preprocessing.py      # Prétraitement d'images
│   ├── ocr_engine.py         # Moteur OCR
│   └── classifier.py         # Classification de layout
│
├── main.py                   # Point d'entrée
├── requirements.txt          # Dépendances
└── README.md                 # Ce fichier
```

## 🎮 Utilisation

### Mode Basique
```bash
python main.py
```
Traite toutes les images PNG dans `data/inputs/` et génère les résultats dans `data/outputs/`.

### Avec Options

```bash
# Afficher les détails du traitement
python main.py --verbose

# Sauvegarder les images prétraitées (pour débogage)
python main.py --save-debug

# Dossiers personnalisés
python main.py --input mon_dossier/images --output mon_dossier/resultats

# Définir un seuil de confiance
python main.py --confidence-threshold 70

# Combinaison d'options
python main.py --verbose --save-debug --confidence-threshold 65
```

### Options Disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--input` | Dossier contenant les images PNG | `data/inputs` |
| `--output` | Dossier de sortie | `data/outputs` |
| `--save-debug` | Sauvegarder les images prétraitées | Désactivé |
| `--verbose` | Afficher les détails du traitement | Désactivé |
| `--confidence-threshold` | Seuil de confiance minimal (%) | 60 |

## 📊 Comprendre les Résultats

### Sortie Console

**Mode Normal :**
```
🖼️  keyboard_01.png... ✅ QWERTY (95%)
🖼️  keyboard_02.png... ✅ AZERTY (88%)
🖼️  keyboard_03.png... ❓ UNKNOWN (45%)
```

**Mode Verbose :**
```
============================================================
🖼️  Traitement: keyboard_01.png
============================================================
📐 Normalisation de la résolution...
🔍 Extraction de la zone d'intérêt...
🎨 Prétraitement multi-passes (3 versions)...
🔤 Reconnaissance OCR...
  🔍 Résultats OCR bruts (9): ['QWERTY', 'QWERTY', 'QWERT', ...]
  🗳️  Meilleur résultat: 'QWERTY' (votes: 7/9, confiance: 77.8%)
🎯 Classification du layout...
  📊 Scores de correspondance:
     QWERTY: 100
     QWERTZ: 60
     AZERTY: 20
✅ Résultat: QWERTY (confiance: 91%)
⏱️  Temps: 2.34s
```

### Rapport JSON (`data/outputs/report.json`)

```json
{
  "timestamp": "2024-01-15T14:30:00",
  "summary": {
    "total_images": 50,
    "successful": 48,
    "failed": 2,
    "accuracy": "96.00%"
  },
  "results": [
    {
      "filename": "keyboard_01.png",
      "detected_layout": "QWERTY",
      "confidence": 95,
      "detected_chars": "QWERTY",
      "processing_time": "2.34s",
      "ocr_confidence": 88,
      "pattern_scores": {
        "QWERTY": 100,
        "QWERTZ": 60,
        "AZERTY": 20
      }
    }
  ]
}
```

### Interprétation du Score de Confiance

| Score | Interprétation |
|-------|----------------|
| 90-100% | ✅ Excellente détection |
| 70-89% | ✅ Bonne détection |
| 60-69% | ⚠️ Détection acceptable |
| < 60% | ❌ Résultat non fiable (UNKNOWN) |

## 🔧 Fonctionnement Technique

### Pipeline de Traitement

```
Photo PNG
    ↓
[1] Normalisation (largeur 1200px)
    ↓
[2] Extraction ROI (première rangée)
    ↓
[3] Prétraitement Multi-Passes (3 versions)
    │   ├─ Version A: Éclairage normal
    │   ├─ Version B: Éclairage sombre
    │   └─ Version C: Éclairage clair
    ↓
[4] OCR Multi-Config (3 configs × 3 versions = 9 résultats)
    ↓
[5] Vote Majoritaire
    ↓
[6] Classification par Pattern Matching
    ↓
Résultat + Score de Confiance
```

### Stratégie de Détection

Le programme se concentre sur les **6 premières touches** de la première rangée :

- **QWERTY** : Q-W-E-R-T-**Y**
- **QWERTZ** : Q-W-E-R-T-**Z**
- **AZERTY** : **A**-**Z**-E-R-T-Y

Seules 2-3 touches suffisent pour différencier les layouts !

## 🐛 Dépannage

### Erreur "tesseract is not installed"
**Solution** : Installer Tesseract OCR (voir section Prérequis)

### Erreur "No module named 'cv2'"
**Solution** :
```bash
pip install opencv-python
```

### Mauvais taux de reconnaissance
**Solutions** :
1. Vérifier la qualité des images (résolution suffisante)
2. Utiliser `--save-debug` pour voir les images prétraitées
3. Ajuster les paramètres de prétraitement dans `src/preprocessing.py`

### "UNKNOWN" pour toutes les images
**Causes possibles** :
- Images trop floues ou mal cadrées
- Tesseract mal configuré
- Éclairage extrême (trop sombre/clair)

**Solution** : Utiliser `--verbose --save-debug` pour diagnostiquer

## 📈 Performances Attendues

| Condition | Taux de Réussite |
|-----------|------------------|
| Photos de qualité, bon éclairage | 95-98% |
| Éclairage variable | 85-92% |
| Images difficiles | 70-85% |
| **Moyenne générale** | **~90%** |

## 🎯 Améliorations Futures

- [ ] Support des claviers Dvorak, Colemak
- [ ] Détection de l'angle de prise de vue
- [ ] Interface graphique (GUI)
- [ ] API REST
- [ ] Modèle de deep learning

## 📝 Notes

- Le programme est optimisé pour les **photos prises de face**
- Les images doivent être au format **PNG**
- Résolutions variables supportées (normalisation automatique)
- Traitement par batch pour efficacité maximale

## 🤝 Contribution

Suggestions et améliorations bienvenues !

## 📄 Licence

Projet éducatif - Libre d'utilisation

---

**Bon traitement ! 🚀**