# 🍽️ Module IA — Reconnaissance Alimentaire & Estimation Calorique

Module de vision par IA qui analyse la photo d'un plat pour identifier l'aliment et estimer les calories. Conçu pour s'intégrer dans l'application de gestion des maladies chroniques.

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Modèle | EfficientNet-B0 (timm) |
| Dataset | Food-101 + custom (15-20 aliments) |
| Entraînement | PyTorch + AdamW + Cosine LR |
| Export | ONNX (opset 17) |
| Inférence | ONNX Runtime |
| API | FastAPI + Uvicorn |
| Conteneur | Docker (python:3.11-slim) |

## Architecture

```
                    ┌─────────────┐
                    │  Image JPG  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │  /predict   │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   Prétraitement         │
              │   Resize → Crop → Norm  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   ONNX Runtime          │
              │   EfficientNet-B0       │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Softmax → Label       │
              │   + Score de confiance  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Table Nutritionnelle  │
              │   kcal × portion        │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  JSON       │
                    │  Réponse    │
                    └─────────────┘
```

## Installation

```bash
# Cloner le repo
git clone <repo-url>
cd IA-chronic-dieases

# Environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installer les dépendances (entraînement complet)
pip install -r requirements.txt

# OU uniquement l'inférence
pip install -r requirements-inference.txt
```

## Utilisation

### 1. Préparer le dataset custom (optionnel)

Organisez vos images dans `custom_dataset/` :

```
custom_dataset/
├── couscous/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── tajine/
│   └── ...
└── harira/
    └── ...
```

> **Recommandation :** 50-100 images par classe, en résolution ≥ 224×224.

### 2. Entraîner le modèle

```bash
python train.py
```

L'entraînement se fait en deux phases :
1. **Phase 1 (époques 1→5)** : Backbone gelé, seul le classifier est entraîné
2. **Phase 2 (époques 6→15)** : Derniers blocs dégelés pour affinage

Le meilleur modèle est sauvegardé dans `models/best_model.pth`.

### 3. Exporter en ONNX

```bash
python export_onnx.py
```

Génère :
- `models/food_model.onnx` — Modèle optimisé
- `models/class_labels.json` — Mapping index → label

### 4. Lancer l'API d'inférence

```bash
# Mode développement
python inference_api.py

# OU avec uvicorn
uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Tester l'API

```bash
# Health check
curl http://localhost:8000/health

# Prédiction (portion par défaut)
curl -X POST "http://localhost:8000/predict" \
  -F "file=@mon_plat.jpg"

# Prédiction avec portion personnalisée (150g)
curl -X POST "http://localhost:8000/predict?portion_g=150" \
  -F "file=@mon_plat.jpg"

# Info nutritionnelle directe
curl "http://localhost:8000/nutrition/pizza?portion_g=200"
```

**Réponse exemple :**

```json
{
  "label": "pizza",
  "confidence": 0.9412,
  "nutrition": {
    "calories": 532.0,
    "protein_g": 22.8,
    "carbs_g": 66.0,
    "fat_g": 20.8,
    "fiber_g": 4.0
  },
  "portion_g": 200.0,
  "default_portion_g": 200.0,
  "inference_time_ms": 142.35
}
```

**Documentation interactive :** [http://localhost:8000/docs](http://localhost:8000/docs)

## Déploiement Docker

```bash
# Build (après entraînement + export)
docker build -t food-recognition .

# Run
docker run -p 8000:8000 food-recognition

# Vérifier
curl http://localhost:8000/health
```

## Structure du projet

```
IA-chronic-dieases/
├── config.py                 # Hyper-paramètres et constantes
├── data_preparation.py       # Chargement Food-101 + dataset custom
├── train.py                  # Fine-tuning EfficientNet-B0
├── export_onnx.py            # Export .pth → .onnx
├── nutrition_table.py        # Table nutritionnelle (kcal/100g)
├── inference_api.py          # API FastAPI (inférence + calories)
├── requirements.txt          # Dépendances complètes
├── requirements-inference.txt # Dépendances inférence uniquement
├── Dockerfile                # Conteneur d'inférence
├── .dockerignore             # Exclusions Docker
├── README.md                 # Ce fichier
├── custom_dataset/           # Images personnalisées
└── models/                   # Poids exportés (.onnx, .json)
```

## Performance

- **Inférence** : < 200ms sur CPU (ONNX Runtime optimisé)
- **Taille du modèle ONNX** : ~20 MB (EfficientNet-B0)
- **Objectif** : < 3 secondes bout-en-bout incluant upload réseau

## Licence

Ce projet fait partie de l'application de gestion des maladies chroniques.
