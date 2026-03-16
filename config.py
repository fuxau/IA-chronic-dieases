"""
config.py — Configuration centralisée pour le module de reconnaissance alimentaire.

Tous les hyper-paramètres, chemins et constantes sont définis ici
pour faciliter l'expérimentation et la maintenance.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Chemins du projet
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_DATASET_DIR = PROJECT_ROOT / "custom_dataset"
MODELS_DIR = PROJECT_ROOT / "models"

# Créer les répertoires s'ils n'existent pas
DATA_DIR.mkdir(exist_ok=True)
CUSTOM_DATASET_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Modèle
# ──────────────────────────────────────────────
MODEL_NAME = "efficientnet_b0"          # Architecture backbone (timm)
IMG_SIZE = 224                           # Taille d'entrée du réseau
NUM_CHANNELS = 3                         # RGB
PRETRAINED = True                        # Utiliser les poids ImageNet

# ──────────────────────────────────────────────
# Entraînement
# ──────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 4
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SCHEDULER = "cosine"                     # cosine | step | plateau
LABEL_SMOOTHING = 0.1

# Fine-tuning progressif
FREEZE_BACKBONE = True                   # Phase 1 : geler le backbone
UNFREEZE_AFTER_EPOCH = 5                 # Phase 2 : dégeler après N époques
UNFREEZE_LAYERS = 3                      # Nombre de blocs à dégeler

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
FOOD101_CLASSES = 101                    # Classes Food-101
CUSTOM_CLASSES = 0                       # Sera mis à jour dynamiquement
TRAIN_SPLIT = 0.8                        # 80% train / 20% val

# Normalisation ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Export & Inférence
# ──────────────────────────────────────────────
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
ONNX_MODEL_PATH = MODELS_DIR / "food_model.onnx"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"

CONFIDENCE_THRESHOLD = 0.5              # Seuil minimum de confiance
ONNX_OPSET_VERSION = 17                 # Version opset ONNX

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE_MB = 10                    # Taille max upload (Mo)
DEFAULT_PORTION_G = 100                  # Portion par défaut (grammes)
