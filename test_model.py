#!/usr/bin/env python3
"""
test_model.py — Script de test du modèle de reconnaissance alimentaire.

Usage :
    # Tester avec une image locale :
    python test_model.py chemin/vers/image.jpg

    # Tester avec une URL :
    python test_model.py https://example.com/photo_pizza.jpg

    # Tester avec les images du dataset Food-101 :
    python test_model.py --food101

    # Tester le modèle ONNX au lieu du modèle PyTorch :
    python test_model.py image.jpg --onnx
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

import config
from nutrition_table import calculate_calories, get_nutrition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Chargement du modèle
# ──────────────────────────────────────────────
def load_pytorch_model() -> tuple:
    """Charge le modèle PyTorch depuis le checkpoint."""
    checkpoint_path = config.BEST_MODEL_PATH

    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint introuvable : {checkpoint_path}")
        logger.error("   Lance d'abord l'entraînement : python train.py")
        sys.exit(1)

    logger.info(f"Chargement du modèle PyTorch : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = checkpoint["num_classes"]
    model_name = checkpoint["config"]["model_name"]

    # Recréer l'architecture
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Labels
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_label = {v: k for k, v in class_to_idx.items()}

    logger.info(f"  ✅ Modèle chargé ({num_classes} classes, epoch {checkpoint.get('epoch', '?')})")
    logger.info(f"  📊 Meilleure accuracy : {checkpoint.get('best_acc', '?'):.2f}%")

    return model, idx_to_label


def load_onnx_model():
    """Charge le modèle ONNX Runtime."""
    import onnxruntime as ort

    onnx_path = config.ONNX_MODEL_PATH
    if not onnx_path.exists():
        logger.error(f"❌ Modèle ONNX introuvable : {onnx_path}")
        logger.error("   Lance d'abord l'export : python export_onnx.py")
        sys.exit(1)

    logger.info(f"Chargement du modèle ONNX : {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))

    # Labels
    labels_path = config.CLASS_LABELS_PATH
    if labels_path.exists():
        with open(labels_path, "r") as f:
            labels_data = json.load(f)
        idx_to_label = {int(k): v for k, v in labels_data["idx_to_label"].items()}
    else:
        idx_to_label = {}

    logger.info(f"  ✅ Modèle ONNX chargé ({len(idx_to_label)} classes)")
    return session, idx_to_label


# ──────────────────────────────────────────────
# Prétraitement
# ──────────────────────────────────────────────
def get_test_transform() -> transforms.Compose:
    """Transformations de test (identiques à la validation)."""
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE + 32),   # 256
        transforms.CenterCrop(config.IMG_SIZE),     # 224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
        ),
    ])


def load_image(source: str) -> Image.Image:
    """Charge une image depuis un chemin local ou une URL."""
    if source.startswith("http"):
        import requests as req
        import tempfile
        logger.info(f"📥 Téléchargement de l'image : {source}")
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        try:
            resp = req.get(source, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"❌ Impossible de télécharger l'image : {e}")
            sys.exit(1)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        path = Path(source)
        if not path.exists():
            logger.error(f"❌ Image introuvable : {path}")
            sys.exit(1)
        return Image.open(path).convert("RGB")


# ──────────────────────────────────────────────
# Prédiction
# ──────────────────────────────────────────────
def predict_pytorch(model, image: Image.Image, idx_to_label: dict, top_k: int = 5) -> list:
    """Prédit avec le modèle PyTorch."""
    transform = get_test_transform()
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    elapsed = (time.perf_counter() - start) * 1000

    # Top-K
    top_probs, top_indices = probabilities.topk(top_k)
    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = idx_to_label.get(idx.item(), f"class_{idx.item()}")
        results.append({"label": label, "confidence": prob.item()})

    return results, elapsed


def predict_onnx(session, image: Image.Image, idx_to_label: dict, top_k: int = 5) -> list:
    """Prédit avec ONNX Runtime."""
    transform = get_test_transform()
    input_tensor = transform(image).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    elapsed = (time.perf_counter() - start) * 1000

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        label = idx_to_label.get(int(idx), f"class_{idx}")
        results.append({"label": label, "confidence": float(probabilities[idx])})

    return results, elapsed


# ──────────────────────────────────────────────
# Affichage des résultats
# ──────────────────────────────────────────────
def display_results(results: list, elapsed_ms: float, source: str):
    """Affiche les résultats de prédiction."""
    print()
    print("═" * 60)
    print(f"  🍽️  RÉSULTATS — {os.path.basename(source)}")
    print("═" * 60)
    print(f"  ⏱️  Temps d'inférence : {elapsed_ms:.1f} ms")
    print()

    for i, r in enumerate(results):
        label = r["label"]
        conf = r["confidence"]
        bar = "█" * int(conf * 30)

        # Indicateur visuel de confiance
        if conf > 0.5:
            emoji = "🟢"
        elif conf > 0.2:
            emoji = "🟡"
        else:
            emoji = "🔴"

        print(f"  {i+1}. {emoji} {label:<25} {conf:>6.2%}  {bar}")

    # Détails nutritionnels pour le top-1
    top_label = results[0]["label"]
    top_conf = results[0]["confidence"]
    nutrition = get_nutrition(top_label)
    cal = calculate_calories(top_label, nutrition.default_portion_g)

    print()
    print("─" * 60)
    print(f"  📊 Nutrition — {top_label} (portion {nutrition.default_portion_g}g)")
    print("─" * 60)
    print(f"  🔥 Calories  : {cal['calories']:.0f} kcal")
    print(f"  🥩 Protéines : {cal['protein_g']:.1f} g")
    print(f"  🍞 Glucides  : {cal['carbs_g']:.1f} g")
    print(f"  🧈 Lipides   : {cal['fat_g']:.1f} g")
    print(f"  🌾 Fibres    : {cal['fiber_g']:.1f} g")
    print("═" * 60)
    print()


# ──────────────────────────────────────────────
# Test avec les images Food-101
# ──────────────────────────────────────────────
def test_food101_samples(model_or_session, idx_to_label: dict, use_onnx: bool = False):
    """Teste le modèle sur des échantillons aléatoires de Food-101."""
    food101_dir = Path(config.DATA_DIR) / "food-101" / "images"
    if not food101_dir.exists():
        logger.error(f"❌ Dataset Food-101 introuvable : {food101_dir}")
        logger.error("   Lance d'abord : python pipeline.py (ou python train.py) pour télécharger")
        sys.exit(1)

    # Prendre 1 image aléatoire de 10 classes différentes
    import random
    classes = sorted([d.name for d in food101_dir.iterdir() if d.is_dir()])
    sample_classes = random.sample(classes, min(10, len(classes)))

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  🧪 TEST AUTOMATIQUE — 10 classes aléatoires Food-101  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    correct = 0
    total = 0

    for cls in sample_classes:
        cls_dir = food101_dir / cls
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not images:
            continue

        img_path = random.choice(images)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        if use_onnx:
            results, elapsed = predict_onnx(model_or_session, image, idx_to_label)
        else:
            results, elapsed = predict_pytorch(model_or_session, image, idx_to_label)

        predicted = results[0]["label"]
        conf = results[0]["confidence"]
        match = "✅" if predicted == cls else "❌"
        if predicted == cls:
            correct += 1
        total += 1

        print(f"  {match} Réel: {cls:<25} → Prédit: {predicted:<25} ({conf:.1%}, {elapsed:.0f}ms)")

    print()
    print(f"  Accuracy sur l'échantillon : {correct}/{total} ({100*correct/max(total,1):.0f}%)")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🍽️  Teste le modèle de reconnaissance alimentaire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python test_model.py photo_pizza.jpg          # Tester une image locale
  python test_model.py https://url/image.jpg    # Tester une URL
  python test_model.py --food101                # Tester sur Food-101
  python test_model.py photo.jpg --onnx         # Utiliser le modèle ONNX
        """,
    )
    parser.add_argument("image", nargs="?", help="Chemin ou URL de l'image à analyser")
    parser.add_argument("--onnx", action="store_true", help="Utiliser le modèle ONNX au lieu de PyTorch")
    parser.add_argument("--food101", action="store_true", help="Tester sur des échantillons Food-101")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de prédictions à afficher (défaut: 5)")

    args = parser.parse_args()

    if not args.image and not args.food101:
        parser.print_help()
        print("\n❌ Spécifie une image ou utilise --food101 pour tester")
        sys.exit(1)

    # Charger le modèle
    if args.onnx:
        model_or_session, idx_to_label = load_onnx_model()
    else:
        model_or_session, idx_to_label = load_pytorch_model()

    # Mode test Food-101
    if args.food101:
        test_food101_samples(model_or_session, idx_to_label, use_onnx=args.onnx)
        return

    # Mode test image unique
    image = load_image(args.image)
    logger.info(f"📷 Image : {args.image} ({image.size[0]}×{image.size[1]})")

    if args.onnx:
        results, elapsed = predict_onnx(model_or_session, image, idx_to_label, args.top_k)
    else:
        results, elapsed = predict_pytorch(model_or_session, image, idx_to_label, args.top_k)

    display_results(results, elapsed, args.image)


if __name__ == "__main__":
    main()
