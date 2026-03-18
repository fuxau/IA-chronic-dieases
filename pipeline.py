#!/usr/bin/env python3
"""
==============================================================================
  pipeline.py — Script tout-en-un de reconnaissance alimentaire
==============================================================================

  Récupération des données  →  Préparation  →  Fine-tuning EfficientNet  →  Export ONNX

  Usage :
      python pipeline.py

  Ce script fait tout automatiquement :
      1. Télécharge Food-101 (75 750 train / 25 250 test)
      2. Fusionne avec ton dataset custom (si présent)
      3. Fine-tune un EfficientNet-B0 pré-entraîné ImageNet
      4. Sauvegarde les poids (.pth) et exporte en ONNX (.onnx)

  ─────────────────────────────────────────────────────────────
  STRUCTURE DES DOSSIERS POUR LE DATASET CUSTOM
  ─────────────────────────────────────────────────────────────

  Pour ajouter tes propres aliments, crée un dossier "custom_dataset/"
  à la racine du projet avec la structure suivante :

      custom_dataset/
      ├── couscous/                 ← Nom de la classe (en minuscules, pas d'accents)
      │   ├── img_001.jpg
      │   ├── img_002.jpg
      │   └── ... (50-100 images recommandées)
      ├── tajine/
      │   ├── img_001.jpg
      │   └── ...
      ├── harira/
      │   └── ...
      ├── msemmen/
      │   └── ...
      └── baghrir/
          └── ...

  Règles :
      - 1 sous-dossier = 1 classe
      - Noms de dossier en minuscules, underscores pour les espaces
      - Formats acceptés : .jpg, .jpeg, .png
      - Résolution minimale recommandée : 224×224 pixels
      - 50 à 100 images par classe pour de bons résultats
      - Si le dossier est vide ou absent, seul Food-101 est utilisé

==============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import timm
from PIL import Image, UnidentifiedImageError

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Modifie ces valeurs selon ton setup
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Chemins
    "data_dir": "./data",                       # Où Food-101 sera téléchargé
    "custom_dataset_dir": "./custom_dataset",   # Ton dataset perso (voir structure ci-dessus)
    "output_dir": "./models",                   # Poids et exports

    # Modèle
    "model_name": "efficientnet_b0",            # Architecture (timm)
    "img_size": 224,                            # Taille d'entrée du réseau
    "pretrained": True,                         # Poids pré-entraînés ImageNet

    # Entraînement
    "batch_size": 32,                           # Taille du batch (réduis à 16 si GPU < 8GB)
    "epochs": 15,                               # Nombre d'époques total
    "learning_rate": 1e-4,                      # Learning rate initial
    "weight_decay": 1e-5,                       # Régularisation L2
    "label_smoothing": 0.1,                     # Lissage des labels (anti-surapprentissage)
    "num_workers": min(os.cpu_count() or 4, 8), # Workers pour le DataLoader

    # Fine-tuning progressif
    "freeze_backbone": True,                    # Phase 1 : geler le backbone
    "unfreeze_after_epoch": 5,                  # Phase 2 : dégeler après N époques
    "unfreeze_layers": 3,                       # Nb de blocs à dégeler en phase 2

    # Export ONNX
    "onnx_opset": 17,                           # Version opset ONNX

    # ImageNet stats (ne pas modifier)
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  0. WRAPPER ROBUSTE POUR IMAGES CORROMPUES
# ══════════════════════════════════════════════════════════════════════════════

class SafeDataset(Dataset):
    """
    Wrapper qui intercepte les images corrompues dans un dataset.

    Food-101 contient quelques fichiers corrompus (ex: ramen/3721099.jpg).
    Au lieu de crasher, ce wrapper :
      1. Tente de charger l'image normalement
      2. Si l'image est corrompue → retourne une autre image aléatoire
      3. Log les fichiers problématiques pour audit
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.corrupted_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        try:
            return self.dataset[idx]
        except (UnidentifiedImageError, OSError, IOError, Exception) as e:
            # L'image est corrompue — on log et on prend une autre image
            if idx not in self.corrupted_indices:
                self.corrupted_indices.add(idx)
                logger.warning(f"⚠️  Image corrompue (index {idx}), remplacée : {e}")

            # Essayer l'image suivante (avec wrap-around)
            for offset in range(1, 100):
                fallback_idx = (idx + offset) % len(self.dataset)
                try:
                    return self.dataset[fallback_idx]
                except Exception:
                    continue

            raise RuntimeError(f"Impossible de trouver une image valide autour de l'index {idx}")


def get_device() -> torch.device:
    """
    Détecte le meilleur device disponible : CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  1. PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def get_train_transforms() -> transforms.Compose:
    """
    Augmentations pour l'entraînement — réduit le surapprentissage.

    Applique aléatoirement :
        - Recadrage aléatoire (80-100% de l'image)
        - Retournement horizontal (50% de chance)
        - Rotation (-15° à +15°)
        - Variation de couleur (luminosité, contraste, saturation)
        - Translation légère
        - Effacement aléatoire d'un patch (CutOut)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(CONFIG["img_size"], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # CutOut
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Transformations pour validation/test — AUCUNE augmentation.
    Resize à 256 puis CenterCrop à 224 (standard ImageNet).
    """
    return transforms.Compose([
        transforms.Resize(CONFIG["img_size"] + 32),   # 256
        transforms.CenterCrop(CONFIG["img_size"]),     # 224
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
    ])


def prepare_data() -> tuple[DataLoader, DataLoader, dict[str, int], int]:
    """
    Prépare tout le pipeline de données :
        1. Télécharge Food-101 automatiquement (≈5 GB, première exécution uniquement)
        2. Charge le dataset custom s'il existe
        3. Fusionne les deux et construit les DataLoaders

    Returns:
        train_loader  — DataLoader d'entraînement
        val_loader    — DataLoader de validation
        class_to_idx  — Mapping {nom_classe: index}
        num_classes   — Nombre total de classes
    """
    data_dir = CONFIG["data_dir"]
    custom_dir = CONFIG["custom_dataset_dir"]
    os.makedirs(data_dir, exist_ok=True)

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    # ─── Food-101 ─────────────────────────────────────────────
    logger.info("📥 Chargement de Food-101 (téléchargement auto si nécessaire)...")

    food101_train_raw = datasets.Food101(
        root=data_dir, split="train",
        transform=train_transform, download=True,
    )
    food101_val_raw = datasets.Food101(
        root=data_dir, split="test",
        transform=val_transform, download=True,
    )

    # Wrapper robuste : skippe les images corrompues (ex: ramen/3721099.jpg)
    food101_train = SafeDataset(food101_train_raw)
    food101_val = SafeDataset(food101_val_raw)

    logger.info(f"   Food-101 train : {len(food101_train):,} images")
    logger.info(f"   Food-101 val   : {len(food101_val):,} images")
    logger.info(f"   Classes        : {len(food101_train_raw.classes)}")

    all_classes = sorted(food101_train_raw.classes)
    train_datasets = [food101_train]
    val_datasets = [food101_val]

    # ─── Dataset custom (optionnel) ───────────────────────────
    custom_classes = []

    if os.path.isdir(custom_dir):
        subdirs = [
            d for d in os.listdir(custom_dir)
            if os.path.isdir(os.path.join(custom_dir, d)) and not d.startswith(".")
        ]
        if subdirs:
            logger.info(f"📂 Dataset custom détecté : {custom_dir}/")

            # Charger avec ImageFolder (1 sous-dossier = 1 classe)
            custom_full = datasets.ImageFolder(custom_dir, transform=train_transform)
            custom_classes = custom_full.classes
            logger.info(f"   Classes custom : {custom_classes}")
            logger.info(f"   Images         : {len(custom_full):,}")

            # Split 80% train / 20% val
            n_total = len(custom_full)
            n_train = int(n_total * 0.8)
            n_val = n_total - n_train

            gen = torch.Generator().manual_seed(42)
            custom_train, _ = random_split(custom_full, [n_train, n_val], generator=gen)

            # Re-créer le val avec les transforms de validation (pas d'augmentation)
            custom_full_val = datasets.ImageFolder(custom_dir, transform=val_transform)
            _, custom_val = random_split(custom_full_val, [n_train, n_val], generator=gen)

            train_datasets.append(custom_train)
            val_datasets.append(custom_val)

            logger.info(f"   Split custom   : {n_train} train / {n_val} val")
        else:
            logger.info(f"📂 Dossier {custom_dir}/ vide — Food-101 seul sera utilisé")
    else:
        logger.info(f"ℹ️  Pas de dossier {custom_dir}/ — Food-101 seul sera utilisé")

    # ─── Mapping des classes unifié ───────────────────────────
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    # Ajouter les classes custom qui ne sont PAS déjà dans Food-101
    next_idx = len(class_to_idx)
    for cls in sorted(custom_classes):
        key = cls.lower().replace(" ", "_")
        if key not in class_to_idx:
            class_to_idx[key] = next_idx
            next_idx += 1
            logger.info(f"   + Nouvelle classe : '{key}' → index {next_idx - 1}")

    num_classes = len(class_to_idx)

    # Sauvegarder le mapping des labels (nécessaire pour l'inférence)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    labels_path = os.path.join(CONFIG["output_dir"], "class_labels.json")
    idx_to_label = {str(v): k for k, v in class_to_idx.items()}
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_label": idx_to_label,
            "num_classes": num_classes,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Labels sauvegardés : {labels_path}")

    # ─── Fusionner et créer les DataLoaders ───────────────────
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    # pin_memory=True accélère le transfert CPU→GPU, mais n'est pas supporté
    # sur Apple MPS — on le désactive automatiquement
    device = get_device()
    use_pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=use_pin_memory,
    )

    logger.info(f"✅ Données prêtes : {len(train_ds):,} train / {len(val_ds):,} val / {num_classes} classes")
    logger.info(f"   Batches train : {len(train_loader):,} / val : {len(val_loader):,}")

    return train_loader, val_loader, class_to_idx, num_classes


# ══════════════════════════════════════════════════════════════════════════════
#  2. CONSTRUCTION DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int) -> nn.Module:
    """
    Construit le modèle EfficientNet-B0 avec un classifier personnalisé.

    Architecture du classifier :
        EfficientNet features (1280 dims)
          → Dropout(0.3)
          → Linear(1280, 512) + ReLU
          → Dropout(0.2)
          → Linear(512, num_classes)

    Le backbone est chargé avec les poids ImageNet pré-entraînés.

    Args:
        num_classes: Nombre total de classes (Food-101 + custom)

    Returns:
        Modèle PyTorch prêt pour l'entraînement
    """
    logger.info(f"🏗️  Construction du modèle : {CONFIG['model_name']}")

    # Charger EfficientNet-B0 pré-entraîné sur ImageNet
    model = timm.create_model(
        CONFIG["model_name"],
        pretrained=CONFIG["pretrained"],
        num_classes=num_classes,
    )

    # Remplacer le classifier par défaut par un classifier plus robuste
    # EfficientNet-B0 a un attribut 'classifier' (pas 'fc')
    in_features = model.classifier.in_features  # 1280 pour EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Architecture   : {CONFIG['model_name']}")
    logger.info(f"   Features       : {in_features} → 512 → {num_classes}")
    logger.info(f"   Paramètres     : {total_params:,}")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Gèle le backbone (toutes les couches SAUF le classifier).

    C'est la PHASE 1 du fine-tuning :
    On entraîne uniquement le nouveau classifier pendant quelques époques.
    Les features ImageNet restent intactes.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"🔒 Backbone GELÉ — {trainable:,} / {total:,} paramètres entraînables")


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 3) -> None:
    """
    Dégèle les N derniers blocs du backbone + le classifier.

    C'est la PHASE 2 du fine-tuning :
    On affine les features de haut niveau en plus du classifier,
    avec un learning rate réduit pour ne pas casser les poids ImageNet.
    """
    # Tout geler d'abord
    for param in model.parameters():
        param.requires_grad = False

    # Dégeler le classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Dégeler les derniers blocs du backbone EfficientNet
    if hasattr(model, "blocks"):
        total_blocks = len(model.blocks)
        start = max(0, total_blocks - num_blocks)
        for i in range(start, total_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
        logger.info(f"🔓 Dégelé : blocs [{start}..{total_blocks - 1}] du backbone")

    # Dégeler aussi les couches finales (bn2, conv_head)
    for layer_name in ["bn2", "conv_head"]:
        if hasattr(model, layer_name):
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"   Paramètres entraînables : {trainable:,} / {total:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Entraîne le modèle sur une époque complète. Retourne la loss moyenne."""
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward + gradient clipping (stabilité)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # Log toutes les 100 itérations
        if (i + 1) % 100 == 0:
            logger.info(
                f"   Batch {i + 1:>5}/{num_batches} │ "
                f"Loss: {running_loss / (i + 1):.4f}"
            )

    return running_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Évalue le modèle sur le set de validation.

    Returns:
        val_loss     — Loss moyenne sur la validation
        top1_acc     — Accuracy Top-1 (%)
        top5_acc     — Accuracy Top-5 (%)
    """
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        total += labels.size(0)

        # Top-1 accuracy
        _, predicted = outputs.max(dim=1)
        correct_top1 += predicted.eq(labels).sum().item()

        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

    return (
        running_loss / len(loader),
        100.0 * correct_top1 / total,
        100.0 * correct_top5 / total,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_to_idx: dict[str, int],
    num_classes: int,
) -> nn.Module:
    """
    Pipeline d'entraînement complet en 2 phases.

    Phase 1 (époques 1 → 5)  : Backbone GELÉ
        → On entraîne uniquement le classifier sur les nouvelles classes.
        → Le réseau apprend à mapper les features ImageNet vers nos aliments.

    Phase 2 (époques 6 → 15) : Derniers blocs DÉGELÉS
        → On affine les features de haut niveau avec un LR réduit (×0.1).
        → Le réseau spécialise ses filtres pour la nourriture.

    Le meilleur modèle (basé sur val accuracy) est sauvegardé automatiquement.
    """
    # ─── Device ───────────────────────────────────────────────
    device = get_device()
    logger.info(f"🖥️  Device : {device}")
    if device.type == "cuda":
        logger.info(f"   GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif device.type == "mps":
        logger.info(f"   GPU    : Apple Silicon (MPS)")

    model = model.to(device)

    # ─── Phase 1 : geler le backbone ─────────────────────────
    if CONFIG["freeze_backbone"]:
        freeze_backbone(model)

    # ─── Loss, Optimizer, Scheduler ──────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-6,
    )

    # ─── Historique ──────────────────────────────────────────
    best_acc = 0.0
    best_model_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
    history = {"train_loss": [], "val_loss": [], "top1": [], "top5": [], "lr": []}

    logger.info("")
    logger.info("═" * 70)
    logger.info("  ENTRAÎNEMENT — EfficientNet-B0 Fine-tuning")
    logger.info("═" * 70)
    logger.info(f"  Époques     : {CONFIG['epochs']}")
    logger.info(f"  Batch size  : {CONFIG['batch_size']}")
    logger.info(f"  LR initial  : {CONFIG['learning_rate']}")
    logger.info(f"  Classes     : {num_classes}")
    logger.info(f"  Stratégie   : Phase 1 (gelé) → Phase 2 (dégel epoch {CONFIG['unfreeze_after_epoch'] + 1})")
    logger.info("═" * 70)
    logger.info("")

    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.time()

        # ─── Phase 2 : dégeler les derniers blocs ────────────
        if epoch == CONFIG["unfreeze_after_epoch"] + 1:
            logger.info("")
            logger.info("─" * 70)
            logger.info(f"🔓 PHASE 2 : Dégel des {CONFIG['unfreeze_layers']} derniers blocs")
            logger.info("─" * 70)

            unfreeze_last_blocks(model, CONFIG["unfreeze_layers"])

            # Recréer optimizer avec LR réduit pour le fine-tuning
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CONFIG["learning_rate"] * 0.1,
                weight_decay=CONFIG["weight_decay"],
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["epochs"] - CONFIG["unfreeze_after_epoch"],
                eta_min=1e-7,
            )
            logger.info(f"   Nouveau LR : {CONFIG['learning_rate'] * 0.1:.1e}")
            logger.info("")

        # ─── Train ───────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # ─── Validation ──────────────────────────────────────
        val_loss, top1, top5 = validate(model, val_loader, criterion, device)

        # ─── Scheduler ───────────────────────────────────────
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # ─── Log ─────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        marker = " ★" if top1 > best_acc else ""

        logger.info(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} │ "
            f"Train: {train_loss:.4f} │ "
            f"Val: {val_loss:.4f} │ "
            f"Top-1: {top1:.2f}% │ "
            f"Top-5: {top5:.2f}% │ "
            f"LR: {lr:.2e} │ "
            f"{elapsed:.0f}s{marker}"
        )

        # ─── Historique ──────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["top1"].append(top1)
        history["top5"].append(top5)
        history["lr"].append(lr)

        # ─── Sauvegarder le meilleur modèle ──────────────────
        if top1 > best_acc:
            best_acc = top1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "config": {
                    "model_name": CONFIG["model_name"],
                    "img_size": CONFIG["img_size"],
                },
            }, best_model_path)
            logger.info(f"   💾 Meilleur modèle sauvegardé → {best_model_path}")

    # ─── Résumé ──────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("")
    logger.info("═" * 70)
    logger.info(f"  ✅ ENTRAÎNEMENT TERMINÉ")
    logger.info(f"     Durée totale     : {total_time / 60:.1f} minutes")
    logger.info(f"     Meilleur Top-1   : {best_acc:.2f}%")
    logger.info(f"     Meilleur modèle  : {best_model_path}")
    logger.info("═" * 70)
    logger.info("")

    # Sauvegarder l'historique
    history_path = os.path.join(CONFIG["output_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Recharger les meilleurs poids
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cpu").eval()

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  4. EXPORT ONNX
# ══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(model: nn.Module, num_classes: int) -> str:
    """
    Exporte le modèle entraîné au format ONNX.

    Le format ONNX permet de :
        - Déployer sur mobile (ONNX Runtime Mobile / Core ML)
        - Servir en production sans PyTorch (plus léger, plus rapide)
        - Utiliser sur n'importe quel framework (TensorFlow, etc.)

    Étapes :
        1. Export via torch.onnx.export
        2. Validation structurelle avec onnx.checker
        3. Vérification numérique PyTorch vs ONNX Runtime

    Args:
        model     : Modèle PyTorch en mode eval (CPU)
        num_classes: Nombre de classes (pour vérification)

    Returns:
        Chemin du fichier ONNX généré
    """
    onnx_path = os.path.join(CONFIG["output_dir"], "food_model.onnx")
    model.eval()

    # Entrée factice : 1 image RGB 224×224
    dummy_input = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"])

    logger.info("📦 Export ONNX en cours...")
    logger.info(f"   Input shape : {list(dummy_input.shape)}")
    logger.info(f"   Opset       : {CONFIG['onnx_opset']}")

    # ─── Export ───────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=CONFIG["onnx_opset"],
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"   ✅ Export réussi : {onnx_path} ({file_size:.1f} MB)")

    # ─── Validation structurelle ─────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("   ✅ Validation ONNX : structure valide")
    except ImportError:
        logger.warning("   ⚠️  Package 'onnx' non installé — validation structurelle ignorée")
    except Exception as e:
        logger.error(f"   ❌ Validation ONNX échouée : {e}")

    # ─── Vérification numérique ──────────────────────────────
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # Sortie PyTorch
        with torch.no_grad():
            pt_output = model(dummy_input).numpy()

        # Sortie ONNX Runtime
        ort_output = session.run(None, {input_name: dummy_input.numpy()})[0]

        # Comparaison
        max_diff = np.max(np.abs(pt_output - ort_output))
        logger.info(f"   Écart max PyTorch↔ONNX : {max_diff:.2e}")

        if max_diff < 1e-5:
            logger.info("   ✅ Cohérence numérique validée")
        else:
            logger.warning(f"   ⚠️  Écart supérieur au seuil ({max_diff:.2e} > 1e-5)")

    except ImportError:
        logger.warning("   ⚠️  Package 'onnxruntime' non installé — vérification ignorée")
    except Exception as e:
        logger.error(f"   ❌ Vérification numérique échouée : {e}")

    return onnx_path


# ══════════════════════════════════════════════════════════════════════════════
#  5. POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Pipeline complet exécuté séquentiellement :
        1. prepare_data()    → Téléchargement + préparation des DataLoaders
        2. build_model()     → Construction d'EfficientNet-B0 customisé
        3. train()           → Fine-tuning en 2 phases
        4. export_to_onnx()  → Sauvegarde finale au format ONNX

    Fichiers générés dans ./models/ :
        - best_model.pth        → Checkpoint PyTorch (pour reprendre l'entraînement)
        - food_model.onnx       → Modèle optimisé pour la production
        - class_labels.json     → Mapping index ↔ nom d'aliment
        - training_history.json → Métriques par époque
    """
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  🍽️  MODULE IA — RECONNAISSANCE ALIMENTAIRE                ║")
    logger.info("║  Pipeline : Données → Entraînement → Export ONNX          ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")

    start = time.time()

    # Étape 1 : Données
    logger.info("━" * 70)
    logger.info("  ÉTAPE 1/4 — PRÉPARATION DES DONNÉES")
    logger.info("━" * 70)
    train_loader, val_loader, class_to_idx, num_classes = prepare_data()

    # Étape 2 : Modèle
    logger.info("")
    logger.info("━" * 70)
    logger.info("  ÉTAPE 2/4 — CONSTRUCTION DU MODÈLE")
    logger.info("━" * 70)
    model = build_model(num_classes)

    # Étape 3 : Entraînement
    logger.info("")
    logger.info("━" * 70)
    logger.info("  ÉTAPE 3/4 — ENTRAÎNEMENT (FINE-TUNING)")
    logger.info("━" * 70)
    model = train(model, train_loader, val_loader, class_to_idx, num_classes)

    # Étape 4 : Export ONNX
    logger.info("")
    logger.info("━" * 70)
    logger.info("  ÉTAPE 4/4 — EXPORT ONNX")
    logger.info("━" * 70)
    onnx_path = export_to_onnx(model, num_classes)

    # Résumé final
    total = time.time() - start
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  ✅ PIPELINE TERMINÉ AVEC SUCCÈS                           ║")
    logger.info("╠══════════════════════════════════════════════════════════════╣")
    logger.info(f"║  Durée totale : {total / 60:.1f} minutes")
    logger.info(f"║  Classes      : {num_classes}")
    logger.info(f"║  Modèle ONNX  : {onnx_path}")
    logger.info(f"║  Labels       : {CONFIG['output_dir']}/class_labels.json")
    logger.info("║                                                            ║")
    logger.info("║  Prochaine étape :                                         ║")
    logger.info("║    python inference_api.py                                 ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
