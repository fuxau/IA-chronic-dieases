"""
data_preparation.py — Préparation du dataset pour l'entraînement.

Combine le dataset Food-101 (torchvision) avec un dataset personnalisé
d'aliments supplémentaires. Applique les augmentations et crée les
DataLoaders train/val.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Wrapper robuste pour images corrompues
# ──────────────────────────────────────────────
class SafeDataset(Dataset):
    """
    Wrapper qui intercepte les images corrompues dans un dataset.

    Food-101 contient quelques fichiers corrompus (ex: ramen/3721099.jpg,
    tiramisu/1321095.jpg). Au lieu de crasher, ce wrapper :
      1. Tente de charger l'image normalement
      2. Si l'image est corrompue → retourne une autre image
      3. Log les fichiers problématiques
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.corrupted_indices: set = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        try:
            return self.dataset[idx]
        except (UnidentifiedImageError, OSError, IOError, Exception) as e:
            if idx not in self.corrupted_indices:
                self.corrupted_indices.add(idx)
                logger.warning(f"⚠️  Image corrompue (index {idx}), remplacée : {e}")
            # Essayer les images suivantes
            for offset in range(1, 100):
                fallback_idx = (idx + offset) % len(self.dataset)
                try:
                    return self.dataset[fallback_idx]
                except Exception:
                    continue
            raise RuntimeError(f"Impossible de trouver une image valide autour de l'index {idx}")


# ──────────────────────────────────────────────
# Transformations
# ──────────────────────────────────────────────
def get_train_transforms() -> transforms.Compose:
    """Augmentations pour l'entraînement."""
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Transformations pour la validation (sans augmentation)."""
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE + 32),  # 256
        transforms.CenterCrop(config.IMG_SIZE),    # 224
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Transformations pour l'inférence (identiques à la validation)."""
    return get_val_transforms()


# ──────────────────────────────────────────────
# Chargement Food-101
# ──────────────────────────────────────────────
def load_food101(split: str = "train", transform=None) -> datasets.Food101:
    """
    Charge le dataset Food-101 via torchvision.

    Args:
        split: 'train' ou 'test'
        transform: Transformations à appliquer

    Returns:
        Dataset Food-101
    """
    logger.info(f"Chargement de Food-101 (split={split})...")
    dataset = datasets.Food101(
        root=str(config.DATA_DIR),
        split=split,
        transform=transform,
        download=True,
    )
    logger.info(f"  → {len(dataset)} images chargées, {len(dataset.classes)} classes")
    return SafeDataset(dataset), dataset.classes


# ──────────────────────────────────────────────
# Chargement du dataset personnalisé
# ──────────────────────────────────────────────
def load_custom_dataset(transform=None) -> datasets.ImageFolder | None:
    """
    Charge le dataset personnalisé depuis custom_dataset/.

    Structure attendue :
        custom_dataset/
        ├── couscous/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── tajine/
        │   ├── img001.jpg
        │   └── ...
        └── ...

    Returns:
        ImageFolder dataset ou None si le répertoire est vide.
    """
    custom_dir = config.CUSTOM_DATASET_DIR

    if not custom_dir.exists():
        logger.warning(f"Répertoire custom introuvable : {custom_dir}")
        return None

    # Vérifier qu'il y a des sous-dossiers avec des images
    subdirs = [d for d in custom_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        logger.info("Pas de dataset custom détecté — utilisation de Food-101 uniquement.")
        return None

    logger.info(f"Chargement du dataset custom depuis {custom_dir}...")
    dataset = datasets.ImageFolder(
        root=str(custom_dir),
        transform=transform,
    )
    logger.info(f"  → {len(dataset)} images, {len(dataset.classes)} classes custom : {dataset.classes}")
    return dataset


# ──────────────────────────────────────────────
# Construction des classes unifiées
# ──────────────────────────────────────────────
def build_unified_class_mapping(
    food101_classes: list[str],
    custom_classes: list[str] | None = None,
) -> dict[str, int]:
    """
    Construit un mapping unifié label → index.

    Les 101 classes Food-101 sont en premier, suivies des classes custom
    qui ne sont pas déjà dans Food-101.
    """
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(food101_classes))}

    if custom_classes:
        next_idx = len(class_to_idx)
        for cls in sorted(custom_classes):
            normalized = cls.lower().replace(" ", "_")
            if normalized not in class_to_idx:
                class_to_idx[normalized] = next_idx
                next_idx += 1
                logger.info(f"  + Classe custom ajoutée : '{normalized}' (idx={next_idx - 1})")

    return class_to_idx


def save_class_labels(class_to_idx: dict[str, int], path: Path | None = None):
    """Sauvegarde le mapping label→index en JSON."""
    path = path or config.CLASS_LABELS_PATH
    # Inverser pour avoir idx→label
    idx_to_label = {v: k for k, v in class_to_idx.items()}
    output = {
        "class_to_idx": class_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "num_classes": len(class_to_idx),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Labels sauvegardés dans {path} ({len(class_to_idx)} classes)")


# ──────────────────────────────────────────────
# Création des DataLoaders
# ──────────────────────────────────────────────
def create_dataloaders(
    include_custom: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, int], int]:
    """
    Crée les DataLoaders train et validation.

    Workflow :
    1. Charger Food-101 train + test
    2. (Optionnel) Charger dataset custom et split 80/20
    3. Fusionner avec ConcatDataset
    4. Construire le mapping de classes unifié

    Returns:
        (train_loader, val_loader, class_to_idx, num_classes)
    """
    # --- Food-101 ---
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    food101_train, food101_classes = load_food101(split="train", transform=train_transform)
    food101_val, _ = load_food101(split="test", transform=val_transform)

    train_datasets = [food101_train]
    val_datasets = [food101_val]
    all_classes = list(food101_classes)

    # --- Custom dataset ---
    custom_classes = None
    if include_custom:
        custom_train = load_custom_dataset(transform=train_transform)
        if custom_train is not None:
            custom_classes = custom_train.classes

            # Split train/val
            n_total = len(custom_train)
            n_train = int(n_total * config.TRAIN_SPLIT)
            n_val = n_total - n_train

            custom_train_split, custom_val_split = random_split(
                custom_train, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

            # Recréer le dataset val avec les bonnes transformations
            custom_val = load_custom_dataset(transform=val_transform)
            _, custom_val_split = random_split(
                custom_val, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

            train_datasets.append(custom_train_split)
            val_datasets.append(custom_val_split)

            logger.info(f"Custom split : {n_train} train / {n_val} val")

    # --- Mapping des classes ---
    class_to_idx = build_unified_class_mapping(all_classes, custom_classes)
    num_classes = len(class_to_idx)
    save_class_labels(class_to_idx)

    # --- Fusionner les datasets ---
    combined_train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    combined_val = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    logger.info(f"Dataset final : {len(combined_train)} train / {len(combined_val)} val / {num_classes} classes")

    # --- DataLoaders ---
    # Détecter le device pour pin_memory (non supporté sur MPS)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        combined_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        combined_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, class_to_idx, num_classes


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, class_to_idx, num_classes = create_dataloaders()
    print(f"\n✅ Préparation terminée !")
    print(f"   Classes : {num_classes}")
    print(f"   Batches train : {len(train_loader)}")
    print(f"   Batches val   : {len(val_loader)}")
