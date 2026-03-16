"""
train.py — Script d'entraînement : Fine-tuning d'EfficientNet-B0 sur Food-101.

Stratégie en 2 phases :
  Phase 1 (époques 1→5)  : Backbone gelé, seul le classifier est entraîné.
  Phase 2 (époques 6→15) : Derniers blocs du backbone dégelés pour affinage.
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

import config
from data_preparation import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Construction du modèle
# ──────────────────────────────────────────────
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Construit un EfficientNet-B0 avec un classifier personnalisé.

    Args:
        num_classes: Nombre de classes de sortie.
        pretrained: Utiliser les poids pré-entraînés ImageNet.

    Returns:
        Modèle EfficientNet-B0 configuré.
    """
    logger.info(f"Construction du modèle {config.MODEL_NAME} (pretrained={pretrained})...")

    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Récupérer la taille de la couche avant le classifier
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    logger.info(f"  → Classifier : {in_features} → 512 → {num_classes}")
    return model


def freeze_backbone(model: nn.Module):
    """Gèle toutes les couches sauf le classifier."""
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Backbone gelé : {trainable:,} / {total:,} paramètres entraînables")


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 3):
    """Dégèle les derniers blocs du backbone pour le fine-tuning."""
    # D'abord, tout geler
    for param in model.parameters():
        param.requires_grad = False

    # Dégeler le classifier
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True

    # Dégeler les derniers blocs du backbone
    blocks = list(model.named_children())
    # EfficientNet a des blocs dans model.blocks
    if hasattr(model, 'blocks'):
        total_blocks = len(model.blocks)
        for i in range(max(0, total_blocks - num_blocks), total_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
        logger.info(f"  Dégelé : blocs {total_blocks - num_blocks}→{total_blocks - 1} du backbone")

    # Dégeler aussi bn2 et conv_head si présents
    for name in ['bn2', 'conv_head']:
        if hasattr(model, name):
            for param in getattr(model, name).parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Paramètres entraînables : {trainable:,} / {total:,}")


# ──────────────────────────────────────────────
# Boucle d'entraînement
# ──────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Entraîne le modèle pour une époque. Retourne la loss moyenne."""
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            avg = running_loss / (batch_idx + 1)
            logger.info(f"  Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} | Loss: {avg:.4f}")

    return running_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Évalue le modèle sur le set de validation.

    Returns:
        (val_loss, top1_accuracy, top5_accuracy)
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

        # Top-1
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()

        # Top-5
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()

    val_loss = running_loss / len(loader)
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    return val_loss, top1_acc, top5_acc


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────
def train():
    """Pipeline d'entraînement complet."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if device.type == "cuda":
        logger.info(f"  GPU : {torch.cuda.get_device_name(0)}")

    # Dataset
    train_loader, val_loader, class_to_idx, num_classes = create_dataloaders()

    # Modèle
    model = build_model(num_classes, pretrained=config.PRETRAINED)
    model = model.to(device)

    # Phase 1 : geler le backbone
    if config.FREEZE_BACKBONE:
        freeze_backbone(model)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # Historique
    best_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_top1_acc": [],
        "val_top5_acc": [],
        "lr": [],
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Début de l'entraînement — {config.EPOCHS} époques")
    logger.info(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # Phase 2 : dégeler les derniers blocs
        if epoch == config.UNFREEZE_AFTER_EPOCH + 1:
            logger.info(f"\n🔓 Phase 2 : Dégel des {config.UNFREEZE_LAYERS} derniers blocs")
            unfreeze_last_blocks(model, config.UNFREEZE_LAYERS)
            # Recréer l'optimizer pour les nouveaux paramètres
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.LEARNING_RATE * 0.1,  # LR réduit pour le fine-tuning
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.EPOCHS - config.UNFREEZE_AFTER_EPOCH,
                eta_min=1e-7,
            )

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validation
        val_loss, top1_acc, top5_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_top1_acc"].append(top1_acc)
        history["val_top5_acc"].append(top5_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1: {top1_acc:.2f}% | "
            f"Top-5: {top5_acc:.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Sauvegarder le meilleur modèle
        if top1_acc > best_acc:
            best_acc = top1_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'class_to_idx': class_to_idx,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'img_size': config.IMG_SIZE,
                },
            }, config.BEST_MODEL_PATH)
            logger.info(f"  ✅ Meilleur modèle sauvegardé (Top-1: {best_acc:.2f}%)")

    # Résumé
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Entraînement terminé en {total_time / 60:.1f} minutes")
    logger.info(f"Meilleur Top-1 Accuracy : {best_acc:.2f}%")
    logger.info(f"Modèle sauvegardé : {config.BEST_MODEL_PATH}")
    logger.info(f"{'='*60}")

    # Sauvegarder l'historique
    history_path = config.MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Historique sauvegardé : {history_path}")


if __name__ == "__main__":
    train()
