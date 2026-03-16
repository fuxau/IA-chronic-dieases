"""
export_onnx.py — Export du modèle PyTorch entraîné au format ONNX.

Étapes :
1. Charger le checkpoint PyTorch (.pth)
2. Exporter au format ONNX avec torch.onnx.export
3. Valider la structure avec onnx.checker
4. Vérifier la cohérence numérique PyTorch vs ONNX Runtime
5. Exporter les labels de classes
"""

import json
import logging

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import timm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path=None) -> tuple[nn.Module, dict]:
    """
    Charge le modèle entraîné depuis un checkpoint.

    Returns:
        (model, checkpoint_info) — Le modèle en mode eval et les métadonnées.
    """
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            "Lancez d'abord l'entraînement avec : python train.py"
        )

    logger.info(f"Chargement du checkpoint : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = checkpoint['num_classes']
    model_name = checkpoint['config']['model_name']

    # Recréer l'architecture
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Recréer le classifier custom (doit correspondre à train.py)
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    info = {
        'epoch': checkpoint.get('epoch', '?'),
        'best_acc': checkpoint.get('best_acc', '?'),
        'num_classes': num_classes,
    }
    logger.info(f"  → Modèle chargé (epoch={info['epoch']}, acc={info['best_acc']}%, classes={num_classes})")

    return model, info


def export_to_onnx(model: nn.Module, output_path=None) -> str:
    """
    Exporte le modèle PyTorch au format ONNX.

    Args:
        model: Modèle PyTorch en mode eval.
        output_path: Chemin de sortie .onnx.

    Returns:
        Chemin du fichier ONNX généré.
    """
    output_path = output_path or config.ONNX_MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Entrée factice
    dummy_input = torch.randn(1, config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    logger.info(f"Export ONNX vers : {output_path}")
    logger.info(f"  Input shape : {dummy_input.shape}")
    logger.info(f"  Opset version : {config.ONNX_OPSET_VERSION}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.ONNX_OPSET_VERSION,
        do_constant_folding=True,          # Optimisation des constantes
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},     # Batch dynamique
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"  ✅ Export réussi ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return str(output_path)


def validate_onnx(onnx_path=None):
    """Valide la structure du modèle ONNX."""
    onnx_path = onnx_path or config.ONNX_MODEL_PATH

    logger.info("Validation de la structure ONNX...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("  ✅ Modèle ONNX valide")

    # Afficher les métadonnées
    graph = onnx_model.graph
    logger.info(f"  Inputs  : {[i.name for i in graph.input]}")
    logger.info(f"  Outputs : {[o.name for o in graph.output]}")
    logger.info(f"  Nodes   : {len(graph.node)}")


def verify_numerical_consistency(model: nn.Module, onnx_path=None, tolerance: float = 1e-5):
    """
    Vérifie que les sorties PyTorch et ONNX Runtime sont cohérentes.

    Compare les logits sur une entrée aléatoire et vérifie que l'écart
    maximal est inférieur au seuil de tolérance.
    """
    onnx_path = str(onnx_path or config.ONNX_MODEL_PATH)

    logger.info(f"Vérification de la cohérence numérique (tolérance={tolerance})...")

    # Sortie PyTorch
    dummy_input = torch.randn(1, config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    # Sortie ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = session.run(None, ort_inputs)[0]

    # Comparaison
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    logger.info(f"  Écart max  : {max_diff:.2e}")
    logger.info(f"  Écart moyen: {mean_diff:.2e}")

    if max_diff < tolerance:
        logger.info(f"  ✅ Cohérence validée (écart < {tolerance})")
    else:
        logger.warning(f"  ⚠️  Écart supérieur au seuil ! ({max_diff:.2e} > {tolerance})")

    return max_diff < tolerance


def export_class_labels(checkpoint_path=None, output_path=None):
    """Exporte les labels de classes en JSON depuis le checkpoint."""
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH
    output_path = output_path or config.CLASS_LABELS_PATH

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint.get('class_to_idx', {})

    idx_to_label = {str(v): k for k, v in class_to_idx.items()}

    labels_data = {
        "class_to_idx": class_to_idx,
        "idx_to_label": idx_to_label,
        "num_classes": len(class_to_idx),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Labels exportés : {output_path} ({len(class_to_idx)} classes)")


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────
def main():
    """Pipeline complet d'export ONNX."""
    logger.info("=" * 60)
    logger.info("EXPORT ONNX — Module de Reconnaissance Alimentaire")
    logger.info("=" * 60)

    # 1. Charger le modèle entraîné
    model, info = load_trained_model()

    # 2. Exporter en ONNX
    onnx_path = export_to_onnx(model)

    # 3. Valider la structure
    validate_onnx()

    # 4. Vérifier la cohérence numérique
    is_consistent = verify_numerical_consistency(model)

    # 5. Exporter les labels
    export_class_labels()

    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("EXPORT TERMINÉ")
    logger.info(f"  Modèle ONNX : {config.ONNX_MODEL_PATH}")
    logger.info(f"  Labels       : {config.CLASS_LABELS_PATH}")
    logger.info(f"  Cohérence    : {'✅ OK' if is_consistent else '⚠️  ÉCART DÉTECTÉ'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
