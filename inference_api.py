"""
inference_api.py — API FastAPI pour l'inférence de reconnaissance alimentaire.

Endpoints :
  POST /predict       — Upload d'image → label + calories + macros
  GET  /health        — Health check
  GET  /labels        — Liste des classes reconnues
  GET  /nutrition/{label} — Données nutritionnelles pour un aliment

L'inférence utilise ONNX Runtime pour des performances optimales (< 3s).
"""

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from nutrition_table import calculate_calories, get_nutrition, list_available_foods

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Modèles Pydantic (Schemas de réponse)
# ──────────────────────────────────────────────
class NutritionResponse(BaseModel):
    """Informations nutritionnelles calculées."""
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class PredictionResponse(BaseModel):
    """Réponse complète de prédiction."""
    label: str
    confidence: float
    nutrition: NutritionResponse
    portion_g: float
    default_portion_g: float
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    num_classes: int
    model_path: str


# ──────────────────────────────────────────────
# Moteur d'inférence ONNX
# ──────────────────────────────────────────────
class FoodRecognitionEngine:
    """Moteur de reconnaissance alimentaire basé sur ONNX Runtime."""

    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self.idx_to_label: dict[int, str] = {}
        self.num_classes: int = 0
        self._loaded: bool = False

    def load(self, model_path: Path | None = None, labels_path: Path | None = None):
        """Charge le modèle ONNX et les labels."""
        model_path = model_path or config.ONNX_MODEL_PATH
        labels_path = labels_path or config.CLASS_LABELS_PATH

        if not model_path.exists():
            logger.warning(
                f"Modèle ONNX introuvable : {model_path}. "
                "L'API démarre en mode dégradé (sans modèle). "
                "Lancez d'abord : python train.py && python export_onnx.py"
            )
            return

        # Charger le modèle ONNX
        logger.info(f"Chargement du modèle ONNX : {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )

        # Charger les labels
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
            self.idx_to_label = {int(k): v for k, v in labels_data["idx_to_label"].items()}
            self.num_classes = labels_data["num_classes"]
        else:
            logger.warning(f"Fichier de labels introuvable : {labels_path}")

        self._loaded = True
        logger.info(f"  ✅ Modèle chargé ({self.num_classes} classes)")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.session is not None

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Prétraitement de l'image pour l'inférence.

        Pipeline : Resize → CenterCrop → Normalize → NCHW format
        """
        # Resize (côté le plus court = 256)
        target_size = config.IMG_SIZE + 32  # 256
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_size = (int(target_size * aspect_ratio), target_size)
        else:
            new_size = (target_size, int(target_size / aspect_ratio))
        image = image.resize(new_size, Image.LANCZOS)

        # Center crop 224×224
        left = (image.width - config.IMG_SIZE) // 2
        top = (image.height - config.IMG_SIZE) // 2
        image = image.crop((left, top, left + config.IMG_SIZE, top + config.IMG_SIZE))

        # Convertir en array float32 [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normaliser (ImageNet)
        mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(config.IMAGENET_STD, dtype=np.float32)
        img_array = (img_array - mean) / std

        # HWC → CHW → NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image: Image.Image) -> tuple[str, float, float]:
        """
        Exécute l'inférence sur une image.

        Args:
            image: Image PIL en RGB.

        Returns:
            (label, confidence, inference_time_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("Modèle non chargé")

        # Prétraitement
        input_tensor = self.preprocess(image)

        # Inférence ONNX
        start_time = time.perf_counter()
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Post-traitement : softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))  # stabilité numérique
        probabilities = exp_logits / exp_logits.sum()

        # Meilleure prédiction
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        label = self.idx_to_label.get(predicted_idx, f"class_{predicted_idx}")

        return label, confidence, inference_time


# ──────────────────────────────────────────────
# Instance globale du moteur
# ──────────────────────────────────────────────
engine = FoodRecognitionEngine()


# ──────────────────────────────────────────────
# Application FastAPI
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'application."""
    logger.info("🚀 Démarrage de l'API de Reconnaissance Alimentaire")
    engine.load()
    yield
    logger.info("🛑 Arrêt de l'API")


app = FastAPI(
    title="Food Recognition API",
    description=(
        "API de reconnaissance d'images alimentaires et d'estimation calorique. "
        "Uploadez une photo de plat pour obtenir le nom de l'aliment, "
        "le score de confiance et les informations nutritionnelles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permettre les appels depuis n'importe quel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Vérifie l'état de santé de l'API et du modèle."""
    return HealthResponse(
        status="healthy" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        num_classes=engine.num_classes,
        model_path=str(config.ONNX_MODEL_PATH),
    )


@app.get("/labels", tags=["System"])
async def get_labels():
    """Retourne la liste de toutes les classes alimentaires reconnues."""
    if not engine.is_loaded:
        return {"labels": list_available_foods(), "source": "nutrition_table"}
    return {
        "labels": list(engine.idx_to_label.values()),
        "num_classes": engine.num_classes,
        "source": "model",
    }


@app.get("/nutrition/{label}", tags=["Nutrition"])
async def get_nutrition_info(
    label: str,
    portion_g: float = Query(default=100.0, ge=1, le=2000, description="Portion en grammes"),
):
    """Retourne les informations nutritionnelles pour un aliment donné."""
    result = calculate_calories(label, portion_g)
    info = get_nutrition(label)
    result["default_portion_g"] = info.default_portion_g
    return result


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Image du plat (JPEG, PNG)"),
    portion_g: float = Query(
        default=None,
        ge=1,
        le=2000,
        description="Portion en grammes (défaut : portion standard de l'aliment)",
    ),
):
    """
    Analyse une image de plat et retourne :
    - Le label de l'aliment identifié
    - Le score de confiance
    - Les informations nutritionnelles (calories, macros)
    - Le temps d'inférence

    L'inférence s'exécute via ONNX Runtime pour des performances < 3s.
    """
    # Vérifiez que le modèle est chargé
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modèle non disponible. "
                "Lancez d'abord l'entraînement et l'export ONNX."
            ),
        )

    # Valider le type de fichier
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {file.content_type}. Utilisez JPEG, PNG ou WebP.",
        )

    # Lire et convertir l'image
    try:
        contents = await file.read()
        if len(contents) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Image trop volumineuse (max {config.MAX_FILE_SIZE_MB} MB)",
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image : {str(e)}")

    # Inférence
    try:
        label, confidence, inference_time_ms = engine.predict(image)
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")

    # Vérifier le seuil de confiance
    if confidence < config.CONFIDENCE_THRESHOLD:
        logger.warning(f"Confiance faible : {label} ({confidence:.2%})")

    # Calcul des calories
    nutrition_info = get_nutrition(label)
    actual_portion = portion_g if portion_g is not None else nutrition_info.default_portion_g
    nutrition_data = calculate_calories(label, actual_portion)

    return PredictionResponse(
        label=label,
        confidence=round(confidence, 4),
        nutrition=NutritionResponse(
            calories=nutrition_data["calories"],
            protein_g=nutrition_data["protein_g"],
            carbs_g=nutrition_data["carbs_g"],
            fat_g=nutrition_data["fat_g"],
            fiber_g=nutrition_data["fiber_g"],
        ),
        portion_g=actual_portion,
        default_portion_g=nutrition_info.default_portion_g,
        inference_time_ms=round(inference_time_ms, 2),
    )


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference_api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
