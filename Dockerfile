# ────────────────────────────────────────────────────────────
# Dockerfile — Module de Reconnaissance Alimentaire (Inférence)
#
# Ce conteneur est optimisé pour l'inférence uniquement.
# L'entraînement doit être fait en dehors du conteneur.
#
# Build :  docker build -t food-recognition .
# Run   :  docker run -p 8000:8000 food-recognition
# ────────────────────────────────────────────────────────────

# ============================================================
# Stage 1 : Base avec dépendances
# ============================================================
FROM python:3.11-slim AS base

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 2 : Installation des dépendances Python
# ============================================================
FROM base AS dependencies

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# ============================================================
# Stage 3 : Application finale
# ============================================================
FROM dependencies AS production

# Copier le code de l'application
COPY config.py .
COPY nutrition_table.py .
COPY inference_api.py .

# Copier le modèle ONNX et les labels
# (doivent être générés AVANT le build Docker)
COPY models/food_model.onnx models/
COPY models/class_labels.json models/

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Métadonnées
LABEL maintainer="IA-chronic-diseases" \
      description="Module IA de reconnaissance alimentaire et estimation calorique" \
      version="1.0.0"

# Port exposé
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Point d'entrée
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
