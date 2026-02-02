# On utilise l'image officielle de PyTorch (déjà configurée pour NVIDIA)
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y torchcodec
# Copie du code
COPY server.py .

# Le port que le conteneur va ouvrir
EXPOSE 8000

# Lancement du serveur
CMD ["python", "server.py"]