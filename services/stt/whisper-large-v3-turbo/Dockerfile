# On utilise l'image officielle de PyTorch (déjà configurée pour NVIDIA)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY server.py .

# Le port que le conteneur va ouvrir
EXPOSE 8000

# Lancement du serveur
CMD ["python", "server.py"]