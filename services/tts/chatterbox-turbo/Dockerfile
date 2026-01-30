# On utilise l'image officielle de PyTorch (déjà configurée pour NVIDIA)
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY server.py .

# Le port que le conteneur va ouvrir
EXPOSE 8000

# Lancement du serveur
CMD ["python", "server.py"]