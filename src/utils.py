import sounddevice as sd
import os
import torch
import mss
from PIL import Image
import numpy as np
from scipy.io.wavfile import write



def play_audio(filename):
    """Lecture simple d'un fichier audio"""
    try:
        # On utilise aplay (standard Linux) pour éviter les conflits de flux
        os.system(f"aplay {filename}")
    except:
        import soundfile as sf
        data, fs = sf.read(filename)
        sd.play(data, fs)
        sd.wait()

def record_audio(vad_model):
    target_fs = 16000
    filename = "data/results/input_user.wav"
    
    # print("\n--- PRÊT À ÉCOUTER ---")
    # input("Appuyez sur ENTRÉE pour commencer à parler...")

    recording = []
    # Paramètres de détection
    CHUNK_SIZE = 512 # Nombre de samples par morceau (doit être 512, 1024 ou 1536 pour Silero)
    SILENCE_LIMIT = 1 # Secondes de silence avant de couper
    
    silence_counter = 0
    speech_detected = False

    print("Écoute en cours... Parlez maintenant.")

    # Ouverture du flux micro
    with sd.InputStream(samplerate=target_fs, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
        while True:
            # Lire un morceau d'audio
            audio_chunk, overflowed = stream.read(CHUNK_SIZE)
            recording.append(audio_chunk)
            
            # Transformer en tenseur pour le modèle VAD
            tensor_chunk = torch.from_numpy(audio_chunk.flatten())
            
            # Obtenir la probabilité que ce soit de la parole (0 à 1)
            speech_prob = vad_model(tensor_chunk, target_fs).item()
            
            if speech_prob > 0.5: # Seuil de détection de parole
                if not speech_detected:
                    print("● Parole détectée...")
                speech_detected = True
                silence_counter = 0 # Reset le compteur de silence
            elif speech_detected:
                silence_counter += CHUNK_SIZE / target_fs
                
            # Si on a détecté de la parole PUIS un long silence : on s'arrête
            if speech_detected and silence_counter > SILENCE_LIMIT:
                print("Fin de parole détectée.")
                break
            
            # Sécurité : si rien n'est dit après 5 secondes, on annule
            if not speech_detected and len(recording) * CHUNK_SIZE / target_fs > 5:
                print("Aucun son détecté, fermeture.")
                return None

    # Assemblage et sauvegarde
    audio_full = np.concatenate(recording, axis=0)
    
    # Normalisation
    max_vol = np.max(np.abs(audio_full))
    if max_vol > 0:
        audio_full = audio_full / max_vol
        
    audio_int16 = (audio_full * 32767).astype(np.int16)
    write(filename, target_fs, audio_int16)
    
    return filename

def take_screenshot():
    save_file = "data/results/last_screenshot.jpg"
    with mss.mss() as sct:
        # Capture l'écran principal
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        
        # Convertir en image PIL
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # essayer de resize /2 voir si ça va plus vite et c'est ok
        img.thumbnail((896, 896))
                
        img.save(save_file)
        return save_file