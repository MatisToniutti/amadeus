import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import librosa
from speechToText import speechToText
from textGeneration import textGeneration
from textToSpeech import textToSpeech

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

def record_audio():
    """Version robuste pour PipeWire sans ROOT"""
    # 1. Configuration des fréquences
    target_fs = 16000 # Fréquence standard pour le STT (Whisper/Wav2Vec)
    duration = 3      # Durée fixe de 3 secondes comme tu le voulais
    filename = "results/input_user.wav"
    
    # Créer le dossier results s'il n'existe pas
    if not os.path.exists("results"):
        os.makedirs("results")

    print("\n--- PRÊT À ÉCOUTER ---")
    input("Appuyez sur ENTRÉE pour parler pendant 3 secondes...")

    try:
        # 2. Capture audio
        # On laisse sounddevice choisir le 'device=None' (par défaut PipeWire)
        print("Enregistrement en cours... (3s)")
        recording = sd.rec(int(duration * target_fs), samplerate=target_fs, channels=1, dtype='float32')
        sd.wait()
        
        # 3. Traitement et Normalisation
        max_vol = np.max(np.abs(recording))
        
        if max_vol < 0.001:
            print("!!! Aucun son détecté. Vérifiez votre micro dans pavucontrol.")
        else:
            # Normalisation pour que le STT comprenne bien
            recording = recording / max_vol
        
        # 4. Conversion en Int16 pour le format WAV standard
        audio_int16 = (recording * 32767).astype(np.int16)
        
        # 5. Sauvegarde
        write(filename, target_fs, audio_int16)
        print(f"Enregistrement terminé et sauvegardé.")
        
        return filename

    except Exception as e:
        print(f"Erreur lors de l'enregistrement : {e}")
        return None

def main():
    chat_history = []
    print("Système Amadeus activé.")
    print("Appuyez sur Ctrl+C pour arrêter.")

    try:
        while True:
            # 1. Écoute
            audio_path = record_audio()
            if not audio_path: continue

            # 2. Speech to Text
            user_text = speechToText(audio_path)
            print("Toi : " + user_text)
            if user_text.strip() == "": break

            # 3. Text Generation
            law_response = textGeneration(user_text)
            print("Law : " + law_response)

            # 4. Text to Speech
            textToSpeech(law_response)

            # 5. Lecture de la réponse
            play_audio("results/testLaw.wav")

    except KeyboardInterrupt:
        print("\nAmadeus s'éteint. Au revoir, Chapeau de paille.")

if __name__ == "__main__":
    # Plus besoin de sudo !
    main()