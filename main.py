import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import librosa
from speechToText import speechToText, load_STT_model, load_STT_processor
from textGeneration import textGeneration, load_TG_model, load_TG_tokenizer
from textToSpeech import textToSpeech, load_TTS_model
import time
import mss
from PIL import Image
import customtkinter as ctk
import threading

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
    target_fs = 16000
    filename = "results/input_user.wav"
    
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
    with mss.mss() as sct:
        # Capture l'écran principal
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        
        # Convertir en image PIL
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # essayer de resize /2 voir si ça va plus vite et c'est ok
        img.thumbnail((896, 896))
                
        img.save("results/last_screenshot.jpg")
        return "results/last_screenshot.jpg"

# def main():
#     chat_history = []

#     print("chargement des modèles")

#     STT_model = load_STT_model()
#     STT_processor = load_STT_processor()
#     print("STT chargé")

#     TG_model = load_TG_model()
#     TG_tokenizer = load_TG_tokenizer()
#     print("TG chargé")

#     TTS_model = load_TTS_model()
#     print("TTS chargé")

#     print("Système Amadeus activé.")
#     print("Appuyez sur Ctrl+C pour arrêter.")

#     try:
#         while True:
#             # capture d'écran
#             screenshot = take_screenshot()

#             # 1. Écoute
#             audio_path = record_audio()

#             # 2. Speech to Text
#             start_stt = time.perf_counter()
#             user_text = speechToText(audio = audio_path, model = STT_model, processor = STT_processor)
#             end_stt = time.perf_counter()
            
#             print(f"Toi : {user_text} ({end_stt - start_stt:.2f}s)\n")
            
#             if user_text.strip() == "": break

#             # 3. Text Generation
#             start_llm = time.perf_counter()
#             law_response = textGeneration(user_text, model = TG_model, tokenizer= TG_tokenizer, chat_history=chat_history, img = screenshot)
#             end_llm = time.perf_counter()
            
#             print(f"Law : {law_response} ({end_llm - start_llm:.2f}s)\n")

#             # 4. Text to Speech
#             start_tts = time.perf_counter()
#             textToSpeech(law_response, TTS_model)
#             end_tts = time.perf_counter()
            
#             print(f"Génération Audio : {end_tts - start_tts:.2f}s\n")

#             # 5. Lecture de la réponse
#             play_audio("results/testLaw.wav")
            
#             # Total
#             total_latency = (end_stt - start_stt) + (end_llm - start_llm) + (end_tts - start_tts)
#             print(f"Temps de réponse total de Law : {total_latency:.2f}s\n\n")

#             chat_history.append("User : "+user_text)
#             chat_history.append("You (Law) : "+law_response)

#     except KeyboardInterrupt:
#         print("\nAmadeus s'éteint")

class LawAssistantGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Amadeus - Law Desktop Pet")
        self.geometry("400x600")
        
        # --- État du Bot ---
        self.status = "Prêt" # "Écoute", "Réflexion", "Parle"
        
        # --- Interface ---
        self.label_status = ctk.CTkLabel(self, text=f"État : {self.status}", font=("Helvetica", 18))
        self.label_status.pack(pady=20)

        # Bouton Principal (On l'utilisera pour déclencher la boucle)
        self.btn_talk = ctk.CTkButton(self, text="PARLER (ESPACE)", command=self.start_conversation_thread)
        self.btn_talk.pack(pady=20)

        # Indicateur visuel (Le futur logo)
        self.indicator = ctk.CTkFrame(self, width=50, height=50, fg_color="green")
        self.indicator.pack(pady=10)

        # Bind de la touche Espace
        self.bind('<space>', lambda event: self.start_conversation_thread())

    def update_status(self, new_status, color):
        self.status = new_status
        self.label_status.configure(text=f"État : {self.status}")
        self.indicator.configure(fg_color=color)

    def start_conversation_thread(self):
        # On lance la logique IA dans un thread séparé pour ne pas bloquer la fenêtre
        thread = threading.Thread(target=self.run_logic)
        thread.start()

    def run_logic(self):
        try:
            # capture d'écran
            screenshot = take_screenshot()

            # 1. Écoute
            self.update_status("Écoute...", "red")
            audio_path = record_audio()
            

            # 2. Speech to Text
            start_stt = time.perf_counter()
            self.update_status("Réflexion...", "orange")
            user_text = speechToText(audio = audio_path, model = STT_model, processor = STT_processor)
            end_stt = time.perf_counter()
            
            print(f"Toi : {user_text} ({end_stt - start_stt:.2f}s)\n")
            

            # 3. Text Generation
            start_llm = time.perf_counter()
            law_response = textGeneration(user_text, model = TG_model, tokenizer= TG_tokenizer, chat_history=chat_history, img = screenshot)
            end_llm = time.perf_counter()
            
            print(f"Law : {law_response} ({end_llm - start_llm:.2f}s)\n")

            # 4. Text to Speech
            start_tts = time.perf_counter()
            textToSpeech(law_response, TTS_model)
            end_tts = time.perf_counter()
            
            print(f"Génération Audio : {end_tts - start_tts:.2f}s\n")

            # 5. Lecture de la réponse
            self.update_status("Law parle", "blue")
            play_audio("results/testLaw.wav")
            
            # Total
            total_latency = (end_stt - start_stt) + (end_llm - start_llm) + (end_tts - start_tts)
            print(f"Temps de réponse total de Law : {total_latency:.2f}s\n\n")

            chat_history.append("User : "+user_text)
            chat_history.append("You (Law) : "+law_response)
            self.update_status("Prêt", "green")

        except KeyboardInterrupt:
            print("\nAmadeus s'éteint")
        

if __name__ == "__main__":
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    #main()
    chat_history = []
    print("chargement des modèles")

    STT_model = load_STT_model()
    STT_processor = load_STT_processor()
    print("STT chargé")

    TG_model = load_TG_model()
    TG_tokenizer = load_TG_tokenizer()
    print("TG chargé")

    TTS_model = load_TTS_model()
    print("TTS chargé")

    print("Système Amadeus activé.")
    print("Appuyez sur Ctrl+C pour arrêter.")
    app = LawAssistantGUI()
    app.mainloop()