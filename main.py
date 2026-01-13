import torch
from src.modules.speechToText import speechToText, load_STT_model, load_STT_processor
from src.modules.textGeneration import textGeneration, load_TG_model, load_TG_tokenizer
from src.modules.textToSpeech import textToSpeech, load_TTS_model
import time
import customtkinter as ctk
import threading
import psutil
import pynvml
from src.utils import take_screenshot, record_audio, play_audio


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
        self.indicator = ctk.CTkFrame(self, width=20, height=20, fg_color="green")
        self.indicator.pack(pady=10)

        #la fenêtre reste devant
        self.attributes('-topmost', True)

        # Bind de la touche Espace
        self.bind('<space>', lambda event: self.start_conversation_thread())

        # Initialisation de NVIDIA Management Library
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Ton GPU 0
            self.gpu_ready = True
        except:
            self.gpu_ready = False

        # --- UI ELEMENTS ---
        self.label_title = ctk.CTkLabel(self, text="Système Status", font=("Helvetica", 18, "bold"))
        self.label_title.pack(pady=10)

        self.answer_label = ctk.CTkLabel(self, text="", font=("Helvetica", 16))
        self.answer_label.pack(pady=10)

        # RAM Système
        self.ram_label = ctk.CTkLabel(self, text="RAM: 0%")
        self.ram_label.pack()
        self.ram_bar = ctk.CTkProgressBar(self, width=300)
        self.ram_bar.pack(pady=5)

        # VRAM (GPU)
        self.vram_label = ctk.CTkLabel(self, text="VRAM: 0MB / 0MB")
        self.vram_label.pack()
        self.vram_bar = ctk.CTkProgressBar(self, width=300)
        self.vram_bar.pack(pady=5)

        # Lancer la boucle de mise à jour
        self.update_stats()

    def update_stats(self):
        # 1. RAM Système
        ram = psutil.virtual_memory()
        self.ram_label.configure(text=f"RAM Système: {ram.percent}%")
        self.ram_bar.set(ram.percent / 100)

        # 2. VRAM NVIDIA
        if self.gpu_ready:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_gb = info.used / 1024**3
            total_gb = info.total / 1024**3
            percent = (info.used / info.total)
            
            self.vram_label.configure(text=f"VRAM: {used_gb:.2f}GB / {total_gb:.2f}GB")
            self.vram_bar.set(percent)
            
            # Alerte couleur si > 90%
            if percent > 0.9:
                self.vram_bar.configure(progress_color="red")
            else:
                self.vram_bar.configure(progress_color="#1f538d")

        # Rappeler cette fonction dans 1000ms (1 seconde)
        self.after(1000, self.update_stats)

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
            audio_path = record_audio(vad_model=vad_model)
            

            # 2. Speech to Text
            start_stt = time.perf_counter()
            self.update_status("Réflexion...", "orange")
            user_text = speechToText(audio = audio_path, model = STT_model, processor = STT_processor)
            end_stt = time.perf_counter()
            
            print(f"Toi : {user_text} ({end_stt - start_stt:.2f}s)\n")
            

            # 3. Text Generation
            start_llm = time.perf_counter()
            law_response = textGeneration(user_text, model = TG_model, tokenizer= TG_tokenizer, chat_history=chat_history, img = screenshot)
            self.answer_label.configure(text=law_response)
            end_llm = time.perf_counter()
            
            print(f"Law : {law_response} ({end_llm - start_llm:.2f}s)\n")

            # 4. Text to Speech
            start_tts = time.perf_counter()
            textToSpeech(law_response, TTS_model)
            end_tts = time.perf_counter()
            
            print(f"Génération Audio : {end_tts - start_tts:.2f}s\n")

            # 5. Lecture de la réponse
            self.update_status("Law parle", "blue")
            play_audio("data/results/testLaw.wav")
            
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