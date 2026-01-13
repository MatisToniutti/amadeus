import torch
import time
from src.modules.speechToText import speechToText, load_STT_model, load_STT_processor
from src.modules.textGeneration import textGeneration, load_TG_model, load_TG_tokenizer
from src.modules.textToSpeech import textToSpeech, load_TTS_model
from src.utils import take_screenshot, record_audio, play_audio

class Engine:
    def __init__(self):
        self.models = {}
        self.chat_history = []
        self.is_ready = False

    def load_all_models(self):
        """charge tous les modèles de base au démarrage"""
        print("Chargement des modèles")

        self.models["vad_model"], _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        print("Voice detection chargé")

        self.models["stt_model"] = load_STT_model()
        self.models["stt_processor"] = load_STT_processor()
        print("STT chargé")

        self.models["tg_model"] = load_TG_model()
        self.models["tg_tokenizer"] = load_TG_tokenizer()
        print("TG chargé")

        self.models["tts_model"] = load_TTS_model()
        print("TTS chargé")

        self.is_ready = True
        print("Système Amadeus activé.")
        print("Appuyez sur Ctrl+C pour arrêter.")

    def run_pipeline(self, ui_callback):
        """Réalise une boucle complète d'interaction"""
        if not self.is_ready:
            print("Erreur : modèles non chargés")
            return
        
        try:
            screenshot = take_screenshot()

            # 1. Écoute
            ui_callback("Écoute...", "red")
            audio_path = record_audio(vad_model=self.models["vad_model"])
            if not audio_path: # Si VAD n'a rien entendu ou arrêt
                ui_callback("Prêt", "green", "")
                return

            # 2. Speech to Text
            start_stt = time.perf_counter()
            ui_callback("Transcription...", "orange")
            user_text = speechToText(audio = audio_path, 
                                     model = self.models["stt_model"],
                                     processor = self.models["stt_processor"])
            end_stt = time.perf_counter()
            
            print(f"Toi : {user_text} ({end_stt - start_stt:.2f}s)\n")
            

            # 3. Text Generation
            start_llm = time.perf_counter()
            ui_callback("Génération du texte...", "orange")
            law_response = textGeneration(user_text, 
                                          model = self.models["tg_model"], 
                                          tokenizer =  self.models["tg_tokenizer"], 
                                          chat_history = self.chat_history, 
                                          img = screenshot)
            ui_callback("Génération de l'audio...", "orange", law_response)
            end_llm = time.perf_counter()
            
            print(f"Law : {law_response} ({end_llm - start_llm:.2f}s)\n")

            # 4. Text to Speech
            start_tts = time.perf_counter()
            textToSpeech(law_response, self.models["tts_model"])
            end_tts = time.perf_counter()
            
            print(f"Génération Audio : {end_tts - start_tts:.2f}s\n")

            # 5. Lecture de la réponse
            ui_callback("Law parle", "blue")
            play_audio("data/results/testLaw.wav")
            
            # Total
            total_latency = (end_stt - start_stt) + (end_llm - start_llm) + (end_tts - start_tts)
            print(f"Temps de réponse total de Law : {total_latency:.2f}s\n\n")

            self.chat_history.append("User : "+user_text)
            self.chat_history.append("You (Law) : "+law_response)
            ui_callback("Prêt", "green")

        except KeyboardInterrupt:
            print("\nAmadeus s'éteint")