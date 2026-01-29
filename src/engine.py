import torch
import time
from src.utils import take_screenshot, record_audio, play_audio
import subprocess
import requests

class Engine:
    def __init__(self):
        self.models = {}
        self.chat_history = []
        self.is_ready = False
        self.volume = 100
        self.current_models = {
            "tg":"lfm2.5-1.2b-instruct"
            }
        self.available_models = {
            "tg": ["gemma-3-4b-it","gemma-3-1b-it","lfm2.5-1.2b-instruct"]
        }

    def start_base_services(self):
        """charge tous les modèles de base au démarrage"""
        print("Chargement des modèles")

        self.models["vad_model"], _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        print("Voice detection chargé")

        """Lance Whisper et le TTS"""
        print("Démarrage des services de base...")
        subprocess.run(["docker", "compose", "--profile", "base", "--profile", "lfm2.5-1.2b-instruct", "up", "-d"])

        self.is_ready = True
        print("Système Amadeus activé.")
        print("Appuyez sur Ctrl+C pour arrêter.")

    def switch_model(self, new_model, type="tg"):
        """
        Change le modèle LLM actif.
        new_model: correspond au nom du profil dans le yaml
        """
        print(f"🛑 Arrêt de l'ancien modèle...")
        subprocess.run(["docker", "stop", self.current_models[type]])
        new_model = new_model.strip()

        print(f"🚀 Démarrage du modèle : {new_model}")
        cmd = [
            "docker", "compose",
            "--profile", "base",
            "--profile", new_model,
            "up", "-d", "--remove-orphans"
        ]
        subprocess.run(cmd)
        
        # On attend que le service réponde (Healthcheck basique)
        type_to_port = {
            "tg":8001,
            "tts":8004,
            "stt":8003
        }
        self.wait_for_service(port=type_to_port[type])

        self.current_models[type] = new_model

    def wait_for_service(self, port):
        """Boucle d'attente jusqu'à ce que le conteneur soit prêt"""
        url = f"http://localhost:{port}/docs" # Endpoint FastAPI par défaut
        for _ in range(30): # Essai pendant 30s
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"✅ Service sur le port {port} est prêt !")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        print("⚠️ Le service met du temps à démarrer...")

    def run_pipeline(self, ui_callback):
        """Réalise une boucle complète d'interaction"""
        if not self.is_ready:
            print("Erreur : modèles non chargés")
            return
        
        try:
            screenshot = take_screenshot() if self.current_models["tg"] == "google/gemma-3-4b-it" else None

            # 1. Écoute
            ui_callback("Écoute...", "red")
            audio_path = record_audio(vad_model=self.models["vad_model"])
            if not audio_path: # Si VAD n'a rien entendu ou arrêt
                ui_callback("Prêt", "green", "")
                return

            # 2. Speech to Text
            start_stt = time.perf_counter()
            ui_callback("Transcription...", "orange")
            user_text = self.query_stt(audio_path = audio_path)
            end_stt = time.perf_counter()
            
            print(f"Toi : {user_text} ({end_stt - start_stt:.2f}s)\n")
            

            # 3. Text Generation
            start_llm = time.perf_counter()
            ui_callback("Génération du texte...", "orange")
            law_response = self.query_llm(
                                    prompt=user_text,  
                                    history = self.chat_history, 
                                    )
            ui_callback("Génération de l'audio...", "orange", law_response)
            end_llm = time.perf_counter()
            
            print(f"Law : {law_response} ({end_llm - start_llm:.2f}s)\n")

            # 4. Text to Speech
            start_tts = time.perf_counter()
            self.query_tts(text=law_response)
            end_tts = time.perf_counter()
            
            print(f"Génération Audio : {end_tts - start_tts:.2f}s\n")

            # 5. Lecture de la réponse
            ui_callback("Law parle", "blue")
            play_audio("data/results/testLaw.wav", volume=self.volume)
            
            # Total
            total_latency = (end_stt - start_stt) + (end_llm - start_llm) + (end_tts - start_tts)
            print(f"Temps de réponse total de Law : {total_latency:.2f}s\n\n")

            self.chat_history.append("User : "+user_text)
            self.chat_history.append("You (Law) : "+law_response)
            ui_callback("Prêt", "green")

        except KeyboardInterrupt:
            print("\nAmadeus s'éteint")

    def query_llm(self, prompt, history, model_port=8001):
        url = f"http://localhost:{model_port}/generate"
        payload = {
            "prompt": prompt,
            "chat_history": history
        }
        
        try:
            response = requests.post(url, json=payload)
            #raise une erreur si le status correspond à une erreur
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return f"Erreur de communication avec le cerveau : {e}"
        
    def query_stt(self, audio_path, model_port=8003):
        url = f"http://localhost:{model_port}/speechToText"
        payload = {
            "audio": audio_path
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()["text"]
        except Exception as e:
            return f"Erreur de communication avec le cerveau : {e}"
        
    def query_tts(self, text, model_port=8004):
        url = f"http://localhost:{model_port}/textToSpeech"
        payload = {
            "text": text
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["text"]
        except Exception as e:
            return f"Erreur de communication avec le cerveau : {e}"

    def set_volume(self, volume):
        self.volume = volume

    def reset_history(self):
        self.chat_history = []