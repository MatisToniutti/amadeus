
# Amadeus: Multimodal Desktop Assistant

Amadeus est un assistant personnel local combinant vision, audition et parole. Le projet repose sur l'intégration de modèles que j'essaie d'optimiser pour fonctionner simultanément localement sur une seule interface graphique.

🚀 **Caractéristiques**
* Pipeline Multimodal : Intégration complète de la transcription (STT), de la réflexion (LLM/Vision) et de la synthèse vocale (TTS).

* Vision-as-Context : Capture d'écran automatique à chaque interaction pour permettre à l'IA d'analyser l'activité de l'utilisateur.

* Interaction Push to talk : Appuie d'un bouton global afin de débuter l'enregistrement, puis détection d'activité vocale (VAD) afin de stopper l'enregistrement.

* Interface CustomTkinter : GUI incluant un monitoring en temps réel de la VRAM/RAM ainsi que la réponse et l'état de l'assistant.

* Optimisation VRAM : Utilisation de la quantification pour permettre la cohabitation de plusieurs modèles lourds sur un GPU de 16 Go.


🛠️ **Stack Technique**
* Intelligence Artificielle : Gemma 3 4B it(Vision-Language), Faster-Whisper (STT), Silero (VAD), Chatterbox-Turbo (TTS).

* Interface : CustomTkinter (Python).

* Traitement de données : PyTorch, PIL (Image processing), SoundDevice (Audio).

* Hardware : Optimisé pour NVIDIA (CUDA).


📦 **Installation & Setup**

Pas prévus pour l'instant