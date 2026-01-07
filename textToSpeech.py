import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def load_TTS_model():
    model = ChatterboxTTS.from_pretrained(device="cuda")
    return model

def textToSpeech(text, model):
    save_file = "results/testLaw.wav"
    AUDIO_PROMPT_PATH="samples/shortLawVoice.mp3"
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save(save_file, wav, model.sr)
    return save_file

if __name__ == "__main__":
    text = "U-uh… hello. I’m… fine, I guess? It’s… it’s been a bit… busy. You know, with the… the *thing*."
    textToSpeech(text)