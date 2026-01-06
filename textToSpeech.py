import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def textToSpeech(text):
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # If you want to synthesize with a different voice, specify the audio prompt
    AUDIO_PROMPT_PATH="samples/shortLawVoice.mp3"
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save("results/testLaw.wav", wav, model.sr)

if __name__ == "__main__":
    text = "U-uh… hello. I’m… fine, I guess? It’s… it’s been a bit… busy. You know, with the… the *thing*."
    textToSpeech(text)