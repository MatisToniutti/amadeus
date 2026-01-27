import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# def load_TTS_model():
#     model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
#     return model

# def textToSpeech(text, model):
#     save_file = "data/results/testLaw.wav"
#     AUDIO_PROMPT_PATH="data/samples/shortLawVoice.mp3"
#     wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH, language_id = "ja")
#     ta.save(save_file, wav, model.sr)
#     return save_file

def load_TTS_model():
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    return model

def textToSpeech(text, model):
    save_file = "data/results/testLaw.wav"
    AUDIO_PROMPT_PATH="data/samples/shortLawVoice.mp3"
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save(save_file, wav, model.sr)
    return save_file

# def load_TTS_model():
#     model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
#     return model

# def textToSpeech(text, model):
#     save_file = "data/results/testLaw.wav"
#     AUDIO_PROMPT_PATH="data/samples/shortLawVoice.mp3"
#     model.tts_to_file(text=text,
#                 file_path=save_file,
#                 speaker_wav=AUDIO_PROMPT_PATH,
#                 language="en")
#     return save_file

if __name__ == "__main__":
    text = "U-uh… hello. I’m… fine, I guess? It’s… it’s been a bit… busy. You know, with the… the *thing*."
    textToSpeech(text)