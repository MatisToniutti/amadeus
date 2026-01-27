from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
import uvicorn

app = FastAPI()

MODEL_ID = "chatterboxTurbo"
model = None

class Query(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global model, processor
    print(f"--- Initialisation du service {MODEL_ID} ---")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    print(f"--- Modèle {MODEL_ID} chargé et prêt ! ---")

@app.post("/textToSpeech")
async def textToSpeech(query: Query):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    save_file = "/app/data/results/testLaw.wav"
    AUDIO_PROMPT_PATH="/app/data/samples/shortLawVoice.mp3"
    wav = model.generate(query.text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save(save_file, wav, model.sr)
    return save_file
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)