from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import uvicorn
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

app = FastAPI()

MODEL_ID = "Qwen3-TTS-12Hz-0.6B-Base"
model = None

class Query(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global model
    print(f"--- Initialisation du service {MODEL_ID} ---")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"--- Modèle {MODEL_ID} chargé et prêt ! ---")

@app.post("/textToSpeech")
async def textToSpeech(query: Query):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    save_file = "/app/data/results/testLaw.wav"
    AUDIO_PROMPT_PATH="/app/data/samples/6sLawVoice.mp3"
    ref_text  = "あれは俺の気まぐれだ つまらねえ冗談はやめろ、ドラミンゴ。お前と俺が組めば。"
    wavs, sr = model.generate_voice_clone(
        text=query.text,
        language="Japanese",
        ref_audio=AUDIO_PROMPT_PATH,
        ref_text=ref_text,
    )
    sf.write(save_file, wavs[0], sr)
    return save_file
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)