from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uvicorn
import os

app = FastAPI()

MODEL_ID = "openai/whisper-large-v3-turbo"
model = None
processor = None

class Query(BaseModel):
    audio: str

@app.on_event("startup")
async def load_model():
    global model, processor
    print(f"--- Initialisation du service {MODEL_ID} ---")
        
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa"
    )
    print(f"--- Modèle {MODEL_ID} chargé et prêt ! ---")

@app.post("/speechToText")
async def speechToText(query: Query):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Gestion du chemin du fichier (Volume Docker)
    cleaned_path = query.audio.replace("data/", "").lstrip("/")
    internal_path = os.path.join("/app/data", cleaned_path)
    #internal_path = "app/data/results/input_user.wav"
    if not os.path.exists(internal_path):
        raise HTTPException(status_code=404, detail=f"Audio introuvable ici : {internal_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer, 
        feature_extractor=processor.feature_extractor, 
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30
    )

    # return_timestamps=True aide parfois à stabiliser la sortie
    result = pipe(internal_path, batch_size=2, return_timestamps=True)
    
    return {"text": result["text"]} 
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)