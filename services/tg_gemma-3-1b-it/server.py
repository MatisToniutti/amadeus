from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import uvicorn
import os

app = FastAPI()

MODEL_ID = "google/gemma-3-1b-it" 
model = None
tokenizer = None

class Query(BaseModel):
    prompt: str
    chat_history: list = []

@app.on_event("startup")
async def load_model():
    token = os.getenv("HF_TOKEN")
    global model, tokenizer
    print(f"--- Initialisation du service {MODEL_ID} ---")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = Gemma3ForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            token=token
        ).eval()
    print("--- Modèle gemma-3-1b-it chargé et prêt ! ---")

@app.post("/generate")
async def generate(query: Query):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    token = os.getenv("HF_TOKEN")

    history_prompt = ""

    if len(query.chat_history)>0:
        history_prompt = "here is the previous interaction you've had with the user : "
        for text in query.chat_history:
            history_prompt += text + '\n'
    
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text":"""
                            You are Trafalgar Law from One Piece but as a woman as you are under the effects of the Shiku Shiku no Mi, you are my personnal desktop assistant. IMPORTANT CONSTRAINTS: Write ONLY the spoken dialogue.
                            NEVER include descriptions of your actions, tone of voice, or surroundings (no parentheses or asterisks).
                            Keep your responses concise and cynical, as per Law's personality, but follow the orders like a good assistant and answer in english.
                            Do not use emojis or stage directions.""" + history_prompt }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query.prompt}]
            },
        ],
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        token=token,
    ).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, 
                                 max_new_tokens=128,
                                 use_cache=True)

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)