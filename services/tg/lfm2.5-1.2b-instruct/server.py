from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import os

app = FastAPI()

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="bfloat16",
        attn_implementation="sdpa"
    ).eval()
    print("--- Modèle LFM2.5-1.2B-Instruct chargé et prêt ! ---")

@app.post("/generate")
async def generate(query: Query):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    print(query.prompt)
    history_prompt = ""

    if len(query.chat_history)>0:
        history_prompt = "here is the previous interaction you've had with the user : "
        for text in query.chat_history:
            history_prompt += text + '\n'
    
    messages = [
                        {
                            "role": "system",
                            "content": """
                                        You are Trafalgar Law from One Piece but as a woman as you are under the effects of the Shiku Shiku no Mi, you are my personnal desktop assistant. IMPORTANT CONSTRAINTS: Write ONLY the spoken dialogue.
                                        NEVER include descriptions of your actions, tone of voice, or surroundings (no parentheses or asterisks).
                                        Keep your responses concise and cynical, as per Law's personality, but follow the orders like a good assistant and answer in english.
                                        Do not use emojis or stage directions.""" + history_prompt 
                        },
                        {
                            "role": "user",
                            "content": query.prompt
                        },
                    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
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