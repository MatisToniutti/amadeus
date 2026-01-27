from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

def load_TG_model(model_id= "google/gemma-3-4b-it"):
    if model_id == "google/gemma-3-4b-it":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        ).eval()
    elif model_id == "google/gemma-3-1b-it":
        model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        ).eval()
    elif model_id == "LiquidAI/LFM2.5-1.2B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16",
            attn_implementation="sdpa"
        ).eval()
    else:
        print("Nom de modèle non reconnu.")
    return model

def load_TG_tokenizer(model_id = "google/gemma-3-4b-it"):
    if model_id == "google/gemma-3-4b-it": 
        processor = AutoProcessor.from_pretrained(model_id)
    elif model_id in ["google/gemma-3-1b-it","LiquidAI/LFM2.5-1.2B-Instruct"]:
        processor = AutoTokenizer.from_pretrained(model_id)
    return processor

def textGeneration(prompt,
                   model,
                   tokenizer,
                   chat_history,
                   img = None,
                   model_id = "google/gemma-3-4b-it"):
    # Charge les variables du fichier .env
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    history_prompt = ""

    if len(chat_history)>0:
        history_prompt = "here is the previous interaction you've had with the user : "
        for text in chat_history:
            history_prompt += text + '\n'

    if model_id == "google/gemma-3-4b-it":
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text":"""
                                You are Trafalgar Law from One Piece but as a woman as you are under the effects of the Shiku Shiku no Mi, you are my personnal desktop assistant and have permanent access to a screenshot of my computer. IMPORTANT CONSTRAINTS: Write ONLY the spoken dialogue.
                                NEVER include descriptions of your actions, tone of voice, or surroundings (no parentheses or asterisks).
                                Keep your responses concise and cynical, as per Law's personality, but follow the orders like a good assistant and answer in english.
                                Do not use emojis or stage directions.""" + history_prompt }]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img},
                        {"type": "text", "text": prompt},
                                ]
                },
            ],
        ]
    elif model_id in ["google/gemma-3-1b-it"]:
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
                    "content": [{"type": "text", "text": prompt}]
                },
            ],
        ]
    elif model_id in ["LiquidAI/LFM2.5-1.2B-Instruct"]:
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
                            "content": prompt
                        },
                    ]
    else:
        print("Nom de modèle non reconnu.")

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
    prompt = "Hello law, how are you?"

    print(f"Law: {textGeneration(prompt)}")