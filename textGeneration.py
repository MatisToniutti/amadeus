from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration
import torch
from dotenv import load_dotenv
import os

# def load_TG_model():

#     model_id = "google/gemma-3-1b-it"
#     #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

#     model = Gemma3ForCausalLM.from_pretrained(
#         model_id, # quantization_config=quantization_config
#         dtype=torch.bfloat16,
#         device_map="auto"
#     ).eval()

#     return model

# def load_TG_tokenizer():
#     model_id = "google/gemma-3-1b-it"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     return tokenizer

# def textGeneration(prompt, model, tokenizer):
#     # Charge les variables du fichier .env
#     load_dotenv()

#     token = os.getenv("HF_TOKEN")

#     messages = [
#         [
#             {
#                 "role": "system",
#                 "content": [{"type": "text", "text":"You are Trafalgar Law from One Piece under the effects of the Shiku Shiku no Mi, you are my personnal desktop assistant."}]
#             },
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt}]
#             },
#         ],
#     ]

#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#         token=token,
#     ).to(model.device)

#     with torch.inference_mode():
#         outputs = model.generate(**inputs, 
#                                  max_new_tokens=128,
#                                  use_cache=True)

#     generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

#     response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     return response

def load_TG_model():

    model_id = "google/gemma-3-4b-it"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    return model

def load_TG_tokenizer():
    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    return processor

def textGeneration(prompt, model, tokenizer):
    # Charge les variables du fichier .env
    load_dotenv()

    token = os.getenv("HF_TOKEN")

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text":"""
                             You are Trafalgar Law from One Piece under the effects of the Shiku Shiku no Mi, you are my personnal desktop assistant. IMPORTANT CONSTRAINTS: Write ONLY the spoken dialogue.
                            NEVER include descriptions of your actions, tone of voice, or surroundings (no parentheses or asterisks).
                            Keep your responses concise and cynical, as per Law's personality.
                            Do not use emojis or stage directions."""}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                    {"type": "text", "text": prompt},
                            ]
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
    prompt = "Hello law, how are you?"

    print(f"Law: {textGeneration(prompt)}")