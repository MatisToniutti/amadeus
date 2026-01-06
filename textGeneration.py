from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
from dotenv import load_dotenv
import os



def textGeneration(prompt):
    # Charge les variables du fichier .env
    load_dotenv()

    token = os.getenv("HF_TOKEN")

    model_id = "google/gemma-3-1b-it"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Gemma3ForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text":"You are Trafalgar Law from One Piece under the effects of the Shiku Shiku no Mi, so you must behave like Law but be a little ashamed of your current appearance."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
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
    ).to(model.device).to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128)

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response

if __name__ == "__main__":
    prompt = "Hello law, how are you?"

    print(f"Law: {textGeneration(prompt)}")