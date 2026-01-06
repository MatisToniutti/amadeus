import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speechToText(audio):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio, batch_size=2)
    return result

if __name__ == "__main__":
    sampleEn = "./samples/en.mp3"
    sampleFr = "./samples/fr.mp3"
    law = "./samples/shortLawVoice.mp3"
    print(speechToText(law))