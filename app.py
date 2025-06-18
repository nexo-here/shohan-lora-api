import os
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import requests

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

# Load the LoRA model once
model_id = "stabilityai/stable-diffusion-2-1"
lora_path = "https://huggingface.co/nexo-here/shohan-lora/resolve/main/shohan-lora.safetensors"
trigger_word = "Shohan"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.load_lora_weights(lora_path)

@app.post("/generate")
async def generate(input: PromptInput):
    full_prompt = f"{trigger_word}, {input.prompt}"
    image = pipe(full_prompt).images[0]

    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")
    
