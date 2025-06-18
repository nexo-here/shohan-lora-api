import torch
import os
import requests
from uuid import uuid4
from PIL import Image
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Ensure static directory exists
os.makedirs("/tmp/static", exist_ok=True)

# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="/tmp/static"), name="static")

# LoRA model download info
LORA_URL = "https://huggingface.co/nexo-here/shohan-lora/resolve/main/shohan-lora.safetensors"
LORA_PATH = "shohan-lora.safetensors"

# Download LoRA model if not already present
if not os.path.exists(LORA_PATH):
    print("🔽 Downloading LoRA file...")
    r = requests.get(LORA_URL)
    with open(LORA_PATH, "wb") as f:
        f.write(r.content)
    print("✅ LoRA downloaded.")

# Load base model and apply LoRA
print("🚀 Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora()
print("✅ LoRA loaded and fused.")

# Endpoint for image generation
@app.get("/gen")
async def generate_image(prompt: str = Query(..., description="Enter your prompt")):
    final_prompt = f"Shohan {prompt}"
    image = pipe(final_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    filename = f"{uuid4().hex}.png"
    filepath = f"/tmp/static/{filename}"
    image.save(filepath)

    return {
        "prompt": prompt,
        "image_url": f"https://shohan-lora-api.onrender.com/static/{filename}"
    }
