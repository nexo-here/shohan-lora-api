from fastapi import FastAPI, Query
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

# ✅ STEP 1: Load base model (2.1 works best with LoRA)
base_model = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
).to("cuda")

# ✅ STEP 2: Load LoRA weights from HuggingFace
lora_model = "nexo-here/shohan-lora"
lora_file = "shohan-lora.safetensors"
trigger = "Shohan"

print("🔁 Downloading LoRA file...")
lora_path = hf_hub_download(repo_id=lora_model, filename=lora_file)
pipe.load_lora_weights(lora_path)
print("✅ LoRA loaded.")

@app.get("/gen")
def generate(prompt: str = Query(...)):
    # Add trigger word automatically
    full_prompt = f"{trigger}, {prompt}"

    # Generate image
    image = pipe(full_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # Convert to stream
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
  
