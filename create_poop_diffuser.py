# pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline
import torch
import os


'''
stool_type_1.png
stool_type_2.png
...
stool_type_7.png
'''

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Prompts based on Bristol Stool Types
prompts = {
    1: "cartoon poop emoji shaped like hard, separate small lumps, dry, rabbit poop",
    2: "cartoon poop emoji shaped like a lumpy sausage, thick with hard bumps",
    3: "cartoon poop emoji shaped like a sausage with surface cracks",
    4: "cartoon poop emoji shaped like a smooth, soft snake-like log",
    5: "cartoon poop emoji as soft blobs with clear edges",
    6: "cartoon poop emoji as fluffy mushy pieces with ragged edges",
    7: "cartoon poop emoji that looks like watery diarrhea with no solid parts"
}

output_dir = "stool_emoji_types"
os.makedirs(output_dir, exist_ok=True)

# Generate and save each image
for i in range(1, 8):
    prompt = prompts[i]
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    image.save(f"{output_dir}/stool_type_{i}.png")
    print(f"Generated: stool_type_{i}.png")



