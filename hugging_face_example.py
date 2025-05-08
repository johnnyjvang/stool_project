from diffusers import StableDiffusionPipeline
import torch

# Login with your Hugging Face token if required
# from huggingface_hub import login
# login(token="your_token_here")

# Load pre-trained model (use 'runwayml/stable-diffusion-v1-5' for cartoon-like generations)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Poop type descriptions based on the Bristol Stool Chart
stool_types = {
    1: "separate hard lumps, cartoon poop emoji, brown background, clean vector style",
    2: "sausage-shaped but lumpy, cartoon poop emoji, brown background, clean vector style",
    3: "like a sausage with cracks, cartoon poop emoji, brown background, clean vector style",
    4: "like a smooth soft sausage, cartoon poop emoji, brown background, clean vector style",
    5: "soft blobs with clear edges, cartoon poop emoji, brown background, clean vector style",
    6: "fluffy pieces with ragged edges, cartoon poop emoji, brown background, clean vector style",
    7: "watery, no solid pieces, cartoon poop emoji, brown background, clean vector style"
}

# Generate and save images
for i, prompt in stool_types.items():
    print(f"Generating image for Stool Type {i}...")
    image = pipe(prompt).images[0]
    image.save(f"stool_type_{i}.png")

print("Done! Images saved as stool_type_1.png to stool_type_7.png.")
