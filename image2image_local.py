import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Set device and load model
device = "cuda"
model_id_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Path to the local PNG image
local_image_path = "type1.png"  # Replace with your local image path

# Load the local image
init_image = Image.open(local_image_path).convert("RGB")
init_image = init_image.resize((768, 512))  # Resize to match the model input dimensions

# Prompt for generating the new image
prompt = "A simple cartoon-style representation of Type 1 stool, similar to the input image. The stool should remain hard and pellet-like in different sizes and orientations, keeping the same basic shape and design. Ensure the style and simplicity of the original image are preserved with slight variations in scale and angle."

# Generate the image
images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

# Save the output image
images[0].save("type1_simple.png")
