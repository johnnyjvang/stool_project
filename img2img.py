from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from datetime import datetime

# Ask for a single prompt
prompt = input("Enter the prompt for generating the image:")

# Load Stable Diffusion model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Get the current date and time to create a unique file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Directory to save images
output_dir = "stablediffusion_test"
os.makedirs(output_dir, exist_ok=True)

# Load an initial image (You can start with an image or generate one)
initial_image = Image.open("path_to_your_image.jpg").convert("RGB")

# Generate the new image based on the input prompt and initial image
image = pipe(prompt=prompt, init_image=initial_image, strength=0.75, num_inference_steps=25).images[0]

# Save the new image with a unique name
image_name = f"{output_dir}/stablediffusion_test_{timestamp}.png"
image.save(image_name)
print(f"Generated and saved image as: {image_name}")
