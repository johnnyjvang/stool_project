from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from datetime import datetime

# Ask for a single prompt
prompt = input("Enter the prompt for generating the image: ")

'''
poop 1 : Medical illustration of a cartoon poop emoji that looks like hard, separate small lumps, resembling dry rabbit droppings. Represents Type 1 stool on the Bristol Stool Chart. Indicates constipation, caused by slow digestion, lack of fiber, dehydration, or inactivity. Stylized but realistic texture, neutral background, clear anatomical context.
'''

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

# Load and validate the initial image
image_path = "/home/jvang/Desktop/stool_project/Icon_Bird_512x512.png"
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

initial_image = Image.open(image_path).convert("RGB")
initial_image = initial_image.resize((512, 512))

# Debugging check
print(f"Image loaded: type={type(initial_image)}, mode={initial_image.mode}, size={initial_image.size}")

# Confirm it's a proper PIL Image
if not isinstance(initial_image, Image.Image):
    raise TypeError("The loaded image is not a PIL.Image.Image")

# Generate the new image
image = pipe(prompt=prompt, init_image=initial_image, strength=0.75, num_inference_steps=25).images[0]

# Save the image
image_name = f"{output_dir}/stablediffusion_test_{timestamp}.png"
image.save(image_name)
print(f"Generated and saved image as: {image_name}")
