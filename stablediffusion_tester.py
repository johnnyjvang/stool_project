# pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

# Function to generate and save image
def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Get the current date and time to create a unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Directory to save images
    output_dir = "stablediffusion_test"
    os.makedirs(output_dir, exist_ok=True)

    # Generate the image based on the user input
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

    # Save the image with a unique name
    image_name = f"{output_dir}/stablediffusion_test_{timestamp}.png"
    image.save(image_name)
    print(f"Generated and saved image as: {image_name}")

# Ask for a single prompt from the user
prompt = input("Enter the prompt for generating the image: ")

# Generate and save the image
generate_image(prompt)

# Ask if the user is happy with the image and if they want to refine it
while True:
    refine = input("Is the image good? (yes/no): ").lower()
    if refine == 'yes':
        print("Great! The image is good.")
        break
    else:
        prompt = input("Enter a refined prompt: ")
        generate_image(prompt)
