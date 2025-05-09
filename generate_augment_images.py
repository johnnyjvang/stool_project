import os
# OpenCV for reading/writing images and color space conversion
import cv2             
import numpy as np        
# image augmentation - github repository
from imgaug import augmenters as iaa

# For setting seed and using imgaug's core functionality 
import imgaug as ia             

# Set a random seed for reproducibility.
# This ensures that augmentations (random rotations, flips, etc.) are the same each run,
# which is useful for debugging and consistency across training experiments.
ia.seed(42)

# Define the input and output directories
input_folder = "imgaug_test/"     # Folder with original images (e.g., poop in toilet)
output_folder = "imgaug_ugmented/"   # Folder to save augmented images
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Define a sequence of augmentations to apply using imgaug
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # Apply horizontal flip with 50% probability
    iaa.Flipud(0.2),  # Apply vertical flip with 20% probability (less common in natural settings)

    iaa.Affine(       # Apply affine transformations
        rotate=(-20, 20),  # Random rotation between -20 and +20 degrees
        scale=(0.9, 1.1),  # Zoom in or out between 90% and 110%
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}  # Shift up to ±10% in x and y directions
    ),

    iaa.AddToBrightness((-30, 30)),       # Adjust brightness by adding a value between -30 to +30 to pixel intensities
    iaa.AddToHueAndSaturation((-20, 20)), # Modify hue/saturation to simulate color variance
    iaa.GaussianBlur(sigma=(0.0, 1.0)),   # Apply Gaussian blur with a sigma randomly between 0.0 and 1.0
    iaa.LinearContrast((0.8, 1.2)),       # Adjust contrast linearly, 0.8 = lower contrast, 1.2 = higher
])

# Number of augmentations to generate per original image
N_AUG = 10

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # Skip non-image files

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)  # Load the image using OpenCV (BGR format)

    if image is None:
        print(f"Skipping unreadable file: {filename}")
        continue  # Skip unreadable or corrupt images

    # Convert from BGR (OpenCV default) to RGB (what imgaug expects)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a list of the same image repeated N_AUG times and apply the augmentation pipeline
    images_aug = augmenters(images=[image_rgb] * N_AUG)

    # Save each augmented image
    base_name = os.path.splitext(filename)[0]  # Strip the file extension
    for i, aug_img in enumerate(images_aug):
        # Convert back to BGR for OpenCV saving
        aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

        # Construct filename and save
        out_name = f"{base_name}_aug{i+1}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), aug_bgr)

print("Augmentation complete.")
