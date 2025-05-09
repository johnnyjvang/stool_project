import cv2
import numpy as np
from rembg import remove
from PIL import Image, ImageEnhance

# Load and convert image to RGBA
input_path = 'poop_in_toilet.jpg'
output_path = 'poop_isolated_whitebg.png'

# Step 1: Load image using PIL
original = Image.open(input_path).convert("RGBA")

# Step 2: Remove background using rembg
removed_bg = remove(original)

# Step 3: Convert to numpy array and get alpha mask
rgba = np.array(removed_bg)
alpha = rgba[:, :, 3]

# Optional: clean small noise in the alpha mask
kernel = np.ones((3,3), np.uint8)
cleaned_alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
rgba[:, :, 3] = cleaned_alpha

# Step 4: Place on white background
white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
combined = Image.alpha_composite(white_bg, Image.fromarray(rgba))

# Step 5: Enhance the image (contrast and sharpness)
enhancer_contrast = ImageEnhance.Contrast(combined)
enhancer_sharpness = ImageEnhance.Sharpness(enhancer_contrast.enhance(1.3))  # Increase contrast 30%
final_image = enhancer_sharpness.enhance(1.5)  # Increase sharpness 50%

# Step 6: Save the enhanced output
final_image.convert("RGB").save(output_path)

print(f"Saved isolated and enhanced image to {output_path}")
