import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

def enhance_image_clahe(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge the enhanced L with original a and b, then convert back to RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return image_clahe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance an image using CLAHE.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    enhanced_image = enhance_image_clahe(args.image_path)

    # Display the result
    plt.imshow(enhanced_image)
    plt.title("CLAHE Enhanced Image")
    plt.axis("off")
    plt.show()
