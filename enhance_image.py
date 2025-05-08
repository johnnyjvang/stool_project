import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

def enhance_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Histogram Equalization in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    image_enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return image_enhanced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance an image using histogram equalization.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    enhanced_image = enhance_image(args.image_path)

    # Show the result
    plt.imshow(enhanced_image)
    plt.title("Enhanced Image")
    plt.axis("off")
    plt.show()

    #python3 enhance_image.py poop.jpg

