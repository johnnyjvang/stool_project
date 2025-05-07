import cv2
import numpy as np

def enhance_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Histogram Equalization in LAB color space
    # Method to redistribute pixel intensities so that all intensities are equally distributed
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    image_enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return image_enhanced