import cv2
import numpy as np

def classify_stool(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_parts = len(contours)  # How many separate pieces
    aspect_ratios = []
    areas = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratios.append(w / h)
        areas.append(cv2.contourArea(cnt))

    avg_aspect_ratio = np.mean(aspect_ratios)
    avg_area = np.mean(areas)

    # Simple Rules (based on common sense, will need tuning)
    if num_parts > 5 and avg_area < 500:
        return "Type 1: Separate hard lumps"
    elif num_parts == 1 and avg_aspect_ratio < 2 and avg_area > 1000:
        return "Type 2: Sausage-shaped but lumpy"
    elif num_parts == 1 and avg_aspect_ratio > 1.5 and avg_area > 1000:
        return "Type 3: Sausage with cracks"
    elif num_parts == 1 and avg_aspect_ratio > 3:
        return "Type 4: Smooth sausage/snake"
    elif num_parts > 3 and avg_area < 800:
        return "Type 5: Soft blobs"
    elif num_parts > 5 and avg_area > 300:
        return "Type 6: Mushy stool"
    elif avg_area < 300:
        return "Type 7: Watery, no solid pieces"
    else:
        return "Unclassified"

# Example usage:
print(classify_stool('example_stool.png'))
