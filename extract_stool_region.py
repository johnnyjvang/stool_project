import cv2
import numpy as np

def extract_stool_region(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold to isolate darker regions (poop is usually darker than water)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # Find largest contour (stool blob)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter small specks
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask
