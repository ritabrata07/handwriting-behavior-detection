import cv2
import numpy as np
import os

# --- Functions ---

def calculate_slant_angle(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0
    angles = [(theta * 180 / np.pi) - 90 for rho, theta in (line[0] for line in lines)]
    return np.mean(angles) if angles else 0

def calculate_size(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    areas = [cv2.boundingRect(cnt)[3] for cnt in contours]  # height of each contour
    return np.mean(areas)

def calculate_stroke(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return np.mean(thresh) / 255  # stroke thickness

def predict_behavior(slant_angle, avg_size, stroke):
    if slant_angle > 5 and avg_size > 30:
        return "Outgoing/Confident"
    elif slant_angle < -5 and avg_size < 20:
        return "Introverted/Shy"
    elif stroke > 0.6:
        return "Strong-willed/Stressed"
    else:
        return "Neutral"

# --- Function to search image automatically ---
def find_image(filename, search_dir):
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower() == filename.lower():
                return os.path.join(root, file)
    return None

# --- Main ---

# Your image file name (with spaces and jpg extension)
image_file = "data2.jpg"

# Search in project folder automatically
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from backend
image_path = find_image(image_file, project_dir)

if image_path is None:
    print(f"Error: Could not find '{image_file}' in project folder '{project_dir}'")
else:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to read image at '{image_path}'. Check format/extension.")
    else:
        # Calculate features
        slant_angle = calculate_slant_angle(img)
        avg_size = calculate_size(img)
        stroke = calculate_stroke(img)

        # Predict behavior
        behavior = predict_behavior(slant_angle, avg_size, stroke)
        print(f"Detected Behavior: {behavior}")
        print(f"Slant Angle: {slant_angle:.2f}°, Avg Letter Size: {avg_size:.2f}, Stroke: {stroke:.2f}")
