import cv2
import numpy as np

# Function to calculate slant angle
def calculate_slant_angle(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)
    if len(angles) == 0:
        return 0
    avg_angle = np.mean(angles)
    return avg_angle

# Function to calculate average letter size
def calculate_size(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    areas = [cv2.boundingRect(cnt)[3] for cnt in contours]  # height of each contour
    avg_size = np.mean(areas)
    return avg_size

# Function to calculate average stroke thickness
def calculate_stroke(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    thickness = np.mean(thresh) / 255  # average pixel value for stroke thickness
    return thickness

# Rule-based behavior prediction
def predict_behavior(slant_angle, avg_size, stroke):
    if slant_angle > 5 and avg_size > 30:
        return "Outgoing/Confident 😎"
    elif slant_angle < -5 and avg_size < 20:
        return "Introverted/Shy 🤫"
    elif stroke > 0.6:
        return "Strong-willed/Stressed 💪"
    else:
        return "Neutral 😐"

# --- Main ---
# Load handwriting image in grayscale
image_path = 'handwriting_sample.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calculate features
slant_angle = calculate_slant_angle(img)
avg_size = calculate_size(img)
stroke = calculate_stroke(img)

# Predict behavior
behavior = predict_behavior(slant_angle, avg_size, stroke)
print(f"Detected Behavior: {behavior}")

# Optional: print features
print(f"Slant Angle: {slant_angle:.2f}°, Avg Letter Size: {avg_size:.2f}, Stroke: {stroke:.2f}")
