#pip install opencv-python
import cv2 as cv
import numpy as np
import pandas as pd
import os

# ---- Load image ----
# ---- Extract base name from the input image ----
input_image_path = "2025-08-05_mask.jpg"
base_name = os.path.splitext(os.path.basename(input_image_path))[0]
img = cv.imread(input_image_path)

assert img is not None, "Image not found"

# ---- Convert to grayscale ----
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ---- Denoise (important!) ----
blur = cv.GaussianBlur(gray, (5, 5), 0)

# ---- Automatic threshold (Otsu) ----
_, thresh = cv.threshold(
    blur, 0, 255,
    cv.THRESH_BINARY + cv.THRESH_OTSU
)

# ---- Morphological cleanup ----
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
clean = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# Optional: fill gaps
clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, kernel, iterations=2)

# ---- Find contours ----
contours, hierarchy = cv.findContours(
    clean,
    cv.RETR_EXTERNAL,   # only outer contours (usually better)
    cv.CHAIN_APPROX_SIMPLE
)

# ---- Filter contours (remove noise) ----
min_area = 100  # adjust based on your data
filtered_contours = []

for cnt in contours:
    area = cv.contourArea(cnt)
    if area > min_area:
        filtered_contours.append(cnt)

# ---- Draw contours ----
output = img.copy()
cv.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)

# ---- Draw bounding boxes ----
for cnt in filtered_contours:
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

# ---- Generate table of contours ----
contour_data = []

for i, cnt in enumerate(filtered_contours):
    x, y, w, h = cv.boundingRect(cnt)
    contour_data.append({
        "Leision_ID": i,
        "X": x,
        "Y": y,
        "Width": w,
        "Height": h,
        "Area": cv.contourArea(cnt)
    })

# Create a DataFrame
df = pd.DataFrame(contour_data)

# Save the table as a CSV file with the original image name appended
csv_filename = f"{base_name}_defined_leision_id.csv"
df.to_csv(csv_filename, index=False)

# ---- Draw bounding boxes and add contour IDs ----
for i, cnt in enumerate(filtered_contours):
    x, y, w, h = cv.boundingRect(cnt)
    # Draw bounding box
    cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Add contour ID above the bounding box
    cv.putText(output, str(i), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 1)

# Save the output image with the original image name appended
output_image_filename = f"{base_name}_added_leision_id.png"
cv.imwrite(output_image_filename, output)