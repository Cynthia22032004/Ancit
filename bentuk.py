import cv2
import numpy as np
import os
import pandas as pd

folder_path = r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

results_shape = []

for file in image_files:
    path = os.path.join(folder_path, file)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)  # gunakan kontur terbesar
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0

        results_shape.append({
            'Image': file,
            'Area': area,
            'Perimeter': perimeter,
            'AspectRatio': aspect_ratio,
            'Extent': extent,
            'Solidity': solidity
        })

df_shape = pd.DataFrame(results_shape)
df_shape.to_csv('fitur_bentuk.csv', index=False)
print(df_shape)
