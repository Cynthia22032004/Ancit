import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color
from scipy.stats import entropy

# Folder input
folder_path = r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# List untuk menyimpan hasil
results = []

# Loop semua file
for file in image_files:
    img_path = os.path.join(folder_path, file)
    
    # Baca dan konversi ke grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hitung GLCM (arah 0 derajat, jarak 1 piksel)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Ekstraksi fitur
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    idm = 1. / (1. + contrast)  # Approximate IDM
    corr = graycoprops(glcm, 'correlation')[0, 0]
    glcm_entropy = entropy(glcm.ravel())

    results.append({
        'Image': file,
        'Contrast': contrast,
        'ASM': asm,
        'IDM': idm,
        'Entropy': glcm_entropy,
        'Correlation': corr
    })

# Simpan ke CSV
df = pd.DataFrame(results)
df.to_csv('fitur_tekstur_glcm.csv', index=False)
print(df)
