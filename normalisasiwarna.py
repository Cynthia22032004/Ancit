import cv2
import numpy as np

# 1. Baca gambar
img = cv2.imread("resized_kawung1.jpg")  

# 3. Normalisasi warna (nilai RGB menjadi 0-1)
normalized_img = resized_img / 255.0

# 4. Tampilkan hasil normalisasi (jika ingin melihat)
cv2.imshow("Normalized Image", normalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()