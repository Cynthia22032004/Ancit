import cv2
import numpy as np

# 1. Baca gambar (misalnya dalam mode grayscale atau RGB)
img = cv2.imread(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\normalized_ceplok1.jpg')  # Ganti dengan path ke gambar kamu
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Jika ingin grayscale

# 2. Tentukan ukuran blok
block_size = 8

# 3. Ambil dimensi gambar
height, width, _ = img.shape

# 4. Iterasi per blok 8x8
for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        # Ambil blok 8x8 dari gambar
        block = img[y:y+block_size, x:x+block_size]
        
        # Contoh: cetak ukuran dan rata-rata warna blok
        print(f'Blok ({x}, {y}) - Ukuran: {block.shape}')
        avg_color = block.mean(axis=(0, 1))
        print(f'  Rata-rata warna: {avg_color}')
