import cv2
import numpy as np
import os

# Folder gambar batik kamu
folder_path = r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\ceplok'
output_folder = folder_path + r'\hasil'  # folder untuk simpan hasil
os.makedirs(output_folder, exist_ok=True)

# Dapatkan semua file gambar di folder
file_list = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

# Loop untuk proses semua gambar
for file_name in file_list:
    print(f"\n Memproses: {file_name}")
    file_path = os.path.join(folder_path, file_name)
    img = cv2.imread(file_path)

    if img is None:
        print("Gagal membaca gambar.")
        continue

    # Resize
    resized = cv2.resize(img, (256, 256))

    # Normalisasi
    normalized = resized / 255.0

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Ekstraksi Fitur Warna (RGB)
    r_mean = np.mean(resized[:, :, 2])
    g_mean = np.mean(resized[:, :, 1])
    b_mean = np.mean(resized[:, :, 0])

    print(f"R: {r_mean:.2f}, G: {g_mean:.2f}, B: {b_mean:.2f}")

    # Simpan hasil (opsional)
    basename = os.path.splitext(file_name)[0]
    cv2.imwrite(os.path.join(output_folder, f"{basename}_resized.jpg"), resized)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_gray.jpg"), gray)

    # Simpan versi normalisasi (jika ingin lihat hasil visual)
    norm_vis = (normalized * 255).astype('uint8')
    cv2.imwrite(os.path.join(output_folder, f"{basename}_normalized.jpg"), norm_vis)
