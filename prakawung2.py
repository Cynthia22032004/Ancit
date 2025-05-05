import cv2
import numpy as np

# ==== 1. Baca Gambar ====
img = cv2.imread(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\kawung2.jpg')

if img is None:
    print("Gambar tidak ditemukan.")
    exit()
else:
    print("Gambar berhasil dibaca.")

# ==== 2. Resize ke 256x256 ====
resized_img = cv2.resize(img, (256, 256))
cv2.imwrite(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\resized_kawung2.jpg', resized_img)
print("Gambar berhasil di-resize.")

# ==== 3. Normalisasi Warna (RGB jadi 0–1) ====
normalized_img = resized_img / 255.0
normalized_save = (normalized_img * 255).astype('uint8')  # untuk disimpan
cv2.imwrite(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\normalized_kawung2.jpg', normalized_save)
print("Normalisasi warna selesai.")

# ==== 4. Konversi ke Grayscale ====
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\gray_kawung2.jpg', gray_img)
print("Konversi ke grayscale selesai.")

# ==== 5. Ekstraksi Fitur Warna (RGB) ====
r_mean = np.mean(resized_img[:, :, 2])  # Red channel
g_mean = np.mean(resized_img[:, :, 1])  # Green channel
b_mean = np.mean(resized_img[:, :, 0])  # Blue channel

print(f"Rata-rata Nilai RGB:")
print(f"   R (Red)   : {r_mean:.2f}")
print(f"   G (Green) : {g_mean:.2f}")
print(f"   B (Blue)  : {b_mean:.2f}")

# ==== 6. Tampilkan Gambar Hasil ====

# Tampilkan gambar asli
cv2.imshow("Gambar Asli (kawung6)", img)

# Tampilkan gambar yang sudah di-resize
cv2.imshow("Gambar Resize 256x256", resized_img)

# Tampilkan hasil normalisasi (dalam skala tampilan normal 0–255)
cv2.imshow("Gambar Normalisasi", normalized_save)

# Tampilkan grayscale
cv2.imshow("Grayscale", gray_img)

# Tunggu tombol ditekan untuk menutup semua jendela
cv2.waitKey(0)
cv2.destroyAllWindows()
