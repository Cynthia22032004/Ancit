import cv2

# Path gambar
img = cv2.imread(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\kawung6.jpg')

# Periksa apakah gambar berhasil dibaca
if img is None:
    print("Gambar tidak ditemukan atau tidak bisa dibaca.")
else:
    print("Gambar berhasil dibaca.")
    resized_img = cv2.resize(img, (256, 256))
    cv2.imwrite(r'C:\Users\yayuk\Documents\Semester 6\Analisis citra\UTS\image\resized_kawung6.jpg', resized_img)
    print("Gambar berhasil di-resize dan disimpan.")
