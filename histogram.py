results_hist = []

for file in image_files:
    path = os.path.join(folder_path, file)
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))

    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()

    # Hitung statistik sederhana dari histogram (rata-rata, standar deviasi)
    mean_r, std_r = np.mean(hist_r), np.std(hist_r)
    mean_g, std_g = np.mean(hist_g), np.std(hist_g)
    mean_b, std_b = np.mean(hist_b), np.std(hist_b)

    results_hist.append({
        'Image': file,
        'Mean_R': mean_r,
        'Std_R': std_r,
        'Mean_G': mean_g,
        'Std_G': std_g,
        'Mean_B': mean_b,
        'Std_B': std_b
    })

df_hist = pd.DataFrame(results_hist)
df_hist.to_csv('fitur_histogram.csv', index=False)
print(df_hist)
