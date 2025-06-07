import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split # Opsional, jika ingin split data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Memuat dan Mempersiapkan Data ---
print("--- 1. Memuat dan Mempersiapkan Data ---")

# Ganti 'dataset_LDA_2clusters.csv' dengan path file Anda jika berbeda
file_path = 'dataset_LDA_2clusters.csv'

try:
    # Membaca dataset, menggunakan ';' sebagai pemisah
    # Mengambil hanya 600 baris pertama data (tidak termasuk header)
    df = pd.read_csv(file_path, sep=';', nrows=600)
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan. Pastikan path file benar.")
    exit()
except Exception as e:
    print(f"Error saat membaca file CSV: {e}")
    exit()

# Menampilkan beberapa baris pertama data dan informasi dasar
print("\nDataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("\nDataFrame Describe:")
print(df.describe())
print("\nJumlah nilai unik di kolom 'Cluster':")
print(df['Cluster'].value_counts())

# Cek apakah ada nilai null
print("\nJumlah nilai null per kolom:")
print(df.isnull().sum())
# Jika ada nilai null, Anda mungkin perlu menanganinya (misalnya, drop atau imputasi)
# Untuk kesederhanaan, kita asumsikan tidak ada null atau sudah ditangani

# --- 2. Memisahkan Fitur (Variabel Independen) dan Target (Variabel Dependen/Cluster) ---
print("\n--- 2. Memisahkan Fitur dan Target ---")
# Fitur (variabel independen)
X = df[['BloodPressure', 'Age']]
# Target (variabel dependen)
y = df['Cluster']

print("\nFitur (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())

# (Opsional) Jika Anda ingin membagi data menjadi data latih dan data uji:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Untuk contoh ini, kita akan melatih dan menguji pada keseluruhan 600 data agar hasilnya
# bisa lebih mudah dibandingkan dengan perhitungan manual Excel yang juga menggunakan semua data.
# Jika tujuannya adalah generalisasi model, pemisahan data latih-uji sangat direkomendasikan.

# --- 3. Membuat dan Melatih Model LDA ---
print("\n--- 3. Membuat dan Melatih Model LDA ---")
# n_components adalah jumlah komponen diskriminan yang diinginkan.
# Untuk klasifikasi biner (2 kelas), maksimal ada 1 komponen diskriminan (min(n_classes-1, n_features)).
# Jika n_features > n_classes-1, maka n_components = n_classes-1.
# Di sini, n_classes = 2, n_features = 2. Jadi, min(2-1, 2) = 1.
lda_model = LinearDiscriminantAnalysis(n_components=1) # Bisa juga tidak dispesifikasikan jika n_classes=2

# Melatih model LDA
lda_model.fit(X, y) # Menggunakan semua data untuk pelatihan
print("Model LDA berhasil dilatih.")

# --- 4. Mendapatkan Koefisien Fungsi Diskriminan dan Intercept ---
print("\n--- 4. Koefisien Fungsi Diskriminan ---")
# Koefisien (bobot) untuk setiap fitur. Ini adalah b₁ dan b₂ dalam D = b₀ + b₁X₁ + b₂X₂
coefficients = lda_model.coef_[0] # coef_ adalah array 2D jika n_components > 1
intercept = lda_model.intercept_[0] # intercept_ adalah b₀

print(f"Koefisien (weights/loadings) untuk fitur {X.columns.tolist()}: {coefficients}")
print(f"Intercept (b₀): {intercept}")

print(f"\nFungsi Diskriminan (D): {intercept:.4f} + ({coefficients[0]:.4f} * BloodPressure) + ({coefficients[1]:.4f} * Age)")
print("Catatan: Tanda koefisien bisa berbeda dari SPSS atau perhitungan manual Excel")
print("tergantung pada kelompok mana yang dianggap sebagai referensi 'positif'.")
print("Namun, rasio antar koefisien dan kemampuan klasifikasinya seharusnya serupa.")
print("Scikit-learn LDA umumnya menghasilkan fungsi yang memprediksi probabilitas kelas, ")
print("dan decision_function > 0 biasanya mengarah ke kelas yang lebih tinggi (misal, kelas 1).")


# --- 5. Melakukan Prediksi pada Data ---
print("\n--- 5. Melakukan Prediksi ---")
# Memprediksi kelas untuk data X
y_pred = lda_model.predict(X)

# Mendapatkan skor diskriminan (jarak ke hyperplane pemisah)
# Skor ini adalah nilai dari D = b₀ + b₁X₁ + b₂X₂
# Jika skor > 0, biasanya diprediksi sebagai kelas 1 (jika kelas diberi label 0 dan 1).
discriminant_scores = lda_model.decision_function(X)

# Menambahkan hasil prediksi dan skor diskriminan ke DataFrame asli untuk inspeksi
df_results = df.copy()
df_results['Predicted_Cluster'] = y_pred
df_results['Discriminant_Score_D'] = discriminant_scores

print("\nBeberapa baris data dengan prediksi dan skor diskriminan:")
print(df_results.head())

# --- 6. Mengevaluasi Akurasi Model ---
print("\n--- 6. Mengevaluasi Akurasi Model ---")
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

print(f"\nAkurasi Model: {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(conf_matrix)
# Format Confusion Matrix:
#           Prediksi 0  Prediksi 1
# Aktual 0     TN          FP
# Aktual 1     FN          TP

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=lda_model.classes_, yticklabels=lda_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - LDA')
plt.show()

print("\nClassification Report:")
print(class_report)

# --- 7. Menampilkan Centroid Skor Diskriminan ---
# Scikit-learn LDA.transform() akan memberikan proyeksi data ke sumbu diskriminan.
# Rata-rata dari transformasi ini per grup akan menjadi centroid skor diskriminan.
print("\n--- 7. Centroid Skor Diskriminan (pada Sumbu Diskriminan) ---")
X_lda_transformed = lda_model.transform(X) # Proyeksi data ke 1D sumbu diskriminan

# Tambahkan transformasi ke df_results untuk menghitung rata-rata per grup
df_results['LD1_Score'] = X_lda_transformed.flatten() # Karena hanya 1 komponen

centroid_0 = df_results[df_results['Cluster'] == 0]['LD1_Score'].mean()
centroid_1 = df_results[df_results['Cluster'] == 1]['LD1_Score'].mean()

print(f"Centroid skor diskriminan untuk Cluster 0 (pada sumbu LD1): {centroid_0:.4f}")
print(f"Centroid skor diskriminan untuk Cluster 1 (pada sumbu LD1): {centroid_1:.4f}")

# Titik potong (cutoff) pada sumbu skor diskriminan LD1 bisa diestimasi
# sebagai rata-rata dari kedua centroid jika prior probabilities sama.
# Dalam scikit-learn, klasifikasi terjadi jika decision_function > 0 (untuk kelas positif).
# Nilai decision_function berhubungan dengan X_lda_transformed, tetapi mungkin di-scale/shift.
# Threshold pada decision_function adalah 0 secara default.
# Threshold pada LD1_Score dapat dihitung sebagai: (centroid_0 + centroid_1) / 2
# Atau, jika intercept dan koefisien fungsi D digunakan, thresholdnya adalah 0.

print("Perlu dicatat bahwa 'Discriminant_Score_D' yang dihitung dari coef_ dan intercept_")
print("sudah memiliki threshold implisit di 0 untuk klasifikasi.")
print("Nilai LD1_Score adalah proyeksi data, dan juga dapat digunakan untuk klasifikasi")
print("dengan threshold yang sesuai (misalnya, rata-rata centroid LD1_Score).")


# --- 8. Visualisasi Hasil Diskriminasi (Histogram Skor Diskriminan) ---
print("\n--- 8. Visualisasi Hasil Diskriminasi ---")
plt.figure(figsize=(10, 6))
sns.histplot(data=df_results, x='LD1_Score', hue='Cluster', kde=True, palette='viridis', multiple="stack")
plt.axvline(centroid_0, color=sns.color_palette('viridis')[0], linestyle='--', label=f'Centroid 0 ({centroid_0:.2f})')
plt.axvline(centroid_1, color=sns.color_palette('viridis')[1], linestyle='--', label=f'Centroid 1 ({centroid_1:.2f})')

# Tambahkan garis untuk decision_function = 0 (jika kita ingin menghubungkannya dengan LD1_Score)
# Ini memerlukan pemahaman bagaimana decision_function terkait dengan LD1_Score
# Decision_function(X) = X @ coef_.T + intercept_
# X_lda_transformed = X @ lda_model.scalings_ (scalings_ adalah eigenvector)
# Untuk satu komponen, lda_model.coef_ sebanding dengan lda_model.scalings_
# jadi, Decision_function kurang lebih proporsional dengan LD1_Score plus konstanta
# Jika LD1_Score adalah skor mentah proyeksi, kita bisa menghitung cutting score
cutting_score_ld1 = (centroid_0 + centroid_1) / 2
plt.axvline(cutting_score_ld1, color='red', linestyle=':', lw=2, label=f'Cutting Score LD1 ({cutting_score_ld1:.2f})')

plt.title('Distribusi Skor Diskriminan Linear (LD1) per Cluster')
plt.xlabel('Skor Diskriminan Linear (LD1)')
plt.ylabel('Frekuensi')
plt.legend()
plt.grid(True)
plt.show()

# Jika Anda ingin membandingkan dengan SPSS/Excel yang mungkin menggunakan
# fungsi D = b₀ + b₁X₁ + b₂X₂ dan klasifikasi D > 0.
# Kita sudah memiliki 'Discriminant_Score_D' di df_results.
plt.figure(figsize=(10, 6))
sns.histplot(data=df_results, x='Discriminant_Score_D', hue='Cluster', kde=True, palette='magma', multiple="stack")
plt.axvline(df_results[df_results['Cluster'] == 0]['Discriminant_Score_D'].mean(),
            color=sns.color_palette('magma')[0], linestyle='--', label=f"Mean D_Score (0)")
plt.axvline(df_results[df_results['Cluster'] == 1]['Discriminant_Score_D'].mean(),
            color=sns.color_palette('magma')[1], linestyle='--', label=f"Mean D_Score (1)")
plt.axvline(0, color='red', linestyle=':', lw=2, label='Decision Boundary (D=0)') # Threshold untuk D
plt.title('Distribusi Fungsi Diskriminan (D = b₀ + b₁X₁ + b₂X₂) per Cluster')
plt.xlabel('Skor Fungsi Diskriminan (D)')
plt.ylabel('Frekuensi')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Selesai ---")
