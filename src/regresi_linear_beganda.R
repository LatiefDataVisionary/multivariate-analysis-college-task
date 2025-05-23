# ==============================================================================
#                      ANALISIS REGRESI LINEAR BERGANDA DI R
# ==============================================================================
cat("==============================================================================\n")
cat("                     ANALISIS REGRESI LINEAR BERGANDA DI R                     \n")
cat("==============================================================================\n")

# 1. Persiapan Data
X1 <- c(50, 30, 70, 20, 60, 40, 25, 80, 35, 45)  # Budget Iklan
X2 <- c(5, 3, 7, 2, 6, 4, 3, 8, 4, 5)          # Jumlah Sales
Y <- c(120, 80, 200, 50, 180, 100, 70, 250, 90, 130) # Penjualan

df <- data.frame(X1, X2, Y)

cat("\n--- 1. DATASET AWAL ---\n")
print(df)

# 2. Menghitung Jumlah (Summations) dan n
n <- nrow(df)
sum_X1 <- sum(df$X1)
sum_X2 <- sum(df$X2)
sum_Y <- sum(df$Y)

sum_X1_sq <- sum(df$X1^2)
sum_X2_sq <- sum(df$X2^2)
sum_Y_sq <- sum(df$Y^2) # Meskipun tidak eksplisit di rumus a, b1, b2, ada di "yang mana"

sum_X1Y <- sum(df$X1 * df$Y)
sum_X2Y <- sum(df$X2 * df$Y)
sum_X1X2 <- sum(df$X1 * df$X2)

cat("\n\n--- 2. NILAI-NILAI AWAL ---\n")
cat(sprintf("Jumlah Observasi (n)                 : %d\n", n))
cat("------------------------------------------------------------\n")
cat(sprintf("ΣX1 (Jumlah Budget Iklan)            : %.0f\n", sum_X1))
cat(sprintf("ΣX2 (Jumlah Sales)                   : %.0f\n", sum_X2))
cat(sprintf("ΣY (Jumlah Penjualan)                : %.0f\n", sum_Y))
cat("------------------------------------------------------------\n")
cat(sprintf("ΣX1^2 (Jumlah Kuadrat Budget Iklan)  : %.0f\n", sum_X1_sq))
cat(sprintf("ΣX2^2 (Jumlah Kuadrat Sales)         : %.0f\n", sum_X2_sq))
cat(sprintf("ΣY^2 (Jumlah Kuadrat Penjualan)      : %.0f\n", sum_Y_sq))
cat("------------------------------------------------------------\n")
cat(sprintf("ΣX1Y (Jumlah Produk X1 dan Y)        : %.0f\n", sum_X1Y))
cat(sprintf("ΣX2Y (Jumlah Produk X2 dan Y)        : %.0f\n", sum_X2Y))
cat(sprintf("ΣX1X2 (Jumlah Produk X1 dan X2)      : %.0f\n", sum_X1X2))

# 3. Menghitung Koreksi Jumlah Kuadrat dan Produk (sesuai bagian "yang mana")
# Σx1^2 = ΣX1^2 - (ΣX1)^2 / n
sum_x1_sq_corrected <- sum_X1_sq - (sum_X1^2) / n

# Σx2^2 = ΣX2^2 - (ΣX2)^2 / n
sum_x2_sq_corrected <- sum_X2_sq - (sum_X2^2) / n

# Σy^2 = ΣY^2 - (ΣY)^2 / n
sum_y_sq_corrected <- sum_Y_sq - (sum_Y^2) / n

# Σx1y = ΣX1Y - (ΣX1 * ΣY) / n
sum_x1y_corrected <- sum_X1Y - (sum_X1 * sum_Y) / n

# Σx2y = ΣX2Y - (ΣX2 * ΣY) / n
sum_x2y_corrected <- sum_X2Y - (sum_X2 * sum_Y) / n

# Σx1x2 = ΣX1X2 - (ΣX1 * ΣX2) / n
sum_x1x2_corrected <- sum_X1X2 - (sum_X1 * sum_X2) / n

cat("\n\n--- 3. KOREKSI JUMLAH KUADRAT DAN PRODUK (Notasi Huruf Kecil) ---\n")
cat(sprintf("Σx1^2                                : %10.2f\n", sum_x1_sq_corrected))
cat(sprintf("Σx2^2                                : %10.2f\n", sum_x2_sq_corrected))
cat(sprintf("Σy^2                                 : %10.2f\n", sum_y_sq_corrected))
cat(sprintf("Σx1y                                 : %10.2f\n", sum_x1y_corrected))
cat(sprintf("Σx2y                                 : %10.2f\n", sum_x2y_corrected))
cat(sprintf("Σx1x2                                : %10.2f\n", sum_x1x2_corrected))

# 4. Menghitung Koefisien Regresi b1 dan b2
# Denominator untuk b1 dan b2: (Σx1^2 * Σx2^2) - (Σx1x2)^2
denominator_b <- (sum_x1_sq_corrected * sum_x2_sq_corrected) - (sum_x1x2_corrected^2)

# b1 = [(Σx2^2 * Σx1y) - (Σx1x2 * Σx2y)] / Denominator
b1 <- ((sum_x2_sq_corrected * sum_x1y_corrected) - (sum_x1x2_corrected * sum_x2y_corrected)) / denominator_b

# b2 = [(Σx1^2 * Σx2y) - (Σx1x2 * Σx1y)] / Denominator
b2 <- ((sum_x1_sq_corrected * sum_x2y_corrected) - (sum_x1x2_corrected * sum_x1y_corrected)) / denominator_b

cat("\n\n--- 4. PERHITUNGAN KOEFISIEN REGRESI (b1 dan b2) ---\n")
cat(sprintf("Nilai Denominator untuk b1 dan b2    : %10.2f\n", denominator_b))
cat("------------------------------------------------------------\n")
cat(sprintf("Koefisien b1 (Budget Iklan)          : %10.4f\n", b1))
cat(sprintf("Koefisien b2 (Jumlah Sales)          : %10.4f\n", b2))

# 5. Menghitung Konstanta a
# a = (ΣY - (b1 * ΣX1) - (b2 * ΣX2)) / n
a <- (sum_Y - (b1 * sum_X1) - (b2 * sum_X2)) / n

cat("\n\n--- 5. PERHITUNGAN KONSTANTA REGRESI (a) ---\n")
cat(sprintf("Konstanta a (Intersep)               : %10.4f\n", a))

cat("\n\n==============================================================================\n")
cat("                     PERSAMAAN REGRESI LINEAR BERGANDA                      \n")
cat("==============================================================================\n")
cat(sprintf(" Y = %.4f + %.4f*X1 + %.4f*X2\n", a, b1, b2))
cat("------------------------------------------------------------------------------\n")
cat(sprintf(" Penjualan = %.4f + (%.4f * Budget Iklan) + (%.4f * Jumlah Sales)\n", a, b1, b2))
cat("==============================================================================\n")

cat("\n\n--- INTERPRETASI HASIL ---\n")
cat(sprintf("1. Konstanta (a) = %.4f:\n", a))
cat(sprintf("   Jika Budget Iklan (X1) dan Jumlah Sales (X2) bernilai 0, maka prediksi Penjualan (Y) adalah %.4f unit.\n", a))
cat(sprintf("\n2. Koefisien b1 (Budget Iklan) = %.4f:\n", b1))
cat(sprintf("   Setiap kenaikan 1 unit Budget Iklan (X1), dengan asumsi Jumlah Sales (X2) konstan,\n"))
cat(sprintf("   diperkirakan akan meningkatkan Penjualan (Y) sebesar %.4f unit.\n", b1))
cat(sprintf("\n3. Koefisien b2 (Jumlah Sales) = %.4f:\n", b2))
cat(sprintf("   Setiap kenaikan 1 unit Jumlah Sales (X2), dengan asumsi Budget Iklan (X1) konstan,\n"))
cat(sprintf("   diperkirakan akan meningkatkan Penjualan (Y) sebesar %.4f unit.\n", b2))
cat("==============================================================================\n")

# Verifikasi menggunakan fungsi lm() bawaan R (opsional)
cat("\n\n--- VERIFIKASI MENGGUNAKAN FUNGSI lm() BAWAAN R (OPSIONAL) ---\n")
model_lm <- lm(Y ~ X1 + X2, data = df)
print(summary(model_lm))
cat("\nKoefisien dari lm():\n")
print(coef(model_lm))
cat("==============================================================================\n")