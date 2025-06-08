# Proyek Prediksi Penjualan Retail Fashion

Proyek ini berfokus pada prediksi penjualan harian retail fashion menggunakan model peramalan deret waktu canggih: SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) dan GRU (Gated Recurrent Unit). Tujuannya adalah untuk memberikan perkiraan penjualan yang akurat dengan memanfaatkan data penjualan historis dan berbagai fitur berbasis waktu.

## 📊 Dataset

dataset yang kita gunakan adalah `Fashion_Retail_Sales.csv`, yang berisi catatan transaksi retail fashion. Kolom-kolom utama meliputi:
-   `Customer Reference ID`: Pengenal unik untuk setiap pelanggan.
-   `Item Purchased`: Item fashion yang dibeli.
-   `Purchase Amount (USD)`: Nilai transaksi dalam USD.
-   `Date Purchase`: Tanggal pembelian.
-   `Review Rating`: Skor ulasan pelanggan untuk item yang dibeli.
-   `Payment Method`: Metode pembayaran yang digunakan (misalnya, Kartu Kredit, Tunai).

## 💡 Metodologi

Pipeline prediksi melibatkan beberapa langkah penting:

### 🧹 Pembersihan dan Pra-pemrosesan Data

1.  **Imputasi Nilai Hilang**: Nilai yang hilang dalam `Purchase Amount (USD)` dan `Review Rating` ditangani menggunakan interpolasi linier.
2.  **Konversi & Pengurutan Tanggal**: Kolom `Date Purchase` dikonversi menjadi objek datetime dan dataset diurutkan berdasarkan tanggal.
3.  **Resampling Harian**: Data di-resample ke frekuensi harian, mengagregasi `Purchase Amount (USD)` (jumlah), `Item Purchased` (hitung), dan `Review Rating` (rata-rata). Entri harian yang hilang (misalnya, tidak ada penjualan pada hari tertentu) diisi dengan nol.
4.  **Penghalusan (Smoothing)**: Rata-rata bergerak 7 hari diterapkan pada `Purchase Amount (USD)` untuk penghalusan, dengan `bfill` dan `ffill` menangani kasus-kasus tepi.
5.  **Rekayasa Fitur (Feature Engineering)**: Beberapa fitur berbasis waktu diekstrak untuk menangkap musiman dan tren:
    * `day_of_week` (hari dalam seminggu)
    * `month` (bulan)
    * `quarter` (kuartal)
    * `is_weekend` (apakah akhir pekan)
    * `MA7` (Rata-rata Bergerak 7 hari)
    * `MA30` (Rata-rata Bergerak 30 hari)
6.  **Penskalaan Fitur**: Semua variabel eksogen (`Item Purchased`, `Review Rating`, `day_of_week`, `month`, `quarter`, `is_weekend`, `MA7`, `MA30`) dan variabel target (`Purchase Amount (USD)`) diskalakan menggunakan `MinMaxScaler` untuk menormalkan rentang nilainya.

### 🤖 Pembuatan Model

Data dibagi menjadi 80% set pelatihan dan 20% set pengujian. Dua model berbeda dilatih dan dievaluasi:

1.  **Enhanced SARIMAX Model**:
    * SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) digunakan karena ketahanannya dalam menangani data deret waktu dengan musiman dan tren, serta dapat memasukkan variabel eksogen.
    * **Penyetelan Parameter**: Pendekatan pencarian grid digunakan untuk menemukan order `(p,d,q)` dan order musiman `(P,D,Q,s)` terbaik dengan mengevaluasi berbagai kombinasi berdasarkan AIC (Akaike Information Criterion).
    * **Parameter Terbaik**: `((2, 0, 2), (1, 1, 1, 7))` diidentifikasi sebagai konfigurasi SARIMAX terbaik dengan AIC sebesar -576.36.

2.  **GRU (Gated Recurrent Unit) Neural Network**:
    * Jaringan saraf berbasis GRU dikembangkan untuk menangkap dependensi temporal yang kompleks dalam data.
    * **Pembuatan Urutan (Sequence)**: Data diubah menjadi urutan dengan `n_steps` (panjang urutan) 14, cocok untuk jaringan saraf berulang.
    * **Arsitektur**: Model GRU terdiri dari beberapa lapisan GRU diikuti oleh lapisan Dense untuk keluaran.
        * `GRU(128, return_sequences=True)`
        * `GRU(64, return_sequences=True)`
        * `GRU(32, return_sequences=False)`
        * `Dense(32, activation='relu')`
        * `Dense(16, activation='relu')`
        * `Dense(1, activation='linear')`
    * **Training**: Model dikompilasi dengan pengoptimal Adam (tingkat pembelajaran: 0.0005) dan fungsi kerugian Huber (lebih kuat terhadap outlier). Callback `EarlyStopping` dan `ReduceLROnPlateau` diimplementasikan untuk mencegah overfitting dan mengoptimalkan pembelajaran.

### 📈 Evaluasi

Kedua model dievaluasi menggunakan metrik berikut pada set pengujian:
-   **RMSE (Root Mean Squared Error)**: Mengukur rata-rata magnitudo kesalahan. Lebih rendah lebih baik.
-   **MAE (Mean Absolute Error)**: Mengukur rata-rata magnitudo kesalahan. Lebih rendah lebih baik.
-   **MAPE (Mean Absolute Percentage Error)**: Menyatakan akurasi sebagai persentase kesalahan. Lebih rendah lebih baik.
-   **R² Score (Koefisien Determinasi)**: Menunjukkan proporsi varians dalam variabel dependen yang dapat diprediksi dari variabel independen. Lebih tinggi lebih baik (mendekati 1).

## 🌊 Aspek Musiman dalam Proyek

Dalam proyek ini, **musiman (seasonal)** mengacu pada pola atau fluktuasi yang berulang dalam data penjualan pada interval waktu yang tetap dan dapat diprediksi. Aspek musiman ditangani di beberapa bagian:

1.  **Model SARIMAX:**
    * **Orde Musiman `(P,D,Q,s)`**: Ini adalah bagian paling eksplisit dari musiman dalam model SARIMAX. Nilai `s` yang ditemukan adalah `7`, menunjukkan bahwa model mencari pola musiman yang berulang setiap 7 hari (siklus mingguan), yang sangat relevan untuk data penjualan harian (misalnya, perbedaan penjualan antara hari kerja dan akhir pekan).
    * **`P=1`**: Menangkap ketergantungan linier antara observasi saat ini dan observasi pada periode musiman sebelumnya (seminggu yang lalu).
    * **`D=1`**: Menerapkan differencing musiman tingkat pertama untuk menghilangkan tren musiman.
    * **`Q=1`**: Menangkap ketergantungan pada kesalahan peramalan musiman sebelumnya.
    * Proses penyetelan parameter secara aktif mencari kombinasi orde musiman ini untuk menemukan model yang paling cocok.

2.  **Rekayasa Fitur (Feature Engineering):**
    * Meskipun model GRU tidak secara eksplisit memiliki komponen "seasonal" seperti SARIMAX, fitur-fitur ini membantu kedua model (terutama GRU) untuk belajar dari pola musiman:
        * **`day_of_week`**: Secara langsung menangkap pola penjualan yang bervariasi berdasarkan hari dalam seminggu.
        * **`month`**: Dapat menangkap pola musiman tahunan (misalnya, penjualan lebih tinggi di bulan-bulan tertentu seperti Desember karena liburan).
        * **`quarter`**: Mirip dengan bulan, juga dapat menunjukkan pola penjualan musiman yang berulang setiap tiga bulan.
        * **`is_weekend`**: Fitur biner yang menyoroti perbedaan signifikan antara penjualan hari kerja dan akhir pekan.

## 🚀 Hasil

### Perbandingan Kinerja Model

| Metrik       | SARIMAX | GRU     |
| :----------- | :------ | :------ |
| **RMSE** | 345.28  | **241.28**|
| **MAE** | 271.05  | **170.78**|
| **MAPE (%)** | 22.95%  | **14.35%**|
| **R² Score** | 0.2638  | **0.6491**|

**Ringkasan**: Model **GRU** secara konsisten mengungguli model SARIMAX di semua metrik evaluasi, menunjukkan akurasi yang lebih baik dan proporsi varians yang dijelaskan lebih tinggi.

### Prakiraan 7 Hari ke Depan

Model memprediksi penjualan untuk 7 hari ke depan (2 Oktober - 8 Oktober 2023):

| Tanggal    | SARIMAX (USD) | GRU (USD) |
| :--------- | :------------ | :-------- |
| 2023-10-02 | 821.29        | 1039.73   |
| 2023-10-03 | 994.17        | 1151.02   |
| 2023-10-04 | 1010.13       | 1242.55   |
| 2023-10-05 | 1069.29       | 1306.13   |
| 2023-10-06 | 1054.72       | 1341.73   |
| 2023-10-07 | 1007.85       | 1353.08   |
| 2023-10-08 | 1142.71       | 1345.65   |

*(Catatan: Nilai-nilai tersebut mewakili perkiraan jumlah pembelian dalam USD.)*

### Visualisasi

Proyek ini mencakup beberapa plot untuk memvisualisasikan data dan model kinerja:
-   **Original Purchase Amount (USD) per Transaction**: Menampilkan data penjualan mentah seiring waktu.
-   **Processed Purchase Amount (USD) per Day (Scaled)**: Menunjukkan data setelah pembersihan, resampling, dan penskalaan.
-   **SARIMAX Model Prediction Results**: Membandingkan prediksi SARIMAX dengan nilai aktual, termasuk interval kepercayaan 95%.
-   **GRU Model Prediction Results**: Membandingkan prediksi GRU dengan nilai aktual.
-   **Comparison of SARIMAX and GRU Predictions**: Plot gabungan yang menunjukkan bagaimana kedua model bekerja dibandingkan dengan nilai aktual.
-   **Performance Bar Chart**: Bagan batang yang membandingkan secara visual skor RMSE, MAE, MAPE, dan R² dari SARIMAX dan GRU.
-   **7-Day Forecast Comparison**: Memvisualisasikan prediksi penjualan di masa mendatang dari kedua model.

## 🏃 Cara Menjalankan

1.  **Clone repositori (jika berlaku)**:
    ```bash
    git clone <url_repositori>
    cd <nama_repositori>
    ```
2.  **Instal dependensi**:
    ```bash
    pip install pandas numpy matplotlib scikit-learn statsmodels tensorflow
    ```
3.  **Jalankan Jupyter Notebook**:
    Buka `notebooks/main.ipynb` di lingkungan Jupyter dan eksekusi semua sel.
    ```bash
    jupyter notebook
    ```
4.  **Jalankan Skrip Python**:
    Eksekusi file `src/main.py` secara langsung.
    ```bash
    python src/main.py
    ```

## 🎯 Kesimpulan

GRU terbukti lebih efektif untuk memprediksi penjualan retail fashion dalam dataset ini, mencapai RMSE yang secara signifikan lebih rendah dan skor R² yang lebih tinggi dibandingkan dengan model SARIMAX. Ini menunjukkan bahwa kemampuan GRU untuk menangkap pola non-linier yang kompleks dan dependensi jangka panjang dalam data deret waktu, berkontribusi pada kinerja superiornya. 
Proyek ini memberikan wawasan berharga tentang tren penjualan di masa depan untuk bisnis retail fashion.
