# Sistem Analisis Kepatuhan Pajak (K-Means Clustering)

Aplikasi web berbasis Flask untuk analisis dan klasterisasi tingkat kepatuhan wajib pajak menggunakan algoritma K-Means Clustering.

## Fitur Utama
- **Autentikasi Pengguna**: Login & Logout session.
- **Input & Upload Data**: Mendukung upload file Excel (.xlsx) dan input manual pembayaran pajak.
- **Preprocessing & Pembersihan Data**: Validasi data, eliminasi duplikat, dan penyesuaian jatuh tempo pajak.
- **Kewajiban Pajak**: Perhitungan kepatuhan penyetoran berdasarkan aturan perpajakan (PMK-9/PMK.03/2022).
- **Agregasi Data**: Rekapitulasi rasio tepat waktu, terlambat, belum bayar, dan rata-rata keterlambatan.
- **K-Means Clustering**: Pengelompokan tingkat kepatuhan wajib pajak (Patuh, Kurang Patuh, Tidak Patuh) dilengkapi evaluasi Elbow Method & Silhouette Score.

## Instalasi & Menjalankan Aplikasi
1. Clone repositori:
   ```bash
   git clone https://github.com/fcahyanto4-cmyk/ML_k-means.git
   cd ML_k-means
   ```

2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Konfigurasi database MySQL pada `app.py`:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://<username>:<password>@localhost:3306/<database_name>'
   ```

5. Jalankan aplikasi:
   ```bash
   python app.py
   ```
   Buka peramban pada `http://127.0.0.1:5000`.
