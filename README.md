# 🌋 Earthquake Clustering Viewer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Visualisasi interaktif dan analisis persebaran gempa bumi menggunakan **Machine Learning (K-Means Clustering)**. Aplikasi ini membantu mengidentifikasi pola kegempaan, tingkat risiko, dan karakteristik setiap klaster gempa di wilayah Indonesia dan sekitarnya.

---

## 👥 Anggota Kelompok 2

| No  | Nama                      | NIM            |
| :-- | :------------------------ | :------------- |
| 1.  | **Adam Haritsa Thahara**  | A11.2024.15556 |
| 2.  | **Angela Echa Naresti**   | A11.2024.15791 |
| 3.  | **Farros Rifantiarno R.** | A11.2024.15694 |
| 4.  | **Nadjwa Salsabila W.**   | A11.2024.15670 |
| 5.  | **Mohammad Wisam W.**     | A11.2024.15739 |

---

## 🌟 Fitur Utama

### 1. 🗺️ Peta Interaktif

Visualisasi titik gempa secara _real-time_ di atas peta interaktif. Setiap klaster ditandai dengan warna berbeda untuk memudahkan identifikasi zona gempa.

- **Marker Clustering:** Titik gempa dikelompokkan saat zoom-out agar peta tetap rapi.
- **Detail Popup:** Klik marker untuk melihat Magnitudo, Kedalaman, dan Info Klaster.

### 2. 🤖 K-Means Clustering

Mengelompokkan data gempa berdasarkan:

- **Lokasi** (Latitude, Longitude)
- **Magnitudo** (Kekuatan Gempa)
- **Kedalaman** (Depth)

### 3. 🔍 Filter Data Lengkap

Eksplorasi data dengan fleksibel menggunakan filter di sidebar:

- **Rentang Tanggal:** Pilih periode waktu tertentu.
- **Magnitudo:** Filter berdasarkan kekuatan gempa.
- **Kedalaman:** Filter gempa dangkal, menengah, atau dalam.

### 4. ⚡ Prediksi Cepat (AI Prediction)

Ingin tahu risiko gempa baru? Masukkan parameter (Lat, Lon, Mag, Depth) dan model AI kami akan langsung memprediksi:

- Masuk ke **Cluster** mana.
- **Severity Score** (Tingkat Keparahan).
- **Label Risiko** (e.g., Megathrust Potential, Normal, dll).

### 5. ⚖️ Balanced Sampling (Water Filling Algorithm)

Teknologi **Smart Sampling** memastikan visualisasi data yang seimbang di peta, sehingga klaster minoritas tetap terlihat dan tidak tertutup oleh ribuan data dari klaster mayoritas.

---

## 🛠️ Instalasi & Cara Pakai

Pastikan Anda memiliki [Python](https://www.python.org/) yang terinstall di komputer Anda.

1. **Clone Repository (atau download folder project ini):**

   ```bash
   git clone https://github.com/username/earthquake-clustering-viewer.git
   cd earthquake-clustering-viewer
   ```

2. **Buat Virtual Environment (Opsional tapi Direkomendasikan):**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi:**
   ```bash
   streamlit run streamlit_app.py
   ```
   Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

---

## 📦 Struktur File

- `streamlit_app.py`: 🧠 Kode utama aplikasi (UI & Logic).
- `requirements.txt`: 📋 Daftar library yang dibutuhkan.
- `kmeans_pipeline.joblib`: 🤖 Model Machine Learning yang sudah dilatih.
- `cluster_info.csv`: 📊 Metadata klaster (Label risiko, rata-rata magnitudo, dll).
- `earthquakes_with_cluster.csv`: 💾 Dataset utama gempa bumi.

---

## 🤝 Kontribusi

Pull Request sangat diterima! Jika Anda menemukan bug atau punya ide fitur baru, silakan buka **Issue** atau kirim **Pull Request**.

---

Dirancang dengan ❤️ menggunakan **Streamlit** & **Python**.
