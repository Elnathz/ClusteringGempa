# 🌋 Klasifikasi & Analisis Gempa (K-Means Clustering)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Data_Viz-3F51B5?style=for-the-badge&logo=plotly&logoColor=white)

Proyek **Data Mining** untuk klasifikasi dan analisis gempa bumi menggunakan **K-Means Clustering**. Sistem ini mengidentifikasi pola kegempaan berdasarkan magnitudo dan kedalaman, mengklasifikasi tingkat risiko, dan menyediakan visualisasi interaktif melalui aplikasi **Streamlit**.

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

## � Tujuan Proyek

1. **Melakukan Clustering** gempa bumi menggunakan algoritma K-Means.
2. **Menentukan K Optimal** dengan metode Silhouette Score dan Elbow Method.
3. **Mengklasifikasi Tingkat Risiko** berdasarkan magnitudo (70%) dan kedalaman (30%).
4. **Menyimpan Model** untuk deployment di aplikasi Streamlit.
5. **Visualisasi Data** menggunakan peta interaktif dan grafik analisis.

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

### 1. 📊 Analisis Clustering Otomatis

- **Evaluasi K Optimal:** Menggunakan Silhouette Score untuk menentukan jumlah klaster terbaik.
- **Visualisasi Elbow & Silhouette:** Menampilkan grafik untuk membantu pemilihan K.
- **Remapping Cluster:** Mereorder klaster berdasarkan severity score dari Low → Very High.

### 2. 🎲 Klasifikasi Tingkat Risiko

Sistem memberikan label risiko otomatis berdasarkan formula:

```
Severity Score = (0.7 × Magnitude_norm) + (0.3 × Shallow_Depth_inv_norm)
```

**Label Risiko:**

- 🟢 **Low** - Risiko rendah (gempa kecil/dalam)
- 🟡 **Moderate** - Risiko sedang
- 🟠 **High** - Risiko tinggi
- 🔴 **Very High** - Risiko sangat tinggi (gempa besar/dangkal)

### 3. 🗺️ Visualisasi Interaktif

- **Scatter Plot:** Distribusi gempa (Magnitudo vs Kedalaman).
- **Box Plot:** Analisis statistik per klaster.
- **Pie & Bar Chart:** Persentase dan jumlah gempa per risiko.
- **Peta Streamlit:** Tampilan geografis real-time dengan filter dinamis.

### 4. 🔍 Filter & Eksplorasi Data

Aplikasi Streamlit menyediakan:

- Filter berdasarkan **Rentang Tanggal**
- Filter berdasarkan **Magnitudo** (Min-Max)
- Filter berdasarkan **Kedalaman** (Min-Max)
- Filter berdasarkan **Tingkat Risiko**

### 5. 🤖 Prediksi Gempa Baru

Masukkan magnitudo & kedalaman untuk mendapatkan prediksi:

- Cluster ID
- Risk Level
- Severity Score

---

## 📁 Struktur File & Deskripsi

```
KlasifikasiGempa/
├── earthquake_clustering.ipynb          # 🔬 Main Jupyter Notebook (Analisis & Training)
├── streamlit_app.py                     # 🧠 Aplikasi web interaktif
├── katalog_gempa_v2.tsv                 # 📥 Dataset mentah (gempa bumi)
│
├── kmeans_model.joblib                  # 🤖 Model K-Means terlatih
├── scaler.joblib                        # 📏 MinMaxScaler untuk normalisasi
├── cluster_remap.pkl                    # 🔄 Mapping cluster ke label risiko
│
├── cluster_info.csv                     # 📊 Metadata cluster (centroid, severity_score)
├── earthquakes_with_cluster.csv         # 💾 Dataset dengan label cluster
│
├── requirements.txt                     # 📦 Dependencies
├── README.md                            # 📖 Dokumentasi (file ini)
└── notes/                               # 📝 Catatan tambahan
    ├── PENJELASAN_PERUBAHAN.txt         # Daftar perubahan dari versi awal
    └── informasi_keselamatan_streamlit.txt
```

### Penjelasan File Penting:

| File                           | Deskripsi                                                                                                                                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `earthquake_clustering.ipynb`  | **Notebook utama** berisi: import data, exploratory analysis, K-Means training, evaluasi model (Silhouette, Davies-Bouldin, Calinski-Harabasz), cluster remapping, dan visualisasi hasil. |
| `streamlit_app.py`             | **Aplikasi web** untuk eksplorasi data interaktif dengan sidebar filters, peta geografis, dan prediksi cluster gempa baru.                                                                |
| `katalog_gempa_v2.tsv`         | **Dataset gempa** dengan kolom: `time`, `latitude`, `longitude`, `depth`, `mag` (Magnitudo). Diproses dari sumber USGS/BMKG.                                                              |
| `kmeans_model.joblib`          | **Model machine learning** yang sudah fit terhadap data. Digunakan untuk prediksi cluster pada data baru.                                                                                 |
| `scaler.joblib`                | **MinMaxScaler** yang dilatih pada data training. Penting untuk normalisasi data sebelum prediksi.                                                                                        |
| `cluster_remap.pkl`            | **Dictionary** mapping cluster ID asli → ID baru + label risiko (Low/Moderate/High/Very High).                                                                                            |
| `cluster_info.csv`             | **Metadata cluster** berisi: centroid (mag, depth), severity_score, dan label risiko. Digunakan di Streamlit.                                                                             |
| `earthquakes_with_cluster.csv` | **Dataset lengkap** dengan kolom cluster dan risk category. Hasil akhir dari training.                                                                                                    |

---

## � Instalasi & Setup

### Prerequisites

- **Python 3.8+** (Disarankan 3.10+)
- **pip** atau **conda** untuk package management
- **Git** (opsional, untuk clone repository)

### Langkah-Langkah Instalasi

#### 1️⃣ Clone atau Download Project

```bash
# Jika menggunakan git
git clone <repository-url>
cd KlasifikasiGempa

# Atau download file ZIP dan extract
```

#### 2️⃣ Buat Virtual Environment (Direkomendasikan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📖 Cara Menggunakan

### A. Menjalankan Analisis di Jupyter Notebook

```bash
jupyter notebook earthquake_clustering.ipynb
```

Notebook akan:

1. ✅ Membaca data dari `katalog_gempa_v2.tsv`
2. ✅ Melakukan exploratory analysis
3. ✅ Training K-Means dengan optimal K detection
4. ✅ Membuat visualisasi hasil clustering
5. ✅ Menyimpan model ke `kmeans_model.joblib` dan `scaler.joblib`
6. ✅ Export dataset dengan cluster ke `earthquakes_with_cluster.csv`

**Catatan:** Jalankan semua cell secara berurutan. Model akan disimpan otomatis di akhir notebook.

### B. Menjalankan Aplikasi Web (Streamlit)

Setelah model selesai ditraining, jalankan:

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di `http://localhost:8501` dan menyediakan:

- 🗺️ Peta interaktif dengan filter
- 📊 Visualisasi cluster
- 🔍 Eksplorasi data lengkap
- 🤖 Prediksi cluster untuk gempa baru

---

## 📊 Dataset & Pre-processing

### Input Data: `katalog_gempa_v2.tsv`

Kolom yang digunakan:

- `time` - Tanggal/waktu gempa (UTC)
- `latitude` - Koordinat lintang
- `longitude` - Koordinat bujur
- `depth` - Kedalaman gempa (km)
- `mag` - Magnitudo (skala Richter)

**Jumlah Data:** ~12,000+ gempa

### Pre-processing Steps:

1. ✅ Load TSV dan rename kolom
2. ✅ Konversi tipe data (datetime, numeric)
3. ✅ Drop rows dengan data missing
4. ✅ **Normalisasi** menggunakan MinMaxScaler (range 0-1)
5. ✅ **Feature Selection:** Hanya gunakan `mag` dan `depth` (tidak lokasi)

**Alasan feature selection:**

- Magnitude & Depth adalah faktor risiko gempa
- Lokasi (lat/lon) dihilangkan untuk menghindari bias geografis
- Fokus pada karakteristik fisik gempa

---

## 🔬 Metodologi Machine Learning

### Algoritma: K-Means Clustering

**Hyperparameters:**

- `n_clusters`: Optimal K (auto-detected via Silhouette)
- `random_state`: 42 (reproducibility)
- `n_init`: 10 (multiple random initializations)
- `max_iter`: 300

### Evaluasi Model:

| Metrik                      | Interpretasi                                               |
| --------------------------- | ---------------------------------------------------------- |
| **Silhouette Score**        | Nilai -1 s/d 1. Semakin tinggi → cluster lebih terpisah.   |
| **Davies-Bouldin Index**    | Nilai 0+. Semakin rendah → cluster lebih optimal.          |
| **Calinski-Harabasz Score** | Nilai 0+. Semakin tinggi → cluster lebih padat & terpisah. |

### Penentuan K Optimal:

1. **Elbow Method** - Mencari "siku" pada grafik inertia
2. **Silhouette Score** - Memilih K dengan silhouette tertinggi
3. **Visual Inspection** - Manual review hasil clustering

---

## 🔢 Severity Score & Risk Classification

### Formula Severity Score:

```
norm_magnitude = (magnitude - min_mag) / (max_mag - min_mag)
norm_depth = (depth - min_depth) / (max_depth - min_depth)
inv_depth = 1 - norm_depth  # Gempa dangkal lebih berbahaya

severity_score = (0.7 × norm_magnitude) + (0.3 × inv_depth)
```

**Bobot:**

- 70% → Magnitudo (faktor utama)
- 30% → Shallow Depth (gempa dangkal lebih berbahaya)

### Risk Level Assignment:

Cluster diurutkan berdasarkan severity_score, kemudian diberi label:

- **0** = 🟢 **LOW** - Severity Score terendah
- **1** = 🟡 **MODERATE**
- **2** = 🟠 **HIGH**
- **3** = 🔴 **VERY HIGH** - Severity Score tertinggi

_(Untuk k ≠ 4, label dinamis: Level 1, Level 2, dst)_

---

## 📈 Output Files

## 📈 Output Files

Setelah menjalankan notebook, file berikut akan dihasilkan:

| File                           | Deskripsi                                              |
| ------------------------------ | ------------------------------------------------------ |
| `kmeans_model.joblib`          | Model K-Means terlatih (binary format)                 |
| `scaler.joblib`                | MinMaxScaler untuk normalisasi (binary format)         |
| `cluster_remap.pkl`            | Dictionary mapping cluster → label risiko              |
| `cluster_info.csv`             | Metadata cluster: centroid, severity_score, label      |
| `earthquakes_with_cluster.csv` | Dataset original + kolom `cluster` dan `Risk Category` |

### Contoh Output `cluster_info.csv`:

```csv
cluster,mag,depth,latitude,longitude,severity_score,label
0,3.45,150.5,-6.2,106.8,0.1234,Low
1,4.50,80.2,-7.5,107.3,0.4567,Moderate
2,5.20,45.8,-8.1,108.5,0.7234,High
3,6.10,25.3,-9.2,109.7,0.9123,Very High
```

---

## 🧪 Testing & Debugging

### Error Umum & Solusi:

#### 1. ❌ `FileNotFoundError: katalog_gempa_v2.tsv`

```
Solusi: Pastikan file TSV ada di folder yang sama dengan notebook.
```

#### 2. ❌ `ModuleNotFoundError: No module named 'plotly'`

```
Solusi: pip install plotly pandas scikit-learn joblib
```

#### 3. ❌ Streamlit tidak menemukan `kmeans_model.joblib`

```
Solusi: Jalankan notebook terlebih dahulu untuk generate model files.
        Pastikan streamlit_app.py di folder yang sama.
```

#### 4. ❌ Prediksi memberikan hasil anomali

```
Solusi: Periksa bahwa nilai input (magnitude, depth) masuk dalam range data training.
        Contoh range magnitudo: 1.5 - 7.5 | range depth: 5 - 600 km
```

---

## 📚 Libraries Utama

Referensi lengkap ada di `requirements.txt`:

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine Learning (KMeans, StandardScaler, Metrics)
- **joblib** - Model serialization
- **plotly** - Interactive visualization
- **matplotlib** & **seaborn** - Static plots
- **streamlit** - Web app framework
- **tqdm** - Progress bar

---

## 🎓 Topik Pembelajaran

Proyek ini mencakup konsep:

1. **Data Mining** - Preprocessing, feature engineering, EDA
2. **Clustering** - K-Means, Silhouette analysis, Elbow method
3. **ML Pipeline** - Training, evaluation, model serialization
4. **Visualization** - Interactive maps, statistical plots
5. **Web Development** - Streamlit app with realtime filtering
6. **Domain Knowledge** - Seismology, earthquake risk assessment

---

## 🔗 Referensi & Sumber Data

- **Earthquake Data Source:**
    - USGS Earthquake Hazards Program
    - BMKG (Badan Meteorologi, Klimatologi, dan Geofisika Indonesia)

- **Machine Learning Documentation:**
    - [scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

- **Visualization:**
    - [Plotly Python](https://plotly.com/python/)
    - [Streamlit Docs](https://docs.streamlit.io/)

---

## 💡 Tips & Best Practices

✅ **DO:**

- Selalu jalankan notebook dari awal untuk memastikan model di-update
- Gunakan virtual environment untuk menghindari dependency conflicts
- Dokumentasikan setiap perubahan di `notes/PENJELASAN_PERUBAHAN.txt`
- Test prediksi dengan nilai yang masuk range data training

❌ **DON'T:**

- Jangan manual edit file `.joblib` (binary format)
- Jangan skip normalisasi data (critical untuk K-Means)
- Jangan langsung menjalankan `streamlit_app.py` tanpa model files
- Jangan upload dataset besar di Git (gunakan `.gitignore`)

---

## 🤝 Kontribusi & Pengembangan Lanjutan

### Ide Pengembangan:

- [ ] Implementasi algoritma clustering lain (DBSCAN, Hierarchical)
- [ ] Advanced visualization (3D clustering, interactive heatmap)
- [ ] Feature engineering tambahan (tsunami risk, historical data)
- [ ] API endpoint untuk external integration
- [ ] Dashboard comparison antar versi model
- [ ] Automated model retraining pipeline
- [ ] Time-series forecasting untuk prediksi gempa

### Cara Berkontribusi:

1. Fork repository
2. Buat branch baru: `git checkout -b feature/nama-fitur`
3. Commit changes: `git commit -m "Deskripsi perubahan"`
4. Push ke branch: `git push origin feature/nama-fitur`
5. Buat Pull Request dengan deskripsi jelas

---

## 📝 Changelog & Versi

**v1.0** (Current)

- ✅ K-Means clustering dengan auto K-detection
- ✅ Severity scoring dan risk classification
- ✅ Jupyter notebook analysis
- ✅ Streamlit web app
- ✅ Interactive visualization

**v0.9** (Previous)

- Basic K-Means implementation
- Manual K selection

Lihat file `notes/PENJELASAN_PERUBAHAN.txt` untuk detail perubahan.

---

## ⚖️ License & Attribution

Proyek ini adalah bagian dari **Tugas Data Mining - Kelompok Kelas A11.43UG1, DINUS**.

Digunakan untuk keperluan pendidikan. Data gempa bersumber dari USGS/BMKG (public domain).

---

## 📞 Kontak & Support

Untuk pertanyaan atau issue:

- 📧 Email: 111202415694@mhs.dinus.ac.id
- 💬 GitHub Issues: [buka issue di repo]
- 📋 Notes Tambahan: Lihat folder `notes/`

---

**Dibuat dengan ❤️ menggunakan Python, Streamlit, Scikit-Learn, dan Passion untuk Data Science.**

🌍 _Terima kasih telah menggunakan Aplikasi Klasifikasi Gempa!_
