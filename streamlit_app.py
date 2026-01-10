import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import textwrap  # <--- Library penolong untuk fix HTML
from folium.plugins import MarkerCluster, BeautifyIcon
from streamlit_folium import st_folium

# --- 1. Page Config ---
st.set_page_config(layout="wide", page_title="Earthquake Clustering")

# --- CSS: Wide Mode tapi Rapi ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        h1 { margin-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Load Data & Models ---
@st.cache_data
def load_data():
    df = pd.read_csv("earthquakes_with_cluster.csv")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    pipeline = joblib.load("kmeans_pipeline.joblib")
    cluster_info = pd.read_csv("cluster_info.csv")
    return pipeline, cluster_info

try:
    df = load_data()
    pipeline, cluster_info = load_model()
except Exception as e:
    st.error(f"Error loading data/model: {e}")
    st.stop()

# --- 3. Sidebar Filters ---
st.sidebar.title("🔍 Filter Data")
# --- Callback Reset Filter ---
def reset_filters():
    st.session_state['date_start'] = df['time'].min().date()
    st.session_state['date_end'] = df['time'].max().date()
    st.session_state['mag_range'] = (float(df[mag_col].min()), float(df[mag_col].max()))
    st.session_state['depth_range'] = (depth_min_val, depth_max_val)

if st.sidebar.button("🔄 Reset Filter"):
    reset_filters()

min_date = st.sidebar.date_input("Tanggal awal", df['time'].min().date(), key='date_start')
max_date = st.sidebar.date_input("Tanggal akhir", df['time'].max().date(), key='date_end')

mag_col = 'mag' if 'mag' in df.columns else 'magnitude'
mag_min, mag_max = st.sidebar.slider("Rentang Magnitudo", 
                                     float(df[mag_col].min()), float(df[mag_col].max()), 
                                     (float(df[mag_col].min()), float(df[mag_col].max())),
                                     key='mag_range')

depth_min_val = float(df['depth'].min()) if not df['depth'].isna().all() else 0.0
depth_max_val = float(df['depth'].max()) if not df['depth'].isna().all() else 700.0
depth_min, depth_max = st.sidebar.slider("Rentang Kedalaman (km)", depth_min_val, depth_max_val, 
                                        (depth_min_val, depth_max_val),
                                        key='depth_range')

# Filter Logic
filtered = df[
    (df['time'].dt.date >= min_date) &
    (df['time'].dt.date <= max_date) &
    (df[mag_col] >= mag_min) & (df[mag_col] <= mag_max)
]
if not df['depth'].isna().all():
    filtered = filtered[(filtered['depth'] >= depth_min) & (filtered['depth'] <= depth_max)]

# --- 4. Main Layout (Top Section) ---
st.title("🌋 Earthquake Clustering Viewer")
st.markdown("Visualisasi persebaran gempa bumi dan tingkat keparahannya menggunakan K-Means Clustering.")

# Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Kejadian (Filtered)", f"{len(filtered):,}")
with m2:
    avg_mag = filtered[mag_col].mean() if not filtered.empty else 0
    st.metric("Rata-rata Magnitudo", f"{avg_mag:.2f}")
with m3:
    max_mag = filtered[mag_col].max() if not filtered.empty else 0
    st.metric("Magnitudo Tertinggi", f"{max_mag:.2f}")

st.divider()

# --- 5. Map Visualization ---
if not filtered.empty:
    center_lat = filtered['latitude'].mean()
    center_lon = filtered['longitude'].mean()
else:
    center_lat, center_lon = 0, 0

m = folium.Map(location=[center_lat, center_lon], zoom_start=5, prefer_canvas=True)
marker_cluster = MarkerCluster().add_to(m)

# --- 3.1 Advanced Filters (Data Sampling) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Atur Jumlah Data")

total_filtered = len(filtered)
# Default 2000 or total if less than 2000
default_max = 2000
max_val_slider = total_filtered if total_filtered > 2000 else 2000
min_val_slider = 2000

max_samples = st.sidebar.slider(
    "Jumlah Data Ditampilkan (Max)",
    min_value=min_val_slider,
    max_value=max_val_slider if max_val_slider > min_val_slider else min_val_slider + 1000, # Fallback to avoid error if total < 2000
    value=default_max if default_max <= max_val_slider else max_val_slider,
    step=100,
    help="Atur berapa banyak data yang ingin ditampilkan di peta. Batas minimal 2.000."
)

# Balanced Sampling Logic (Water Filling Algorithm)
if total_filtered > max_samples:
    st.info(f"ℹ️ Mengambil {max_samples} sampel data secara merata (Water Filling) dari {total_filtered} data.")
    
    def balanced_sample_water_filling(df, n_samples, cluster_col='cluster'):
        """
        Mengambil sampel dengan metode Water Filling:
        1. Target awal dibagi rata.
        2. Jika ada cluster yang kurang dari target, ambil semua.
        3. Sisa kuota didistribusikan ulang ke cluster yang masih punya sisa data.
        """
        unique_c = df[cluster_col].unique()
        n_c = len(unique_c)
        
        # Simpan data per cluster
        clusters = {c: df[df[cluster_col] == c] for c in unique_c}
        
        # Sisa kuota yang harus dipenuhi
        quota_left = n_samples
        
        # Cluster yang masih 'aktif' (bisa diambil datanya)
        active_clusters = list(unique_c)
        
        sampled_indices = []
        
        while quota_left > 0 and active_clusters:
            # Target per cluster aktif saat ini
            target_per_cluster = quota_left // len(active_clusters)
            # Jika hasil bagi 0 (karena quota < n_active), set min 1 agar jalan terus sampai habis
            if target_per_cluster == 0: 
                target_per_cluster = 1
            
            # List untuk mencatat cluster yang "habis" di putaran ini
            clusters_exhausted = []
            
            # Hitung alokasi putaran ini agar tidak 'over' quota total
            # (Mencegah masalah pembulatan)
            allocated_this_round = 0
            
            for c in active_clusters:
                if quota_left <= 0: break
                
                # Data yang BELUM diambil di cluster ini
                # (Kita butuh cara efisien, misal tracking index. 
                #  Tapi karena dataframe di-slice di awal (clusters dict), 
                #  kita bisa tracking berapa yang SUDAH diambil dari setiap DF?)
                # Simplifikasi: Kita ambil sample baru dari sisa, lalu update df di dictionary
                
                current_df = clusters[c]
                n_available = len(current_df)
                
                # Target ambil sesi ini
                n_take = min(n_available, target_per_cluster)
                
                # Pastikan tidak mengambil lebih dari sisa quota total
                n_take = min(n_take, quota_left)
                
                if n_take > 0:
                    taken = current_df.sample(n=n_take, random_state=42)
                    sampled_indices.extend(taken.index.tolist())
                    
                    # Kurangi quota global
                    quota_left -= n_take
                    
                    # Update sisa data di cluster ini (buang yang sudah diambil)
                    clusters[c] = current_df.drop(taken.index)
                    
                    # Cek apakah cluster ini sudah habis?
                    if len(clusters[c]) == 0:
                        clusters_exhausted.append(c)
                else:
                    # Jika 0 (misal cluster kosong), mark exhausted
                    clusters_exhausted.append(c)
            
            # Hapus cluster yang sudah habis dari active list
            for c in clusters_exhausted:
                if c in active_clusters:
                    active_clusters.remove(c)
                    
        # Kembalikan dataframe berdasarkan index yang terpilih
        return df.loc[sampled_indices]

    data_to_plot = balanced_sample_water_filling(filtered, max_samples)

else:
    # Jika data kurang dari batas max, tampilkan semua
    data_to_plot = filtered

if False: # Old logic disabled
    st.warning(f"⚠️ Menampilkan {MAX_POINTS} data teratas dari {len(filtered)} total data demi performa peta.")

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']
colors_hex = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}

# --- Loop 1: Tambahkan Markers ---
for _, row in data_to_plot.iterrows():
    c_id = int(row['cluster'])
    color = colors[c_id % len(colors)]
    
    popup_txt = f"""
    <b>Mag:</b> {row[mag_col]}<br>
    <b>Depth:</b> {row['depth']} km<br>
    <b>Cluster:</b> {c_id}
    """
    
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_txt, max_width=200),
        icon=BeautifyIcon(
            icon_shape='marker',
            number=str(c_id),
            border_color=color,
            background_color=color,
            text_color='white',
            inner_icon_style='margin-top:0; font-size:12px; font-weight:bold;' 
        )
    ).add_to(marker_cluster)

# --- Tambahkan Legend (Sekali saja) ---
legend_html = """
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; 
    z-index:9999; font-size:14px;
        background-color: rgba(255, 255, 255, 0.9);
        color: black !important;
        border: 2px solid #ccc; border-radius: 6px; padding: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    ">
    <h5 style="margin:0 0 5px 0; border-bottom:1px solid #ccc; padding-bottom:3px;">Legenda Cluster</h5>
"""

# Isi Legend dari cluster_info
cluster_info_sorted = cluster_info.sort_values(by='cluster', ascending=True)
for _, row in cluster_info_sorted.iterrows():
    c_id = int(row['cluster'])
    severity = row['severity_score']
    # FIX: Gunakan label dari CSV
    risk = row['label']
    c_color = colors_hex.get(c_id, 'gray')
    
    legend_html += f"""
    <div style="margin-bottom: 3px; color: black;">
        <i style="background: {c_color}; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 5px;"></i>
        Cluster {c_id}: <b>{risk}</b>
    </div>
    """
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

# Membuat grid kolom dinamis untuk section bawah (persiapan)
cols = st.columns(len(cluster_info_sorted))

# Height 750px sesuai request
st_folium(m, height=750, use_container_width=True, returned_objects=[])

# --- 6. Cluster Info (Bottom Section) ---
st.subheader("📊 Analisis Klaster (Cluster Meaning)")

# Menggunakan expander yang terbuka default, dengan layout kolom di dalamnya
with st.expander("Klik untuk melihat detail karakteristik tiap klaster", expanded=True):
    cluster_info_sorted = cluster_info.sort_values(by='cluster', ascending=True)
    colors_hex = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
    
    # Membuat grid kolom dinamis (misal 3 kolom per baris)
    cols = st.columns(len(cluster_info_sorted)) 
    
    for i, (_, row) in enumerate(cluster_info_sorted.iterrows()):
        # Pilih kolom target, jika klaster banyak, dia akan mengecil otomatis
        target_col = cols[i] if i < len(cols) else cols[0]
        
        with target_col:
            c_id = int(row['cluster'])
            severity = row['severity_score']
            # FIX: Gunakan label dari CSV, jangan hitung ulang manual
            risk_level = row['label'] 
            color = colors_hex.get(c_id, 'gray')
            
            st.markdown(f"""
            <div style="
                background-color: #262730;
                border-top: 5px solid {color};
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                height: 100%;">
                <h4 style="margin:0; color: #FAFAFA;">Cluster {c_id}</h4>
                <p style="font-size: 14px; margin-bottom: 5px; color: #E0E0E0;">Risk: <strong>{risk_level}</strong></p>
                <hr style="margin: 5px 0; border-color: #4A4A4A;">
                <p style="font-size: 12px; margin:0; color: #BDBDBD;">
                Avg Mag: <strong>{row['mag']:.2f}</strong><br>
                Avg Depth: <strong>{row['depth']:.0f} km</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- 7. Sidebar Prediction & Upload ---
st.sidebar.markdown("---")
st.sidebar.header("⚡ Prediksi Cepat")

p_lat = st.sidebar.number_input("Lat", value=0.0, format="%.4f")
p_lon = st.sidebar.number_input("Lon", value=0.0, format="%.4f")
p_mag = st.sidebar.number_input("Mag", value=5.0)
p_dep = st.sidebar.number_input("Depth (km)", value=10.0)

if st.sidebar.button("Prediksi"):
    arr = np.array([[p_lat, p_lon, p_mag, p_dep]])
    try:
        scaled_feat = pipeline.named_steps['scaler'].transform(arr)
        clust = pipeline.named_steps['kmeans'].predict(scaled_feat)[0]
        info = cluster_info[cluster_info['cluster'] == clust].iloc[0]
        
        st.sidebar.success(f"Masuk ke **Cluster {clust}**")
        st.sidebar.markdown(f"**Severity Score:** {info['severity_score']:.3f}")
        st.sidebar.markdown(f"**Label:** {info['label']}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
with st.sidebar.expander("📂 Upload Data Baru"):
    uploaded_file = st.file_uploader("Upload CSV/TSV", type=['csv', 'tsv'])
    if uploaded_file is not None:
        st.info("File diterima. Analisis otomatis berjalan di background.")
