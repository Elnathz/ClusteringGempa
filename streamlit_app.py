import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Earthquake Clustering v2")

st.title("🌋 Earthquake Clustering & Severity Viewer v2")

# --- 1. Load Data & Models ---
@st.cache_data
def load_data():
    # Perbaikan: Menggunakan 'time' bukan 'datetime' sesuai error log
    df = pd.read_csv("earthquakes_with_cluster.csv")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    pipeline = joblib.load("kmeans_pipeline.joblib")
    # Menggunakan nama variabel konsisten
    cluster_info = pd.read_csv("cluster_info.csv")
    return pipeline, cluster_info

# Load awal
try:
    df = load_data()
    pipeline, cluster_info = load_model()
except Exception as e:
    st.error(f"Error loading data/model: {e}")
    st.stop()

# --- 2. Sidebar Filters (Main Data) ---
st.sidebar.header("Filter Data Utama")
min_date = st.sidebar.date_input("Tanggal awal", df['time'].min().date())
max_date = st.sidebar.date_input("Tanggal akhir", df['time'].max().date())

# Handle kolom 'mag' vs 'magnitude'
mag_col = 'mag' if 'mag' in df.columns else 'magnitude'
mag_min, mag_max = st.sidebar.slider("Rentang Magnitudo", 
                                     float(df[mag_col].min()), float(df[mag_col].max()), 
                                     (float(df[mag_col].min()), float(df[mag_col].max())))

# Handle depth
depth_min_val = float(df['depth'].min()) if not df['depth'].isna().all() else 0.0
depth_max_val = float(df['depth'].max()) if not df['depth'].isna().all() else 700.0
depth_min, depth_max = st.sidebar.slider("Rentang Kedalaman (km)", depth_min_val, depth_max_val, (depth_min_val, depth_max_val))

# Apply Filter
filtered = df[
    (df['time'].dt.date >= min_date) &
    (df['time'].dt.date <= max_date) &
    (df[mag_col] >= mag_min) & (df[mag_col] <= mag_max)
]
if not df['depth'].isna().all():
    filtered = filtered[(filtered['depth'] >= depth_min) & (filtered['depth'] <= depth_max)]

st.markdown(f"Menampilkan **{len(filtered)}** data gempa (Main Dataset)")

# --- 3. Main Map Visualization (Folium + MarkerCluster) ---
# Menggunakan Folium agar fitur 'Zoom Out = Mengumpul' berfungsi
if not filtered.empty:
    center_lat = filtered['latitude'].mean()
    center_lon = filtered['longitude'].mean()
else:
    center_lat, center_lon = 0, 0

m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

# Batasi tampilan di peta agar tidak berat (Max 2000 titik)
MAX_POINTS = 2000
data_to_plot = filtered.head(MAX_POINTS)
if len(filtered) > MAX_POINTS:
    st.caption(f"⚠️ Peta hanya menampilkan {MAX_POINTS} data pertama demi performa. Filter data untuk hasil lebih spesifik.")

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']

for _, row in data_to_plot.iterrows():
    c_id = int(row['cluster'])
    color = colors[c_id % len(colors)]
    
    # Tooltip info
    popup_txt = f"""
    <b>Mag:</b> {row[mag_col]}<br>
    <b>Depth:</b> {row['depth']} km<br>
    <b>Cluster:</b> {c_id}
    """
    
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_txt, max_width=200),
        icon=folium.Icon(color=color, icon='info-sign'),
    ).add_to(marker_cluster)

st_folium(m, width=1000, height=500)

# Info Cluster
st.subheader("Informasi Pusat Klaster (Centroids)")
st.dataframe(cluster_info)

st.divider()

# --- 4. Sidebar: Predict Single Point ---
st.sidebar.markdown("---")
st.sidebar.header("Prediksi Satu Titik")
p_lat = st.sidebar.number_input("Lat", value=0.0)
p_lon = st.sidebar.number_input("Lon", value=0.0)
p_mag = st.sidebar.number_input("Mag", value=5.0)
p_dep = st.sidebar.number_input("Depth (km)", value=10.0)

if st.sidebar.button("Prediksi Titik"):
    # Fitur harus urut: lat, lon, mag, depth (sesuai training)
    arr = np.array([[p_lat, p_lon, p_mag, p_dep]])
    try:
        clust = pipeline.named_steps['kmeans'].predict(pipeline.named_steps['scaler'].transform(arr))[0]
        # Ambil info dari tabel cluster_info
        info = cluster_info[cluster_info['cluster'] == clust].iloc[0].to_dict()
        st.sidebar.success(f"Hasil: Klaster {clust}")
        st.sidebar.json(info)
    except Exception as e:
        st.sidebar.error(f"Gagal memprediksi: {e}")

# --- 5. Sidebar: Upload New Data (Fitur Baru) ---
st.sidebar.markdown("---")
st.sidebar.header("Upload Data Baru (Batch)")
uploaded_file = st.sidebar.file_uploader("Upload CSV/TSV", type=['csv', 'tsv'])

if uploaded_file is not None:
    st.header("📂 Analisis Data Upload")
    sep = '\t' if uploaded_file.name.endswith('.tsv') else ','
    
    try:
        new_df = pd.read_csv(uploaded_file, sep=sep)
        
        # Normalisasi nama kolom agar sesuai model (magnitude -> mag)
        new_df.columns = [c.lower() for c in new_df.columns]
        if 'magnitude' in new_df.columns:
            new_df.rename(columns={'magnitude': 'mag'}, inplace=True)
            
        required_cols = ['latitude', 'longitude', 'mag', 'depth']
        
        # Cek kelengkapan kolom
        if not all(col in new_df.columns for col in required_cols):
            st.error(f"File wajib memiliki kolom (atau variannya): {', '.join(required_cols)}")
        else:
            # Lakukan Prediksi
            features = new_df[required_cols]
            # Handle NaN
            features = features.fillna(0) 
            
            scaled = pipeline.named_steps['scaler'].transform(features)
            new_df['cluster'] = pipeline.named_steps['kmeans'].predict(scaled)
            
            # Map severity info (menggunakan merge agar lebih aman)
            new_df = new_df.merge(cluster_info[['cluster', 'label', 'severity_score']], on='cluster', how='left')
            
            st.success("Klasifikasi Selesai!")
            st.dataframe(new_df.head())
            
            # Visualisasi Data Upload (Simple Folium Map)
            if st.checkbox("Tampilkan Peta Data Upload"):
                m_new = folium.Map(location=[new_df['latitude'].mean(), new_df['longitude'].mean()], zoom_start=4)
                mc_new = MarkerCluster().add_to(m_new)
                
                # Plot max 1000 data upload
                for _, row in new_df.head(1000).iterrows():
                    folium.Marker(
                        [row['latitude'], row['longitude']],
                        popup=f"Cluster: {row['cluster']}<br>Label: {row.get('label', '-')}",
                        icon=folium.Icon(color='blue')
                    ).add_to(mc_new)
                
                st_folium(m_new, width=1000, height=500, key="new_data_map")
                
    except Exception as e:
        st.error(f"Error memproses file: {e}")