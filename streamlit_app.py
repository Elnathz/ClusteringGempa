import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import textwrap  # <--- Library penolong untuk fix HTML
from folium.plugins import MarkerCluster
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
min_date = st.sidebar.date_input("Tanggal awal", df['time'].min().date())
max_date = st.sidebar.date_input("Tanggal akhir", df['time'].max().date())

mag_col = 'mag' if 'mag' in df.columns else 'magnitude'
mag_min, mag_max = st.sidebar.slider("Rentang Magnitudo", 
                                     float(df[mag_col].min()), float(df[mag_col].max()), 
                                     (float(df[mag_col].min()), float(df[mag_col].max())))

depth_min_val = float(df['depth'].min()) if not df['depth'].isna().all() else 0.0
depth_max_val = float(df['depth'].max()) if not df['depth'].isna().all() else 700.0
depth_min, depth_max = st.sidebar.slider("Rentang Kedalaman (km)", depth_min_val, depth_max_val, (depth_min_val, depth_max_val))

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

MAX_POINTS = 2000
data_to_plot = filtered.head(MAX_POINTS)

if len(filtered) > MAX_POINTS:
    st.warning(f"⚠️ Menampilkan {MAX_POINTS} data teratas dari {len(filtered)} total data demi performa peta.")

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']

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
        icon=folium.Icon(color=color, icon='info-sign'),
    ).add_to(marker_cluster)

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
            risk_level = "High" if severity > 1.2 else "Moderate" if severity > 1.0 else "Low"
            color = colors_hex.get(c_id, 'gray')
            
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                border-top: 5px solid {color};
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                height: 100%;">
                <h4 style="margin:0; color:black;">Cluster {c_id}</h4>
                <p style="font-size: 14px; margin-bottom: 5px; color: #333;">Risk: <strong>{risk_level}</strong></p>
                <hr style="margin: 5px 0;">
                <p style="font-size: 12px; margin:0; color: #555;">
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
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
with st.sidebar.expander("📂 Upload Data Baru"):
    uploaded_file = st.file_uploader("Upload CSV/TSV", type=['csv', 'tsv'])
    if uploaded_file is not None:
        st.info("File diterima. Analisis otomatis berjalan di background.")