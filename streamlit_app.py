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
st.subheader("📊 Understanding the Earthquake Clusters")

with st.expander("What do these clusters mean?"):
    st.write("""
    Machine learning has grouped these earthquakes based on their physical characteristics. 
    Because this is 'Clustering', the model finds patterns on its own rather than 
    following strict human rules.
    """)
    
    # Mapping colors to cluster IDs for the legend
    colors_hex = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
    
    col1, col2 = st.columns(2)
    # cluster_info contains: latitude, longitude, mag, depth, cluster, severity_score
    for i, row in cluster_info.iterrows():
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            c_id = int(row['cluster'])
            severity = row['severity_score']
            
            # Risk labels based on the severity_score in your CSV
            risk_level = "High" if severity > 1.2 else "Moderate" if severity > 1.0 else "Low"
            
            st.markdown(f"""
            <div style="border-left: 5px solid {colors_hex.get(c_id, 'gray')}; padding-left: 10px; margin-bottom: 10px;">
                <strong>Cluster {c_id}</strong> (Risk Level: {risk_level})<br>
                Avg Magnitude: {row['mag']:.2f}<br>
                Avg Depth: {row['depth']:.1f} km<br>
                <em>Severity Score: {severity:.3f}</em>
            </div>
            """, unsafe_allow_html=True)

# --- 4. Sidebar: Predict Single Point ---
st.sidebar.markdown("---")
st.sidebar.header("Prediksi Satu Titik")
p_lat = st.sidebar.number_input("Lat", value=0.0)
p_lon = st.sidebar.number_input("Lon", value=0.0)
p_mag = st.sidebar.number_input("Mag", value=5.0)
p_dep = st.sidebar.number_input("Depth (km)", value=10.0)

# Replace the 'if st.sidebar.button("Prediksi Titik")' block:
if st.sidebar.button("Prediksi Titik"):
    # Features must be in order: lat, lon, mag, depth
    arr = np.array([[p_lat, p_lon, p_mag, p_dep]])
    try:
        # Use the pipeline steps: scaler then kmeans
        scaled_feat = pipeline.named_steps['scaler'].transform(arr)
        clust = pipeline.named_steps['kmeans'].predict(scaled_feat)[0]
        
        # Get the info for this specific cluster from cluster_info.csv
        info = cluster_info[cluster_info['cluster'] == clust].iloc[0]
        
        st.sidebar.success(f"📍 Prediction: Cluster {clust}")
        
        st.sidebar.markdown(f"""
        **Analysis:**
        This earthquake is categorized as **Cluster {clust}**. 
        Historically, events in this group have a severity score of **{info['severity_score']:.2f}**.
        """)
        
        # Public-facing education on depth
        if p_dep < 70:
            st.sidebar.warning("⚠️ **Shallow Earthquake:** These occur closer to the surface and are often felt more intensely.")
        else:
            st.sidebar.info("ℹ️ **Deep Earthquake:** These are often felt over a wider area but typically cause less surface damage.")
            
    except Exception as e:
        st.sidebar.error(f"Error making prediction: {e}")

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
            
            #Information above folium map 
            st.info(f"💡 **Map Tip:** The markers are color-coded by cluster. Zoom in to see individual events, or zoom out to see high-density 'hotspots' across the region.")

            # Add a progress/status bar for the 2000 point limit 
            if len(filtered) > MAX_POINTS:
                st.warning(f"Displaying the first {MAX_POINTS} out of {len(filtered)} earthquakes to maintain app speed. Use the sidebar filters to narrow down your search.")
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