import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(layout="wide", page_title="Earthquake Clustering")

st.title("🌋 Earthquake Clustering & Severity Viewer")

@st.cache_data
def load_data():
    df = pd.read_csv("earthquakes_with_cluster.csv")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    pipeline = joblib.load("kmeans_pipeline.joblib")
    centroids = pd.read_csv("cluster_info.csv")
    return pipeline, centroids

df = load_data()
pipeline, centroids = load_model()

# Sidebar filters
st.sidebar.header("Filter Data")
min_date = st.sidebar.date_input("Tanggal awal", df['datetime'].min().date())
max_date = st.sidebar.date_input("Tanggal akhir", df['datetime'].max().date())
mag_min, mag_max = st.sidebar.slider("Rentang Magnitudo", float(df['magnitude'].min()), float(df['magnitude'].max()), (float(df['magnitude'].min()), float(df['magnitude'].max())))
depth_min, depth_max = st.sidebar.slider("Rentang Kedalaman (km)", float(df['depth'].min()), float(df['depth'].max()), (float(df['depth'].min()), float(df['depth'].max())))

filtered = df[
    (df['datetime'].dt.date >= min_date) &
    (df['datetime'].dt.date <= max_date) &
    (df['magnitude'].between(mag_min, mag_max)) &
    (df['depth'].between(depth_min, depth_max))
]

st.markdown(f"Menampilkan **{len(filtered)}** data setelah filter")

# Map visualization
fig = px.scatter_mapbox(filtered, lat='latitude', lon='longitude', color='cluster',
                        hover_data=['datetime','magnitude','depth','location'],
                        zoom=4, height=600, mapbox_style='open-street-map')
st.plotly_chart(fig, use_container_width=True)

# Cluster info
st.subheader("Informasi Klaster")
st.dataframe(centroids)

# Prediction for new input
st.sidebar.header("Prediksi Keparahan Baru")
lat = st.sidebar.number_input("Latitude", value=0.0)
lon = st.sidebar.number_input("Longitude", value=0.0)
mag = st.sidebar.number_input("Magnitudo", value=5.0)
dep = st.sidebar.number_input("Kedalaman (km)", value=10.0)

if st.sidebar.button("Prediksi"):
    arr = np.array([[lat, lon, mag, dep]])
    clust = pipeline.named_steps['kmeans'].predict(pipeline.named_steps['scaler'].transform(arr))[0]
    info = centroids[centroids['cluster']==clust].iloc[0].to_dict()
    st.success(f"Klaster {clust}: {info['severity_label']} (skor {info['severity_score']:.3f})")

# Upload new data for clustering
st.sidebar.header("Upload Data Baru untuk Clustering")
uploaded_file = st.sidebar.file_uploader("Upload file CSV atau TSV", type=['csv', 'tsv'])

if uploaded_file is not None:
    # Determine separator
    sep = '\t' if uploaded_file.name.endswith('.tsv') else ','
    try:
        new_df = pd.read_csv(uploaded_file, sep=sep)
        st.sidebar.success("File berhasil diupload!")
        
        # Check required columns
        required_cols = ['latitude', 'longitude', 'magnitude', 'depth']
        if not all(col in new_df.columns for col in required_cols):
            st.sidebar.error(f"File harus memiliki kolom: {', '.join(required_cols)}")
        else:
            # Predict clusters
            features = new_df[required_cols]
            scaled_features = pipeline.named_steps['scaler'].transform(features)
            new_df['cluster'] = pipeline.named_steps['kmeans'].predict(scaled_features)
            
            # Add severity info
            new_df['severity_label'] = new_df['cluster'].map(centroids.set_index('cluster')['severity_label'])
            new_df['severity_score'] = new_df['cluster'].map(centroids.set_index('cluster')['severity_score'])
            
            st.subheader("Data Baru dengan Klaster")
            st.dataframe(new_df)
            
            # Optional: Visualize on map
            if st.checkbox("Tampilkan di peta"):
                fig_new = px.scatter_mapbox(new_df, lat='latitude', lon='longitude', color='cluster',
                                            hover_data=['magnitude','depth','severity_label'],
                                            zoom=4, height=600, mapbox_style='open-street-map')
                st.plotly_chart(fig_new, use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Error membaca file: {e}")
