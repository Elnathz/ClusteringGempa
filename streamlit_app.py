
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="Earthquake Clustering Viewer")

st.title("Earthquake Clustering & Severity Viewer")

# Load data & models
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
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
min_date = st.sidebar.date_input("Start date", value=df['time'].min().date())
max_date = st.sidebar.date_input("End date", value=df['time'].max().date())
mag_min, mag_max = st.sidebar.slider("Magnitude range", float(df['mag'].min()), float(df['mag'].max()), (float(df['mag'].min()), float(df['mag'].max())))
depth_min, depth_max = st.sidebar.slider("Depth range (km)", float(df['depth'].min(skipna=True) if not df['depth'].isna().all() else 0), float(df['depth'].max(skipna=True) if not df['depth'].isna().all() else 700), (float(np.nanmin(df['depth'].dropna())) if not df['depth'].isna().all() else 0, float(np.nanmax(df['depth'].dropna())) if not df['depth'].isna().all() else 700))
lat_min, lat_max = st.sidebar.number_input("Latitude min", value=float(df['latitude'].min()), format="%.6f"), st.sidebar.number_input("Latitude max", value=float(df['latitude'].max()), format="%.6f")
lon_min, lon_max = st.sidebar.number_input("Longitude min", value=float(df['longitude'].min()), format="%.6f"), st.sidebar.number_input("Longitude max", value=float(df['longitude'].max()), format="%.6f")

if st.sidebar.button("Apply filters"):
    st.session_state['apply'] = True

apply_filters = st.sidebar.button("Apply (quick)")  # keep intuitive access
# Apply filters immediately
filtered = df[
    (df['time'].dt.date >= min_date) &
    (df['time'].dt.date <= max_date) &
    (df['mag'] >= mag_min) & (df['mag'] <= mag_max) &
    (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
    (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
]

# depth filter only if available
if not df['depth'].isna().all():
    filtered = filtered[(filtered['depth'] >= depth_min) & (filtered['depth'] <= depth_max)]

st.markdown(f"**Showing {len(filtered)} events** after filters")

# Map visualization
fig = px.scatter_mapbox(filtered, lat='latitude', lon='longitude', color='cluster',
                        hover_data=['time','mag','depth'], zoom=4, height=600, mapbox_style='open-street-map')
st.plotly_chart(fig, use_container_width=True)

# Cluster summary
st.header("Cluster summary")
st.dataframe(cluster_info)

# Lookup a point
st.sidebar.header("Lookup location / Predict severity")
lat = st.sidebar.number_input("Latitude", value=0.0, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=0.0, format="%.6f")
mag = st.sidebar.number_input("Assumed magnitude (for point)", value=5.0, format="%.2f")
depth = st.sidebar.number_input("Assumed depth (km)", value=10.0, format="%.2f")

if st.sidebar.button("Predict severity for point"):
    # Prepare feature vector according to cluster_info columns
    feat_cols = ['latitude','longitude','mag']
    if 'depth' in filtered.columns and not filtered['depth'].isna().all():
        feat_cols.append('depth')
    feat = []
    for c in feat_cols:
        if c=='latitude': feat.append(lat)
        elif c=='longitude': feat.append(lon)
        elif c=='mag': feat.append(mag)
        elif c=='depth': feat.append(depth)
    arr = np.array(feat).reshape(1,-1)
    # Use pipeline to predict
    cluster_idx = pipeline.named_steps['kmeans'].predict(pipeline.named_steps['scaler'].transform(arr))[0]
    info_row = cluster_info[cluster_info['cluster']==cluster_idx].iloc[0].to_dict()
    st.success(f"Predicted cluster: {cluster_idx} — Label: {info_row.get('label','N/A')} — Severity score: {info_row.get('severity_score',np.nan):.3f}")
    st.json(info_row)

st.markdown("---")
st.markdown("**Notes:** The severity label is based on cluster centroids (magnitude + depth). Use with caution — this is exploratory clustering, not a formal hazard model.")
