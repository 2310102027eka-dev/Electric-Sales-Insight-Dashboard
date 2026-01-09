import os
import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from supabase import create_client, Client
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP KONFIGURASI & API ---
st.set_page_config(page_title="Dashboard Penjualan & AI Insights", layout="wide")

# Custom CSS untuk tampilan lebih modern
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Masalah Konfigurasi Secrets: {e}")
    st.stop()

# --- 2. SETUP SUPABASE & DATA ---
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

@st.cache_data(ttl=600)
def load_data():
    try:
        res = supabase.table("datapenjualanbaru").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return df
        
        if 'cancel_time' in df.columns:
            df['cancel_time'] = pd.to_datetime(df['cancel_time'], errors='coerce')
        
        cols_to_fix = ['order_amount', 'total_refund', 'original_price', 'total_discount']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Pastikan kolom wilayah ada (asumsi nama kolom adalah 'province' dan 'city')
        # Jika nama kolom berbeda di database Anda, silakan sesuaikan
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# --- 3. FUNGSI CLUSTERING (DIMODIFIKASI) ---
def get_clustering_labels(df):
    if df.empty or len(df) < 3:
        return None
    
    # Agregasi per SKU
    df_prod = df.groupby('platform_sku_variation').agg({
        'order_amount': 'sum',
        'total_refund': 'sum',
        'platform_sku_variation': 'count'
    }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()
    
    features = ['transaction_count', 'total_refund', 'order_amount']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_prod[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_prod['cluster_id'] = kmeans.fit_predict(x_scaled)
    
    # Labeling
    cluster_means = df_prod.groupby('cluster_id')[features].mean()
    idx_bermasalah = cluster_means['total_refund'].idxmax()
    rem = [i for i in range(3) if i != idx_bermasalah]
    idx_unggulan = cluster_means.loc[rem, 'transaction_count'].idxmax()
    idx_evaluasi = [i for i in range(3) if i not in [idx_bermasalah, idx_unggulan]][0]
    
    mapping = {
        idx_bermasalah: "Cluster 0 – Produk Bermasalah",
        idx_unggulan: "Cluster 1 – Produk Unggulan",
        idx_evaluasi: "Cluster 2 – Produk Evaluasi"
    }
    df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
    return df_prod[['platform_sku_variation', 'Cluster Name']]

# --- 4. EKSEKUSI DATA & FILTER ---
df_raw = load_data()

if df_raw.empty:
    st.warning("Data kosong.")
    st.stop()

# Tambahkan label cluster ke data mentah agar bisa difilter
df_cluster_map = get_clustering_labels(df_raw)
if df_cluster_map is not None:
    df_raw = df_raw.merge(df_cluster_map, on='platform_sku_variation', how='left')

# --- SIDEBAR ---
st.sidebar.header("🔍 Filter Data")

# Filter 1: Rentang Waktu
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_date = df_raw['cancel_time'].min().date()
    max_date = df_raw['cancel_time'].max().date()
    
    date_option = st.selectbox("Pilih Periode", ["Semua Waktu", "7 Hari Terakhir", "Bulan Ini", "Custom"])
    
    start_date, end_date = min_date, max_date
    if date_option == "7 Hari Terakhir":
        start_date = max_date - pd.Timedelta(days=7)
    elif date_option == "Bulan Ini":
        start_date = max_date.replace(day=1)
    elif date_option == "Custom":
        date_range = st.date_input("Rentang Tanggal", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = date_range

# Filter 2: Wilayah (Provinsi & Kota)
with st.sidebar.expander("📍 Wilayah & Lokasi"):
    # Filter Provinsi
    provinces = sorted(df_raw['province'].dropna().unique()) if 'province' in df_raw.columns else []
    selected_prov = st.multiselect("Pilih Provinsi", provinces)
    
    # Filter Kota (Hanya munculkan kota yang ada di provinsi terpilih)
    if selected_prov:
        filtered_cities = df_raw[df_raw['province'].isin(selected_prov)]['city'].dropna().unique()
    else:
        filtered_cities = df_raw['city'].dropna().unique() if 'city' in df_raw.columns else []
    
    selected_city = st.multiselect("Pilih Kota", sorted(filtered_cities))

# Filter 3: Produk & Segmentasi
with st.sidebar.expander("🎯 Produk & Cluster"):
    categories = df_raw['product_category'].unique()
    selected_cat = st.multiselect("Kategori Produk", categories)
    
    clusters = df_raw['Cluster Name'].dropna().unique() if 'Cluster Name' in df_raw.columns else []
    selected_cluster = st.multiselect("Pilih Cluster", clusters)

# --- APLIKASI FILTER ---
df_filtered = df_raw.copy()
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_date) & 
                          (df_filtered['cancel_time'].dt.date <= end_date)]

if selected_prov:
    df_filtered = df_filtered[df_filtered['province'].isin(selected_prov)]
if selected_city:
    df_filtered = df_filtered[df_filtered['city'].isin(selected_city)]
if selected_cat:
    df_filtered = df_filtered[df_filtered['product_category'].isin(selected_cat)]
if selected_cluster:
    df_filtered = df_filtered[df_filtered['Cluster Name'].isin(selected_cluster)]

# --- 5. DASHBOARD UTAMA ---
st.title("📊 Sales & Segment Dashboard")

# Row 1: Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Omset", f"Rp {df_filtered['order_amount'].sum():,.0f}")
m2.metric("Total Refund", f"Rp {df_filtered['total_refund'].sum():,.0f}")
m3.metric("Transaksi", f"{len(df_filtered):,}")
m4.metric("Provinsi Tercover", f"{df_filtered['province'].nunique() if 'province' in df_filtered.columns else 0}")

st.markdown("---")

# Row 2: Visualisasi
c1, c2 = st.columns(2)
with c1:
    st.subheader("📈 Tren Harian")
    df_trend = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
    fig_line = px.line(df_trend, x='cancel_time', y='order_amount', template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.subheader("📍 Distribusi Provinsi")
    if 'province' in df_filtered.columns:
        df_prov = df_filtered.groupby('province')['order_amount'].sum().reset_index().sort_values('order_amount', ascending=False).head(10)
        fig_bar = px.bar(df_prov, x='order_amount', y='province', orientation='h', color='order_amount')
        st.plotly_chart(fig_bar, use_container_width=True)

# Row 3: Clustering Detail
st.markdown("---")
st.subheader("🎯 Detail Clustering Terpilih")
# Re-generate agregasi untuk chart scatter agar tetap akurat sesuai filter
if not df_filtered.empty:
    df_cluster_viz = df_filtered.groupby(['platform_sku_variation', 'Cluster Name']).agg({
        'order_amount': 'sum',
        'total_refund': 'sum',
        'platform_sku_variation': 'count'
    }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()

    fig_scat = px.scatter(df_cluster_viz, x="transaction_count", y="total_refund", 
                          color="Cluster Name", size="order_amount", 
                          hover_name="platform_sku_variation",
                          color_discrete_map={
                              "Cluster 0 – Produk Bermasalah": "#EF553B",
                              "Cluster 1 – Produk Unggulan": "#00CC96",
                              "Cluster 2 – Produk Evaluasi": "#636EFA"
                          })
    st.plotly_chart(fig_scat, use_container_width=True)
else:
    st.info("Tidak ada data untuk ditampilkan pada cluster.")

# AI Section tetap sama seperti sebelumnya...
# (Bagian AI Insights dapat Anda letakkan di sini)
