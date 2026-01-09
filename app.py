import os
import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from supabase import create_client, Client
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP KONFIGURASI ---
st.set_page_config(page_title="Sales Dashboard & AI Insights", layout="wide")

# Gaya Visual Custom
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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

# --- 2. LOAD DATA ---
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
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# --- 3. CLUSTERING LOGIC ---
def get_clustering_labels(df):
    if df.empty or len(df) < 3:
        return None
    
    # Agregasi per produk untuk clustering
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
    
    # Penamaan Cluster
    cluster_means = df_prod.groupby('cluster_id')[features].mean()
    idx_bermasalah = cluster_means['total_refund'].idxmax()
    rem = [i for i in range(3) if i != idx_bermasalah]
    idx_unggulan = cluster_means.loc[rem, 'transaction_count'].idxmax()
    idx_evaluasi = [i for i in range(3) if i not in [idx_bermasalah, idx_unggulan]][0]
    
    mapping = {
        idx_bermasalah: "🔴 Produk Bermasalah (High Refund)",
        idx_unggulan: "🟢 Produk Unggulan (High Sales)",
        idx_evaluasi: "🟡 Produk Evaluasi (Low Sales)"
    }
    df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
    return df_prod[['platform_sku_variation', 'Cluster Name']]

# --- 4. PROSES DATA & SIDEBAR FILTER ---
df_raw = load_data()

if df_raw.empty:
    st.warning("Data tidak tersedia.")
    st.stop()

# Tempelkan label cluster ke data utama agar bisa difilter
df_cluster_map = get_clustering_labels(df_raw)
if df_cluster_map is not None:
    df_raw = df_raw.merge(df_cluster_map, on='platform_sku_variation', how='left')

# --- SIDEBAR UI ---
st.sidebar.header("🔍 Filter Dashboard")

# 1. Filter Tanggal (Tampilan Baru)
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_date = df_raw['cancel_time'].min().date()
    max_date = df_raw['cancel_time'].max().date()
    
    date_option = st.selectbox("Periode Cepat", ["Semua Waktu", "7 Hari Terakhir", "Bulan Ini", "Custom"])
    
    start_date, end_date = min_date, max_date
    if date_option == "7 Hari Terakhir":
        start_date = max_date - pd.Timedelta(days=7)
    elif date_option == "Bulan Ini":
        start_date = max_date.replace(day=1)
    elif date_option == "Custom":
        dr = st.date_input("Pilih Tanggal", [min_date, max_date])
        if len(dr) == 2: start_date, end_date = dr

# 2. Filter Kota
with st.sidebar.expander("🏙️ Lokasi & Kota", expanded=True):
    all_cities = sorted(df_raw['city'].dropna().unique()) if 'city' in df_raw.columns else []
    selected_cities = st.multiselect("Pilih Kota", all_cities)

# 3. Filter Cluster & Kategori
with st.sidebar.expander("🎯 Segmentasi & Produk"):
    all_clu
