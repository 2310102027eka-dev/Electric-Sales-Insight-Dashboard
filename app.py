import os
import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from supabase import create_client, Client
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP KONFIGURASI & JUDUL HALAMAN ---
st.set_page_config(page_title="Analisis Penjualan & Pengembalian Produk Listrik", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Konfigurasi Secrets Bermasalah: {e}")
    st.stop()

# --- 2. LOAD DATA ---
@st.cache_data(ttl=600)
def load_data():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        res = supabase.table("datapenjualanbaru").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return df
        if 'cancel_time' in df.columns:
            df['cancel_time'] = pd.to_datetime(df['cancel_time'], errors='coerce')
        for col in ['order_amount', 'total_refund', 'original_price', 'total_discount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# --- 3. CLUSTERING LOGIC ---
def perform_clustering(df):
    if df.empty or df['platform_sku_variation'].nunique() < 3:
        return None
    try:
        # Agregasi per SKU termasuk Kategori
        df_prod = df.groupby('platform_sku_variation').agg({
            'order_amount': 'sum', 
            'total_refund': 'sum', 
            'platform_sku_variation': 'count',
            'product_category': 'first'
        }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()
        
        # Standarisasi
        features = ['transaction_count', 'total_refund', 'order_amount']
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df_prod[features])
        
        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_prod['cluster_id'] = kmeans.fit_predict(x_scaled)
        
        # Penamaan Cluster
        c_means = df_prod.groupby('cluster_id')[features].mean()
        idx_bad = c_means['total_refund'].idxmax()
        rem = [i for i in range(3) if i != idx_bad]
        idx_good = c_means.loc[rem, 'transaction_count'].idxmax()
        idx_eval = [i for i in range(3) if i not in [idx_bad, idx_good]][0]
        
        map_clust = {
            idx_bad: "🔴 Cluster 0 (Refund Tinggi)", 
            idx_good: "🟢 Cluster 1 (Produk Unggulan)", 
            idx_eval: "🟡 Cluster 2 (Evaluasi)"
        }
        df_prod['Cluster Name'] = df_prod['cluster_id'].map(map_clust)
        
        # Urutan Kolom Tabel
        df_prod = df_prod[['platform_sku_variation', 'product_category', 'transaction_count', 'order_amount', 'total_refund', 'Cluster Name']]
        return df_prod
    except:
        return None

# --- 4. DATA PROCESSING & SIDEBAR ---
df_raw = load_data()
if df_raw.empty:
    st.warning("Data tidak tersedia di database.")
    st.stop()

# Deteksi Kolom Otomatis (Kota & Alasan)
col_kota = next((c for c in df_raw.columns if any(x in c.lower() for x in ['city', 'kota', 'town', 'prov', 'wilayah'])), None)
col_alasan = next((c for c in df_raw.columns if any(x in c.lower() for x in ['reason', 'alasan'])), None)

st.sidebar.header("🔍 Filter Dashboard")

# A. Rentang Waktu (Kalender Lengkap)
with st.sidebar.expander("📅 Pilih Tanggal", expanded=True):
    min_d, max_d = df_raw['cancel_time'].min().date(), df_raw['cancel_time'].max().date()
    start_d = st.date_input("Mulai", min_d)
    end_d = st.date_input("Selesai", max_d)

# B. Kota & Alasan Refund
with st.sidebar.expander("🏙️ Lokasi & Alasan", expanded=True):
    if col_kota:
        sel_cities = st.multiselect("Pilih Kota", sorted(df_raw[col_kota].dropna().unique()))
    else:
        st.error("Kolom Kota tidak terdeteksi.")
        sel_cities = []
        
    if col_alasan:
        sel_reasons = st.multiselect("Alasan Refund", sorted(df_raw[col_alasan].dropna().unique()))
    else:
        sel_reasons = []

# C. Kategori Produk
with st.sidebar.expander("🏷️ Kategori"):
    sel_cats = st.multiselect("Pilih Kategori", sorted(df_raw['product_category'].unique()))

# EKSEKUSI FILTER
df_filtered = df_raw.copy()
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_d) & (df_filtered['cancel_time
