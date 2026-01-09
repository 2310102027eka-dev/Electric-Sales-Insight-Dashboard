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
st.set_page_config(page_title="Sales Dashboard", layout="wide")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Konfigurasi Secrets Masalah: {e}")
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
        cols_to_fix = ['order_amount', 'total_refund', 'original_price', 'total_discount']
        for col in cols_to_fix:
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
        df_prod = df.groupby('platform_sku_variation').agg({
            'order_amount': 'sum', 'total_refund': 'sum', 'platform_sku_variation': 'count'
        }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()
        features = ['transaction_count', 'total_refund', 'order_amount']
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df_prod[features])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_prod['cluster_id'] = kmeans.fit_predict(x_scaled)
        
        cluster_means = df_prod.groupby('cluster_id')[features].mean()
        idx_bermasalah = cluster_means['total_refund'].idxmax()
        rem = [i for i in range(3) if i != idx_bermasalah]
        idx_unggulan = cluster_means.loc[rem, 'transaction_count'].idxmax()
        idx_evaluasi = [i for i in range(3) if i not in [idx_bermasalah, idx_unggulan]][0]
        
        mapping = {
            idx_bermasalah: "🔴 Produk Bermasalah",
            idx_unggulan: "🟢 Produk Unggulan",
            idx_evaluasi: "🟡 Produk Evaluasi"
        }
        df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
        return df_prod
    except:
        return None

# --- 4. DATA PROCESSING & SIDEBAR ---
df_raw = load_data()
if df_raw.empty:
    st.warning("Data kosong di database.")
    st.stop()

# --- DETEKSI KOLOM OTOMATIS (UNTUK PERBAIKAN FILTER) ---
# Mencari kolom yang mengandung kata 'city' atau 'kota'
col_kota = next((c for c in df_raw.columns if 'city' in c.lower() or 'kota' in c.lower()), None)
# Mencari kolom yang mengandung kata 'reason' atau 'alasan'
col_alasan = next((c for c in df_raw.columns if 'reason' in c.lower() or 'alasan' in c.lower()), None)

st.sidebar.header("🔍 Filter Dashboard")

# A. Filter Tanggal
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_date = df_raw['cancel_time'].min().date()
    max_date = df_raw['cancel_time'].max().date()
    start_date = st.date_input("Dari Tanggal", min_date)
    end_date = st.date_input("Sampai Tanggal", max_date)

# B. Filter Kota (DIPERBAIKI)
with st.sidebar.expander("🏙️ Lokasi & Alasan", expanded=True):
    if col_kota:
        list_kota = sorted(df_raw[col_kota].dropna().unique())
        selected_cities = st.multiselect(f"Pilih Kota (Kolom: {col_kota})", list_kota)
    else:
        st.error("Kolom 'Kota' tidak ditemukan di database!")
        selected_cities = []

    if col_alasan:
        list_alasan = sorted(df_raw[col_alasan].dropna().unique())
        selected_reasons = st.multiselect(f"Alasan Refund (Kolom: {col_alasan})", list_alasan)
    else:
        st.warning("Kolom 'Alasan Refund' tidak ditemukan.")
        selected_reasons = []

# C. Filter Kategori
with st.sidebar.expander("🏷️ Kategori"):
    selected_cats = st.multiselect("Pilih Kategori", sorted(df_raw['product_category'].unique()))

# EKSEKUSI FILTER
df_filtered = df_raw.copy()
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_date) & (df_filtered['cancel_time'].dt.date <= end_date)]

if col_kota and selected_cities:
    df_filtered = df_filtered[df_filtered[col_kota].isin(selected_cities)]
if col_alasan and selected_reasons:
    df_filtered = df_filtered[df_filtered[col_alasan].isin(selected_reasons)]
if selected_cats:
    df_filtered = df_filtered[df_filtered['product_category'].isin(selected_cats)]

# --- 5. TAMPILAN DASHBOARD ---
st.title("📊 Sales Insight Dashboard")

# KPI - 3 KOLOM
m1, m2, m3 = st.columns(3)
m1.metric("Total Omset", f"Rp {df_filtered['order_amount'].sum():,.0f}")
m2.metric("Total Refund", f"Rp {df_filtered['total_refund'].sum():,.0f}")
m3.metric("Total Transaksi", f"{len(df_filtered):,}")

st.markdown("---")

# Visualisasi
c1, c2 = st.columns([6, 4])
with c1:
    st.subheader("📈 Tren Penjualan")
    df_trend = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
    st.plotly_chart(px.line(df_trend, x='cancel_time', y='order_amount', markers=True), use_container_width=True)
with c2:
    st.subheader("🍰 Kategori Produk")
    st.plotly_chart(px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4), use_container_width=True)

# Clustering & Tabel
st.markdown("---")
st.subheader("🎯 Analisis Cluster & Tabel Data")
df_cluster = perform_clustering(df_filtered)
if df_cluster is not None:
    st.plotly_chart(px.scatter(df_cluster, x="transaction_count", y="total_refund", color="Cluster Name", size="order_amount"), use_container_width=True)
    st.dataframe(df_cluster, use_container_width=True)
else:
    st.info("Data tidak cukup untuk clustering.")

# AI Advisor (DENGAN FALLBACK JIKA 429)
st.markdown("---")
st.subheader("🤖 AI Business Advisor")
if st.button("Minta Saran Strategi"):
    # Hitung logika sederhana dulu (Fallback)
    refund_rate = (df_filtered['total_refund'].sum() / df_filtered['order_amount'].sum() * 100) if df_filtered['order_amount'].sum() > 0 else 0
    st.info(f"**Analisis Sistem:** Refund rate Anda saat ini adalah {refund_rate:.2f}%.")
    
    with st.spinner("Menghubungkan ke AI..."):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Berikan 3 saran singkat untuk toko dengan omset Rp {df_filtered['order_amount'].sum()} dan refund rate {refund_rate:.2f}%."
            response = model.generate_content(prompt)
            st.success("Saran Strategi AI:")
            st.write(response.text)
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ Kuota AI Gratis habis (Error 429). Silakan coba lagi beberapa saat lagi. Gunakan Analisis Sistem di atas untuk sementara.")
            else:
                st.error(f"Gagal terhubung ke AI: {e}")
