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
st.set_page_config(page_title="Dashboard Penjualan Syariah", layout="wide")

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
    st.error(f"Konfigurasi Masalah: {e}")
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
        df_prod = df.groupby('platform_sku_variation').agg({
            'order_amount': 'sum', 'total_refund': 'sum', 'platform_sku_variation': 'count'
        }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df_prod[['transaction_count', 'total_refund', 'order_amount']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_prod['cluster_id'] = kmeans.fit_predict(x_scaled)
        
        # Labeling Manual Berdasarkan Mean
        c_means = df_prod.groupby('cluster_id').mean()
        idx_bad = c_means['total_refund'].idxmax()
        rem = [i for i in range(3) if i != idx_bad]
        idx_good = c_means.loc[rem, 'transaction_count'].idxmax()
        idx_eval = [i for i in range(3) if i not in [idx_bad, idx_good]][0]
        
        map_clust = {idx_bad: "🔴 Cluster 0 (High Refund)", idx_good: "🟢 Cluster 1 (Top Sales)", idx_eval: "🟡 Cluster 2 (Evaluasi)"}
        df_prod['Cluster Name'] = df_prod['cluster_id'].map(map_clust)
        return df_prod
    except: return None

# --- 4. DATA PROCESSING & SIDEBAR ---
df_raw = load_data()
if df_raw.empty:
    st.warning("Database kosong.")
    st.stop()

# --- DETEKSI KOLOM OTOMATIS (AGAR FILTER KOTA AKTIF) ---
# Mencari kolom kota (cek city, kota, town, atau province)
col_kota = next((c for c in df_raw.columns if any(x in c.lower() for x in ['city', 'kota', 'town'])), None)
# Mencari kolom alasan refund
col_alasan = next((c for c in df_raw.columns if any(x in c.lower() for x in ['reason', 'alasan'])), None)

st.sidebar.header("🔍 Filter Dashboard")

# A. Tanggal
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_d, max_d = df_raw['cancel_time'].min().date(), df_raw['cancel_time'].max().date()
    start_d = st.date_input("Mulai", min_d)
    end_d = st.date_input("Selesai", max_d)

# B. Kota & Alasan (Sangat penting untuk filter)
with st.sidebar.expander("🏙️ Lokasi & Alasan Refund", expanded=True):
    if col_kota:
        sel_cities = st.multiselect(f"Pilih Kota", sorted(df_raw[col_kota].dropna().unique()))
    else:
        st.warning("Kolom Kota tidak terdeteksi.")
        sel_cities = []
        
    if col_alasan:
        sel_reasons = st.multiselect("Alasan Refund", sorted(df_raw[col_alasan].dropna().unique()))
    else:
        sel_reasons = []

# C. Kategori
with st.sidebar.expander("🏷️ Kategori"):
    sel_cats = st.multiselect("Pilih Kategori", sorted(df_raw['product_category'].unique()))

# EKSEKUSI FILTER
df_filtered = df_raw.copy()
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_d) & (df_filtered['cancel_time'].dt.date <= end_d)]
if col_kota and sel_cities: df_filtered = df_filtered[df_filtered[col_kota].isin(sel_cities)]
if col_alasan and sel_reasons: df_filtered = df_filtered[df_filtered[col_alasan].isin(sel_reasons)]
if sel_cats: df_filtered = df_filtered[df_filtered['product_category'].isin(sel_cats)]

# --- 5. TAMPILAN DASHBOARD ---
st.title("📊 Dashboard Analisis Penjualan & AI Syariah")

# KPI - 3 KOLOM
m1, m2, m3 = st.columns(3)
t_sales = df_filtered['order_amount'].sum()
t_refund = df_filtered['total_refund'].sum()
m1.metric("Total Omset", f"Rp {t_sales:,.0f}")
m2.metric("Total Refund", f"Rp {t_refund:,.0f}")
m3.metric("Total Transaksi", f"{len(df_filtered):,}")

st.markdown("---")

# Visualisasi
c1, c2 = st.columns([6, 4])
with c1:
    st.subheader("📈 Tren Harian")
    df_t = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
    st.plotly_chart(px.line(df_t, x='cancel_time', y='order_amount', markers=True), use_container_width=True)
with c2:
    st.subheader("🍰 Kategori Produk")
    st.plotly_chart(px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4), use_container_width=True)

# Clustering & Tabel
st.markdown("---")
st.subheader("🎯 Segmentasi Cluster Produk")
df_c = perform_clustering(df_filtered)
if df_c is not None:
    st.plotly_chart(px.scatter(df_c, x="transaction_count", y="total_refund", color="Cluster Name", size="order_amount"), use_container_width=True)
    st.subheader("📋 Tabel Detail Clustering")
    st.dataframe(df_c, use_container_width=True)
else:
    st.info("Data tidak cukup untuk clustering.")

# --- 6. AI BUSINESS ADVISOR SYARIAH ---
st.markdown("---")
st.subheader("🤖 AI Consultant: Saran Bisnis Syariah")

if st.button("Minta Analisis & Saran Syariah"):
    # Fallback / Saran Manual jika AI Limit
    refund_rate = (t_refund / t_sales * 100) if t_sales > 0 else 0
    
    with st.spinner("Menghubungi AI Consultant..."):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Anda adalah seorang Konsultan Bisnis Syariah yang ahli dalam Maqasid Syariah dan Muamalah.
            Analisis data berikut:
            - Omset: Rp {t_sales:,.0f}
            - Total Refund: Rp {t_refund:,.0f}
            - Refund Rate: {refund_rate:.2f}%
            - Kategori Terlaris: {df_filtered['product_category'].mode()[0] if not df_filtered.empty else 'N/A'}
            - Alasan Refund Utama: {df_filtered[col_alasan].mode()[0] if col_alasan and not df_filtered.empty else 'N/A'}

            Berikan saran strategis yang mengandung unsur:
            1. Nilai Amanah (Bagaimana menjaga kualitas agar pembeli tidak kecewa/refund).
            2. Nilai Keadilan (Cara menangani refund agar tidak merugikan pembeli maupun penjual).
            3. Nilai Keberkahan (Saran taktis agar bisnis berkembang secara halal dan berkah).
            Berikan jawaban dalam poin-poin singkat dan menyejukkan.
            """
            res = model.generate_content(prompt)
            st.success("Saran Strategi Bisnis Syariah:")
            st.write(res.text)
            
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ **Kuota AI Terlampaui (Error 429).** Berikut adalah Saran Syariah Otomatis untuk Anda:")
                st.info(f"""
                **Saran Syariah Berdasarkan Data:**
                1. **Amanah:** Refund rate Anda {refund_rate:.1f}%. Jika tinggi, periksalah kesesuaian deskripsi produk dengan fisik barang agar tidak ada unsur *Gharar* (ketidakjelasan).
                2. **Keadilan:** Pastikan proses pengembalian dana (refund) dilakukan dengan cepat dan jujur untuk menjaga ridho antara penjual dan pembeli.
                3. **Barakah:** Sedekahkan sebagian dari omset Rp {t_sales:,.0f} Anda sebagai pembersih harta dan penarik keberkahan dalam perniagaan.
                """)
            else:
                st.error(f"Gagal terhubung: {e}")

# Tambahan debug jika filter kota masih gagal (Hanya tampil di development)
if st.checkbox("Lihat Nama Kolom Database (Debug Only)"):
    st.write(df_raw.columns.tolist())
