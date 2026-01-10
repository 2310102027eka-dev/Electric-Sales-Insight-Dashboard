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
st.set_page_config(page_title="Dashboard Sales Syariah", layout="wide")

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
    # Pastikan ada minimal 3 produk unik untuk dibuat 3 cluster
    if df.empty or df['platform_sku_variation'].nunique() < 3:
        return None
    try:
        # Agregasi data per produk
        df_prod = df.groupby('platform_sku_variation').agg({
            'order_amount': 'sum', 
            'total_refund': 'sum', 
            'platform_sku_variation': 'count'
        }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()
        
        # Standarisasi data
        features = ['transaction_count', 'total_refund', 'order_amount']
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df_prod[features])
        
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_prod['cluster_id'] = kmeans.fit_predict(x_scaled)
        
        # Mapping nama cluster berdasarkan logika return
        c_means = df_prod.groupby('cluster_id')[features].mean()
        idx_bad = c_means['total_refund'].idxmax()
        rem = [i for i in range(3) if i != idx_bad]
        idx_good = c_means.loc[rem, 'transaction_count'].idxmax()
        idx_eval = [i for i in range(3) if i not in [idx_bad, idx_good]][0]
        
        map_clust = {
            idx_bad: "🔴 Cluster 0 (Refund Tinggi)", 
            idx_good: "🟢 Cluster 1 (Produk Unggulan)", 
            idx_eval: "🟡 Cluster 2 (Perlu Evaluasi)"
        }
        df_prod['Cluster Name'] = df_prod['cluster_id'].map(map_clust)
        return df_prod
    except Exception as e:
        st.write(f"Debug Cluster Error: {e}")
        return None

# --- 4. DATA PROCESSING & SIDEBAR ---
df_raw = load_data()
if df_raw.empty:
    st.warning("Data tidak ditemukan.")
    st.stop()

# Deteksi Kolom Otomatis
col_kota = next((c for c in df_raw.columns if any(x in c.lower() for x in ['city', 'kota', 'town', 'prov'])), None)
col_alasan = next((c for c in df_raw.columns if any(x in c.lower() for x in ['reason', 'alasan'])), None)

st.sidebar.header("🔍 Filter Dashboard")

# A. Tanggal (Lengkap)
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_d, max_d = df_raw['cancel_time'].min().date(), df_raw['cancel_time'].max().date()
    start_d = st.date_input("Mulai", min_d)
    end_d = st.date_input("Selesai", max_d)

# B. Kota & Alasan
with st.sidebar.expander("🏙️ Lokasi & Alasan", expanded=True):
    if col_kota:
        list_kota = sorted(df_raw[col_kota].dropna().unique())
        sel_cities = st.multiselect("Pilih Kota", list_kota)
    else:
        st.error("Kolom 'Kota' tidak ditemukan di database.")
        sel_cities = []
        
    if col_alasan:
        list_alasan = sorted(df_raw[col_alasan].dropna().unique())
        sel_reasons = st.multiselect("Alasan Refund", list_alasan)
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
st.title("📊 Dashboard Penjualan & Strategi Syariah")

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
    st.plotly_chart(px.line(df_t, x='cancel_time', y='order_amount', markers=True, template="plotly_white"), use_container_width=True)
with c2:
    st.subheader("🍰 Kategori Produk")
    st.plotly_chart(px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4), use_container_width=True)

# --- CLUSTERING ---
st.markdown("---")
st.subheader("🎯 Segmentasi Cluster Produk")
df_c = perform_clustering(df_filtered)

if df_c is not None:
    fig_scat = px.scatter(df_c, x="transaction_count", y="total_refund", color="Cluster Name", size="order_amount", hover_name="platform_sku_variation")
    st.plotly_chart(fig_scat, use_container_width=True)
    st.subheader("📋 Tabel Detail Cluster")
    st.dataframe(df_c, use_container_width=True)
else:
    st.info("💡 **Informasi Cluster**: Tidak muncul karena jumlah produk unik dalam filter kurang dari 3. Silakan perluas rentang tanggal atau pilih semua kota.")

# --- 6. AI BUSINESS ADVISOR SYARIAH ---
st.markdown("---")
st.subheader("🤖 AI Consultant: Saran Strategi Syariah")

if st.button("Minta Analisis & Saran Syariah"):
    refund_rate = (t_refund / t_sales * 100) if t_sales > 0 else 0
    with st.spinner("Menghubungi AI..."):
        try:
            # SOLUSI ERROR 404: Mencoba model yang paling umum tersedia
            model = genai.GenerativeModel('gemini-pro') # Menggunakan 'gemini-pro' sebagai alternatif stabil
            
            prompt = f"""
            Anda adalah pakar Bisnis Syariah. Berikan analisis untuk data:
            Omset: Rp {t_sales:,.0f}, Refund: Rp {t_refund:,.0f}, Rate: {refund_rate:.2f}%.
            Berikan 3 saran strategi berdasarkan:
            1. Amanah (Kualitas Produk)
            2. Keadilan (Pelayanan Refund)
            3. Keberkahan (Etika Bisnis)
            Jawaban harus singkat dan islami.
            """
            res = model.generate_content(prompt)
            st.success("Analisis AI Berhasil:")
            st.write(res.text)
            
        except Exception as e:
            st.warning("⚠️ AI sedang tidak dapat dijangkau. Berikut adalah saran sistem otomatis:")
            st.info(f"""
            **Saran Syariah Berdasarkan Data:**
            - **Amanah**: Jaga kualitas barang agar sesuai deskripsi (menghindari Gharar).
            - **Adil**: Segerakan proses refund senilai Rp {t_refund:,.0f} agar hak pembeli tertunaikan.
            - **Barakah**: Berikan pelayanan terbaik demi keridhoan pembeli (Antaradin).
            """)

# Debugging Kolom (Gunakan jika filter masih macet)
if st.checkbox("Debug: Lihat Semua Kolom"):
    st.write("Kolom terdeteksi:", df_raw.columns.tolist())
    st.write(f"Kolom Kota terpilih: {col_kota}")
    st.write(f"Kolom Alasan terpilih: {col_alasan}")
