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
st.set_page_config(page_title="Sales Insight Pro", layout="wide")

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
    st.error(f"Konfigurasi Secrets Masalah: {e}")
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
        
        # Konversi Tanggal
        if 'cancel_time' in df.columns:
            df['cancel_time'] = pd.to_datetime(df['cancel_time'], errors='coerce')
        
        # Konversi Numerik
        cols_to_fix = ['order_amount', 'total_refund', 'original_price', 'total_discount']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Supabase: {e}")
        return pd.DataFrame()

# --- 3. CLUSTERING LOGIC ---
def perform_clustering(df):
    if df.empty or df['platform_sku_variation'].nunique() < 3:
        return None
    try:
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
        
        cluster_means = df_prod.groupby('cluster_id')[features].mean()
        idx_bermasalah = cluster_means['total_refund'].idxmax()
        rem = [i for i in range(3) if i != idx_bermasalah]
        idx_unggulan = cluster_means.loc[rem, 'transaction_count'].idxmax()
        idx_evaluasi = [i for i in range(3) if i not in [idx_bermasalah, idx_unggulan]][0]
        
        mapping = {
            idx_bermasalah: "Cluster 0 – Produk Bermasalah (High Return)",
            idx_unggulan: "Cluster 1 – Produk Unggulan (High Sales)",
            idx_evaluasi: "Cluster 2 – Produk Evaluasi (Low Sales)"
        }
        df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
        return df_prod
    except:
        return None

# --- 4. DATA PROCESSING & SIDEBAR ---
df_raw = load_data()

if df_raw.empty:
    st.warning("Data kosong.")
    st.stop()

st.sidebar.header("🔍 Filter Dashboard")

# A. Filter Tanggal Lengkap
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_date = df_raw['cancel_time'].min().date()
    max_date = df_raw['cancel_time'].max().date()
    c_t1, c_t2 = st.columns(2)
    start_date = c_t1.date_input("Dari", min_date)
    end_date = c_t2.date_input("Sampai", max_date)

# B. Filter Lokasi & Alasan (DINAMIS)
with st.sidebar.expander("🏙️ Lokasi & Alasan Refund", expanded=True):
    # Cek nama kolom kota yang mungkin berbeda
    col_kota = next((c for c in df_raw.columns if c.lower() in ['city', 'kota']), None)
    list_kota = sorted(df_raw[col_kota].dropna().unique()) if col_kota else []
    selected_cities = st.multiselect("Pilih Kota", list_kota)

    # Cek nama kolom alasan refund
    col_alasan = next((c for c in df_raw.columns if 'reason' in c.lower() or 'alasan' in c.lower()), None)
    list_alasan = sorted(df_raw[col_alasan].dropna().unique()) if col_alasan else []
    selected_reasons = st.multiselect("Alasan Refund", list_alasan)

# C. Filter Kategori
with st.sidebar.expander("🏷️ Kategori Produk"):
    list_cat = sorted(df_raw['product_category'].unique()) if 'product_category' in df_raw.columns else []
    selected_cats = st.multiselect("Pilih Kategori", list_cat)

# --- EKSEKUSI FILTER ---
df_filtered = df_raw.copy()
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_date) & 
                          (df_filtered['cancel_time'].dt.date <= end_date)]

if col_kota and selected_cities:
    df_filtered = df_filtered[df_filtered[col_kota].isin(selected_cities)]
if col_alasan and selected_reasons:
    df_filtered = df_filtered[df_filtered[col_alasan].isin(selected_reasons)]
if selected_cats:
    df_filtered = df_filtered[df_filtered['product_category'].isin(selected_cats)]

# --- 5. TAMPILAN UTAMA ---
st.title("📊 Sales Insight & Clustering")

# Row 1: KPI
m1, m2, m3 = st.columns(3)
m1.metric("Total Omset", f"Rp {df_filtered['order_amount'].sum():,.0f}")
m2.metric("Total Refund", f"Rp {df_filtered['total_refund'].sum():,.0f}")
m3.metric("Total Transaksi", f"{len(df_filtered):,}")

st.markdown("---")

# Row 2: Charts
c1, c2 = st.columns([6, 4])
with c1:
    st.subheader("📈 Tren Penjualan")
    df_trend = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
    fig_line = px.line(df_trend, x='cancel_time', y='order_amount', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
with c2:
    st.subheader("🍰 Kategori Produk")
    if not df_filtered.empty:
        fig_pie = px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# Row 3: Clustering
st.markdown("---")
st.subheader("🎯 Hasil Clustering Produk")
df_cluster = perform_clustering(df_filtered)
if df_cluster is not None:
    fig_scat = px.scatter(df_cluster, x="transaction_count", y="total_refund", 
                          color="Cluster Name", size="order_amount", hover_name="platform_sku_variation")
    st.plotly_chart(fig_scat, use_container_width=True)
    
    st.subheader("📋 Tabel Detail Cluster")
    st.dataframe(df_cluster, use_container_width=True)
else:
    st.info("Data tidak cukup untuk clustering.")

# Row 4: AI Advisor (FIXED CONNECTION)
st.markdown("---")
st.subheader("🤖 AI Business Advisor")
if st.button("Minta Saran Strategi AI"):
    with st.spinner("Menghubungkan ke Gemini AI..."):
        try:
            # Cari model yang tersedia secara dinamis
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = next((m for m in available_models if '1.5-flash' in m), available_models[0])
            
            chat_model = genai.GenerativeModel(model_name)
            prompt = f"""
            Analisis data penjualan ini:
            - Total Omset: Rp {df_filtered['order_amount'].sum():,.0f}
            - Total Refund: Rp {df_filtered['total_refund'].sum():,.0f}
            - Top Kategori: {df_filtered['product_category'].mode()[0] if not df_filtered.empty else 'N/A'}
            
            Berikan 3 saran taktis untuk meningkatkan profit dan mengurangi refund secara jujur.
            """
            response = chat_model.generate_content(prompt)
            st.success(f"Analisis AI ({model_name}):")
            st.write(response.text)
        except Exception as e:
            st.error(f"Gagal terhubung ke AI: {e}")
            st.info("Pastikan API Key Anda benar dan kuota API tersedia.")
