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
st.set_page_config(page_title="Sales Dashboard Pro", layout="wide")

# CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #f0f2f6; }
    .stDataFrame { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Konfigurasi Secrets Error: {e}")
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
def perform_clustering(df):
    if df.empty or df['platform_sku_variation'].nunique() < 3:
        return None
    
    # Agregasi data per produk
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

# --- 4. DATA PROCESSING & FILTER SIDEBAR ---
df_raw = load_data()

if df_raw.empty:
    st.warning("Data kosong di Database.")
    st.stop()

st.sidebar.header("🔍 Filter Dashboard")

# A. Filter Tanggal Lengkap
with st.sidebar.expander("📅 Pilih Rentang Waktu", expanded=True):
    min_date = df_raw['cancel_time'].min().date()
    max_date = df_raw['cancel_time'].max().date()
    
    # Input tanggal mulai dan selesai secara eksplisit
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_date = st.date_input("Mulai", min_date, min_value=min_date, max_value=max_date)
    with col_t2:
        end_date = st.date_input("Selesai", max_date, min_value=min_date, max_value=max_date)

# B. Filter Kota (Aktif Filter, tapi tidak muncul di KPI)
with st.sidebar.expander("🏙️ Lokasi Kota"):
    list_kota = sorted(df_raw['city'].dropna().unique()) if 'city' in df_raw.columns else []
    selected_cities = st.multiselect("Pilih Kota", list_kota)

# C. Filter Kategori
with st.sidebar.expander("🏷️ Kategori Produk"):
    list_cat = sorted(df_raw['product_category'].unique()) if 'product_category' in df_raw.columns else []
    selected_cats = st.multiselect("Pilih Kategori", list_cat)

# --- EKSEKUSI FILTER ---
df_filtered = df_raw.copy()
# Filter Tanggal
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_date) & 
                          (df_filtered['cancel_time'].dt.date <= end_date)]
# Filter Kota
if selected_cities:
    df_filtered = df_filtered[df_filtered['city'].isin(selected_cities)]
# Filter Kategori
if selected_cats:
    df_filtered = df_filtered[df_filtered['product_category'].isin(selected_cats)]

# --- 5. TAMPILAN DASHBOARD ---
st.title("📊 Sales & Return Analysis Dashboard")

# Row 1: KPI Metrics (3 Kolom saja sesuai permintaan)
m1, m2, m3 = st.columns(3)
m1.metric("Total Omset", f"Rp {df_filtered['order_amount'].sum():,.0f}")
m2.metric("Total Refund", f"Rp {df_filtered['total_refund'].sum():,.0f}")
m3.metric("Total Transaksi", f"{len(df_filtered):,}")

st.markdown("---")

# Row 2: Charts
c1, c2 = st.columns([6, 4])
with c1:
    st.subheader("📈 Tren Penjualan Harian")
    df_trend = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
    fig_line = px.line(df_trend, x='cancel_time', y='order_amount', markers=True, template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.subheader("🍰 Kategori Produk")
    if not df_filtered.empty:
        fig_pie = px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4)
        fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Pilih data untuk melihat kategori")

# Row 3: Clustering (Scatter & Table)
st.markdown("---")
st.subheader("🎯 Clustering Produk (K-Means)")
df_cluster = perform_clustering(df_filtered)

if df_cluster is not None:
    # Scatter Plot
    fig_scat = px.scatter(df_cluster, x="transaction_count", y="total_refund", 
                          color="Cluster Name", size="order_amount", 
                          hover_name="platform_sku_variation",
                          color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_scat, use_container_width=True)
    
    # TAMPILAN TABEL DATA (Kembali seperti sebelumnya)
    st.subheader("📋 Tabel Detail Produk per Cluster")
    st.dataframe(df_cluster.sort_values(by='order_amount', ascending=False), use_container_width=True)
else:
    st.warning("Data tidak mencukupi untuk melakukan clustering (minimal 3 produk unik diperlukan).")

# Row 4: AI Advisor
st.markdown("---")
st.subheader("🤖 AI Business Advisor")
if st.button("Minta Saran Strategi AI"):
    if df_filtered.empty:
        st.warning("Data kosong, tidak bisa dianalisis.")
    else:
        with st.spinner("Menganalisis data..."):
            try:
                model_ai = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Analisis data berikut:
                - Omset: Rp {df_filtered['order_amount'].sum():,.0f}
                - Refund: Rp {df_filtered['total_refund'].sum():,.0f}
                - Kota yang difilter: {', '.join(selected_cities) if selected_cities else 'Semua Kota'}
                Berikan 3 saran strategi singkat untuk menaikkan penjualan dan mengurangi refund.
                """
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error("Gagal menghubungi AI.")
