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

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
def get_clustering_labels(df):
    # Minimal butuh 3 baris data berbeda untuk membuat 3 cluster
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
            idx_bermasalah: "🔴 Produk Bermasalah",
            idx_unggulan: "🟢 Produk Unggulan",
            idx_evaluasi: "🟡 Produk Evaluasi"
        }
        df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
        return df_prod[['platform_sku_variation', 'Cluster Name']]
    except:
        return None

# --- 4. DATA PROCESSING ---
df_raw = load_data()

if df_raw.empty:
    st.warning("Data kosong di Database.")
    st.stop()

# Tambahkan Cluster ke df_raw
df_cluster_map = get_clustering_labels(df_raw)
if df_cluster_map is not None:
    df_raw = df_raw.merge(df_cluster_map, on='platform_sku_variation', how='left')
else:
    df_raw['Cluster Name'] = "Belum Terklasifikasi"

# --- 5. SIDEBAR FILTER ---
st.sidebar.header("🔍 Filter Dashboard")

# A. Filter Tanggal
with st.sidebar.expander("📅 Rentang Waktu", expanded=True):
    min_d = df_raw['cancel_time'].min().date()
    max_d = df_raw['cancel_time'].max().date()
    date_opt = st.selectbox("Periode Cepat", ["Semua Waktu", "7 Hari Terakhir", "Bulan Ini", "Custom"])
    
    start_date, end_date = min_d, max_d
    if date_opt == "7 Hari Terakhir":
        start_date = max_d - pd.Timedelta(days=7)
    elif date_opt == "Bulan Ini":
        start_date = max_d.replace(day=1)
    elif date_opt == "Custom":
        dr = st.date_input("Rentang Tanggal", [min_d, max_d])
        if len(dr) == 2: start_date, end_date = dr

# B. Filter Kota (Hanya Kota)
with st.sidebar.expander("🏙️ Lokasi Kota", expanded=True):
    # Cek jika kolom 'city' ada, jika tidak pakai list kosong
    list_kota = sorted(df_raw['city'].dropna().unique()) if 'city' in df_raw.columns else []
    selected_cities = st.multiselect("Pilih Kota", list_kota)

# C. Filter Cluster & Kategori
with st.sidebar.expander("🎯 Cluster & Kategori"):
    list_cluster = sorted(df_raw['Cluster Name'].unique())
    selected_clusters = st.multiselect("Pilih Cluster", list_cluster)
    
    list_cat = sorted(df_raw['product_category'].unique()) if 'product_category' in df_raw.columns else []
    selected_cats = st.multiselect("Kategori Produk", list_cat)

# --- EKSEKUSI FILTER ---
df_filtered = df_raw.copy()
df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= start_date) & 
                          (df_filtered['cancel_time'].dt.date <= end_date)]

if selected_cities:
    df_filtered = df_filtered[df_filtered['city'].isin(selected_cities)]
if selected_clusters:
    df_filtered = df_filtered[df_filtered['Cluster Name'].isin(selected_clusters)]
if selected_cats:
    df_filtered = df_filtered[df_filtered['product_category'].isin(selected_cats)]

# --- 6. MAIN DASHBOARD ---
st.title("📊 Sales & Category Insights")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Omset", f"Rp {df_filtered['order_amount'].sum():,.0f}")
m2.metric("Total Refund", f"Rp {df_filtered['total_refund'].sum():,.0f}")
m3.metric("Transaksi", f"{len(df_filtered):,}")
m4.metric("Kota Aktif", f"{df_filtered['city'].nunique() if 'city' in df_filtered.columns else 0}")

st.markdown("---")

# Visualisasi: Tren & Pie Chart
c1, c2 = st.columns([6, 4])

with c1:
    st.subheader("📈 Tren Penjualan")
    df_trend = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
    fig_line = px.line(df_trend, x='cancel_time', y='order_amount', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.subheader("🍰 Kategori Produk")
    if not df_filtered.empty and 'product_category' in df_filtered.columns:
        fig_pie = px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4)
        fig_pie.update_layout(legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Tidak ada data kategori")

# Visualisasi: Scatter Cluster
st.markdown("---")
st.subheader("🎯 Sebaran Cluster Produk")
if not df_filtered.empty:
    df_viz = df_filtered.groupby(['platform_sku_variation', 'Cluster Name']).agg({
        'order_amount': 'sum', 'total_refund': 'sum', 'platform_sku_variation': 'count'
    }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()

    fig_scat = px.scatter(df_viz, x="transaction_count", y="total_refund", 
                          color="Cluster Name", size="order_amount", 
                          hover_name="platform_sku_variation")
    st.plotly_chart(fig_scat, use_container_width=True)

# AI SECTION
st.markdown("---")
if st.button("🤖 Tanya AI Advisor"):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Data: Omset Rp {df_filtered['order_amount'].sum()}, Refund Rp {df_filtered['total_refund'].sum()}. Berikan 2 saran singkat."
        res = model.generate_content(prompt)
        st.success(res.text)
    except:
        st.error("Gagal terhubung ke AI.")
