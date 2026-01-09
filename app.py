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
st.set_page_config(page_title="Dashboard Final Penjualan & AI Insights", layout="wide")

# PASTIKAN baris try di bawah ini rapat ke kiri (tidak ada spasi sama sekali di depannya)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    
    genai.configure(api_key=GEMINI_API_KEY)
    # Gunakan gemini-1.5-flash agar tidak error NotFound
    model = genai.GenerativeModel('gemini-1.5-flash')
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
    if df.empty:
        return df
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

# --- 3. FUNGSI CLUSTERING ---
def perform_clustering(df):
    if df.empty or len(df) < 3:
        return None
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
        idx_unggulan: "Cluster 1 – Produk Unggulan (High Transaction)",
        idx_evaluasi: "Cluster 2 – Produk Evaluasi (Low Transaction)"
    }
    df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
    return df_prod

# --- 4. EKSEKUSI DATA & SIDEBAR ---
df_raw = load_data()

st.sidebar.header("🔍 Filter Data")
if df_raw.empty:
    st.warning("Data di database kosong. Dashboard tidak bisa ditampilkan.")
else:
    # Logic Filter Sidebar
    min_date = df_raw['cancel_time'].min().date()
    max_date = df_raw['cancel_time'].max().date()
    date_range = st.sidebar.date_input("Rentang Waktu", [min_date, max_date])
    
    selected_category = st.sidebar.multiselect("Kategori Produk", df_raw['product_category'].unique())
    selected_cancel_reason = st.sidebar.multiselect("Alasan Cancel", df_raw['cancel_reason'].unique())

    # Proses Filtering
    df_filtered = df_raw.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= date_range[0]) & 
                                  (df_filtered['cancel_time'].dt.date <= date_range[1])]
    if selected_category:
        df_filtered = df_filtered[df_filtered['product_category'].isin(selected_category)]
    if selected_cancel_reason:
        df_filtered = df_filtered[df_filtered['cancel_reason'].isin(selected_cancel_reason)]

    # --- 5. TAMPILAN DASHBOARD UTAMA ---
    st.title("📊 Sales & Return Final Dashboard")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Omset", f"Rp {df_filtered['order_amount'].sum():,.0f}")
    m2.metric("Total Refund", f"Rp {df_filtered['total_refund'].sum():,.0f}")
    m3.metric("Transaksi", f"{len(df_filtered):,}")

    st.markdown("---")

    # Visualisasi
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Tren Penjualan")
        df_trend = df_filtered.set_index('cancel_time').resample('D')['order_amount'].sum().reset_index()
        fig_line = px.line(df_trend, x='cancel_time', y='order_amount', title="Tren Harian")
        st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        st.subheader("🍰 Kategori Produk")
        fig_pie = px.pie(df_filtered, names='product_category', values='order_amount', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Clustering
    st.markdown("---")
    st.subheader("🎯 Clustering Produk (K-Means)")
    df_cluster = perform_clustering(df_filtered)
    if df_cluster is not None:
        fig_scat = px.scatter(df_cluster, x="transaction_count", y="total_refund", 
                              color="Cluster Name", size="order_amount", hover_name="platform_sku_variation")
        st.plotly_chart(fig_scat, use_container_width=True)
        st.dataframe(df_cluster, use_container_width=True)

    # AI Insights
    st.markdown("---")
    st.subheader("🤖 AI Business Advisor")
    if st.button("Minta Saran AI"):
        with st.spinner("Menganalisis..."):
            prompt = f"""Analisis data: Omset Rp {df_filtered['order_amount'].sum():,.0f}, 
            Refund Rp {df_filtered['total_refund'].sum():,.0f}. Berikan 3 saran strategi bisnis Syariah."""
            response = model.generate_content(prompt)
            st.markdown(response.text)
