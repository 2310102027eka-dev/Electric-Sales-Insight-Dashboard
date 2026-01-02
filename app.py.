import os
import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from supabase import create_client, Client
from datetime import datetime

# --- 1. SETUP KONFIGURASI & API ---
st.set_page_config(page_title="Dashboard Final Penjualan & AI Insights", layout="wide")

# Setup Gemini AI
# Menggunakan API Key yang Anda berikan via environment variable
GEMINI_API_KEY = os.environ.get('AIzaSyAPFFdem2Wn9A6GePvqnEcFr5eLbZIq0-Q')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Setup Supabase
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    return create_client(url, key)

supabase = init_supabase()

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=600)
def load_data():
    try:
        # Mengambil data dari tabel [datapenjualanbaru]
        res = supabase.table("datapenjualanbaru").select("*").execute()
        df = pd.DataFrame(res.data)
        
        # Konversi kolom tanggal (Asumsi ada kolom order_date atau gunakan cancel_time)
        # Kita gunakan cancel_time sesuai instruksi spesifik Anda
        if 'cancel_time' in df.columns:
            df['cancel_time'] = pd.to_datetime(df['cancel_time'], errors='coerce')
        
        # Pastikan kolom numerik
        cols_to_fix = ['order_amount', 'total_refund', 'original_price', 'total_discount']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# --- 3. SIDEBAR (FILTER DATA) ---
df_raw = load_data()

st.sidebar.header("🔍 Filter Data")

if not df_raw.empty:
    # Filter Tanggal
    min_date = df_raw['cancel_time'].min().date() if not df_raw['cancel_time'].isnull().all() else datetime.now().date()
    max_date = df_raw['cancel_time'].max().date() if not df_raw['cancel_time'].isnull().all() else datetime.now().date()
    
    date_range = st.sidebar.date_input("Rentang Waktu", [min_date, max_date])

    # Dynamic Multiselect Filters
    # Saya kelompokkan kolom-kolom penting agar sidebar tidak terlalu panjang namun tetap mencakup permintaan Anda
    st.sidebar.subheader("Filter Spesifik")
    
    selected_category = st.sidebar.multiselect("Kategori Produk", df_raw['product_category'].unique())
    selected_city = st.sidebar.multiselect("Kota (Regency/City)", df_raw['regency_city'].unique())
    selected_platform_sku = st.sidebar.multiselect("Platform SKU", df_raw['platform_sku_variation'].unique())
    selected_cancel_reason = st.sidebar.multiselect("Alasan Cancel", df_raw['cancel_reason'].unique().tolist())

    # --- LOGIKA FILTERING ---
    df_filtered = df_raw.copy()
    
    # Filter by Date
    if len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= date_range[0]) & 
                                  (df_filtered['cancel_time'].dt.date <= date_range[1])]
    
    # Filter by Multiselects
    if selected_category:
        df_filtered = df_filtered[df_filtered['product_category'].isin(selected_category)]
    if selected_city:
        df_filtered = df_filtered[df_filtered['regency_city'].isin(selected_city)]
    if selected_platform_sku:
        df_filtered = df_filtered[df_filtered['platform_sku_variation'].isin(selected_platform_sku)]
    if selected_cancel_reason:
        df_filtered = df_filtered[df_filtered['cancel_reason'].isin(selected_cancel_reason)]

# --- 4. MAIN DASHBOARD ---
st.title("📊 Sales & Return Final Dashboard")

if df_raw.empty:
    st.warning("Data tidak ditemukan di database.")
else:
    # Metric Cards
    m1, m2, m3 = st.columns(3)
    total_sales = df_filtered['order_amount'].sum()
    total_refund = df_filtered['total_refund'].sum()
    total_orders = len(df_filtered)

    m1.metric("Total Omset (Filtered)", f"Rp {total_sales:,.0f}")
    m2.metric("Total Refund (Filtered)", f"Rp {total_refund:,.0f}")
    m3.metric("Jumlah Transaksi", f"{total_orders:,}")

    st.markdown("---")

    # Visualisasi
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Tren Penjualan/Pembatalan")
        df_trend = df_filtered.set_index('cancel_time').resample('M')['order_amount'].sum().reset_index()
        fig_bar = px.bar(df_trend, x='cancel_time', y='order_amount', 
                         title="Tren Bulanan", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("🍰 Komposisi Kategori Produk")
        fig_pie = px.pie(df_filtered, names='product_category', values='order_amount', 
                         title="Proporsi Per Kategori", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # --- 5. AI INSIGHTS (GEN AI) ---
    st.subheader("🤖 AI Business Advisor")
    
    if st.button("Minta Saran AI"):
        with st.spinner("Sedang menganalisis data Anda..."):
            # Ringkasan data untuk dikirim ke AI
            top_category = df_filtered.groupby('product_category')['order_amount'].sum().idxmax() if not df_filtered.empty else "N/A"
            most_cancel_reason = df_filtered['cancel_reason'].mode()[0] if not df_filtered.empty and not df_filtered['cancel_reason'].mode().empty else "N/A"
            
            prompt = f"""
            Anda adalah konsultan bisnis senior. Analisis data penjualan berikut:
            - Total Omset saat ini: Rp {total_sales:,.0f}
            - Total Refund/Kembali: Rp {total_refund:,.0f}
            - Jumlah Transaksi: {total_orders}
            - Kategori Terlaris: {top_category}
            - Alasan Pembatalan Terbanyak: {most_cancel_reason}
            
            Berikan laporan singkat yang mencakup:
            1. Analisis Tren (Apakah kondisi bisnis sehat?)
            2. Penyebab Kemungkinan (Mengapa refund/pembatalan terjadi?)
            3. Saran Strategi Syariah (Bagaimana mengelola transaksi, retur, dan komplain pelanggan sesuai prinsip kejujuran, keadilan, dan keridhaan dalam Islam?)
            """
            
            try:
                response = model.generate_content(prompt)
                st.markdown("### 💡 Hasil Analisis AI")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")

    # Preview Data
    with st.expander("Lihat Data Terfilter (Raw)"):
        st.dataframe(df_filtered)
