import os
import pandas as pd
import streamlit as st
import plotly.express as px
import google.generativeai as genai
from supabase import create_client, Client
from datetime import datetime

# --- 1. SETUP KONFIGURASI & API ---
st.set_page_config(page_title="Dashboard Final Penjualan & AI Insights", layout="wide")

try:
    # 1. Ambil kunci dari secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    
    # 2. Setup Gemini AI
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
except Exception as e:
    st.error(f"Masalah Konfigurasi Secrets: {e}")
    st.stop()

# --- 2. SETUP SUPABASE ---
@st.cache_resource
def init_supabase():
    # Fungsi ini membuat koneksi ke Supabase
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# PENTING: Baris di bawah ini harus ada agar variabel 'supabase' bisa dipakai di seluruh kode
supabase = init_supabase()

# --- 3. DATA FETCHING ---
@st.cache_data(ttl=600)
def load_data():
    try:
        # Sekarang 'supabase' sudah didefinisikan di atas, jadi tidak akan error lagi
        res = supabase.table("datapenjualanbaru").select("*").execute()
        df = pd.DataFrame(res.data)
        
        if df.empty:
            return df
            
        # Konversi tanggal & angka (seperti kode sebelumnya)
        if 'cancel_time' in df.columns:
            df['cancel_time'] = pd.to_datetime(df['cancel_time'], errors='coerce')
        
        cols_to_fix = ['order_amount', 'total_refund', 'original_price', 'total_discount']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Supabase: {e}")
        return pd.DataFrame()

# Lanjutkan ke bagian filter dan dashboard...
        # Konversi kolom tanggal
        if 'cancel_time' in df.columns:
            df['cancel_time'] = pd.to_datetime(df['cancel_time'], errors='coerce')
            # Buat kolom bulan-tahun untuk tren
            df['month_year'] = df['cancel_time'].dt.to_period('M').astype(str)
        
        # Pastikan kolom numerik
        cols_to_fix = ['order_amount', 'total_refund', 'original_price', 'total_discount']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Supabase: {e}")
        return pd.DataFrame()

# --- 3. LOGIKA DASHBOARD ---
df_raw = load_data()

if df_raw.empty:
    st.warning("Data tidak ditemukan atau tabel kosong.")
else:
    # --- SIDEBAR (FILTER DATA) ---
    st.sidebar.header("🔍 Filter Data")
    
    # Filter Tanggal yang Aman
    min_date = df_raw['cancel_time'].min().date() if not df_raw['cancel_time'].isnull().all() else datetime.now().date()
    max_date = df_raw['cancel_time'].max().date() if not df_raw['cancel_time'].isnull().all() else datetime.now().date()
    
    date_range = st.sidebar.date_input("Rentang Waktu", [min_date, max_date])

    # Dynamic Multiselect Filters
    st.sidebar.subheader("Filter Spesifik")
    selected_category = st.sidebar.multiselect("Kategori Produk", sorted(df_raw['product_category'].unique().tolist()))
    selected_city = st.sidebar.multiselect("Kota (Regency/City)", sorted(df_raw['regency_city'].unique().tolist()))
    selected_cancel_reason = st.sidebar.multiselect("Alasan Cancel", sorted(df_raw['cancel_reason'].unique().tolist()))

    # --- PROSES FILTERING ---
    df_filtered = df_raw.copy()
    
    # Cek apakah user sudah memilih start dan end date
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['cancel_time'].dt.date >= date_range[0]) & 
                                  (df_filtered['cancel_time'].dt.date <= date_range[1])]
    
    if selected_category:
        df_filtered = df_filtered[df_filtered['product_category'].isin(selected_category)]
    if selected_city:
        df_filtered = df_filtered[df_filtered['regency_city'].isin(selected_city)]
    if selected_cancel_reason:
        df_filtered = df_filtered[df_filtered['cancel_reason'].isin(selected_cancel_reason)]
       

   # --- 3. ANALISIS CLUSTERING (K-MEANS) ---
def perform_clustering(df):
    if df.empty:
        return None

    # 1. Agregasi Data per Produk
    df_prod = df.groupby('platform_sku_variation').agg({
        'order_amount': 'sum',
        'total_refund': 'sum',
        'platform_sku_variation': 'count'
    }).rename(columns={'platform_sku_variation': 'transaction_count'}).reset_index()

    # 2. Fitur untuk Clustering
    features = ['transaction_count', 'total_refund', 'order_amount']
    x = df_prod[features]

    # 3. Normalisasi Data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 4. Jalankan K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_prod['cluster_id'] = kmeans.fit_predict(x_scaled)

    # 5. Mapping Cluster
    cluster_means = df_prod.groupby('cluster_id')[features].mean()
    
    idx_bermasalah = cluster_means['total_refund'].idxmax()
    remaining_indices = [i for i in range(3) if i != idx_bermasalah]
    idx_unggulan = cluster_means.loc[remaining_indices, 'transaction_count'].idxmax()
    idx_evaluasi = [i for i in range(3) if i not in [idx_bermasalah, idx_unggulan]][0]

    mapping = {
        idx_bermasalah: "Cluster 0 – Produk Bermasalah (High Return)",
        idx_unggulan: "Cluster 1 – Produk Unggulan (High Transaction)",
        idx_evaluasi: "Cluster 2 – Produk Evaluasi (Low Transaction)"
    }
    
    df_prod['Cluster Name'] = df_prod['cluster_id'].map(mapping)
    return df_prod

    # --- 4. MAIN DASHBOARD ---
    st.title("📊 Sales & Return Final Dashboard")
    
    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    total_sales = df_filtered['order_amount'].sum()
    total_refund = df_filtered['total_refund'].sum()
    total_orders = len(df_filtered)
    refund_rate = (total_refund / total_sales * 100) if total_sales > 0 else 0

    m1.metric("Total Omset", f"Rp {total_sales:,.0f}")
    m2.metric("Total Refund", f"Rp {total_refund:,.0f}")
    m3.metric("Jumlah Transaksi", f"{total_orders:,}")
    m4.metric("Refund Rate", f"{refund_rate:.1f}%")

    st.markdown("---")

    # Visualisasi
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Tren Penjualan Bulanan")
        # Kelompokkan berdasarkan month_year agar urutan benar
        df_trend = df_filtered.groupby('month_year')['order_amount'].sum().reset_index()
        fig_line = px.line(df_trend, x='month_year', y='order_amount', 
                          title="Perkembangan Omset", markers=True,
                          labels={'month_year': 'Bulan', 'order_amount': 'Total Omset'})
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        st.subheader("🍰 Komposisi Kategori Produk")
        fig_pie = px.pie(df_filtered, names='product_category', values='order_amount', 
                         title="Proporsi Omset per Kategori", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("---")
    st.subheader("🎯 Product Segmentation (K-Means Clustering)")
    
    df_cluster = perform_clustering(df_filtered)
    
    if df_cluster is not None:
        # Visualisasi Scatter Plot
        fig_cluster = px.scatter(
            df_cluster, 
            x="transaction_count", 
            y="total_refund",
            color="Cluster Name",
            size="order_amount",
            hover_name="platform_sku_variation",
            title="Mapping Produk: Transaksi vs Refund",
            labels={
                "transaction_count": "Jumlah Transaksi",
                "total_refund": "Total Nilai Refund (Rp)",
                "Cluster Name": "Kategori Cluster"
            },
            color_discrete_map={
                "Cluster 0 – Produk Bermasalah (High Return)": "#EF553B",
                "Cluster 1 – Produk Unggulan (High Transaction)": "#00CC96",
                "Cluster 2 – Produk Evaluasi (Low Transaction)": "#636EFA"
            }
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Penjelasan Cluster dalam Kolom
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.error("⚠️ **Cluster 0: Produk Bermasalah**")
            st.caption("Prioritas perbaikan kualitas & pengemasan karena tingkat return tinggi.")
            st.write(df_cluster[df_cluster['cluster_id'] == 0]['platform_sku_variation'].head(5))

        with c2:
            st.success("🌟 **Cluster 1: Produk Unggulan**")
            st.caption("Pertahankan kualitas dan jadikan fokus promosi (Best Seller).")
            st.write(df_cluster[df_cluster['cluster_id'] == 1]['platform_sku_variation'].head(5))

        with c3:
            st.info("🔍 **Cluster 2: Produk Evaluasi**")
            st.caption("Perlu tinjauan harga atau efisiensi katalog karena transaksi rendah.")
            st.write(df_cluster[df_cluster['cluster_id'] == 2]['platform_sku_variation'].head(5))

    # --- 5. AI INSIGHTS ---
    st.markdown("---")
    st.subheader("🤖 AI Business Advisor")
    
    if st.button("Minta Saran Strategis AI"):
        if df_filtered.empty:
            st.error("Data terfilter kosong, AI tidak bisa menganalisis.")
        else:
            with st.spinner("Sedang menganalisis data Anda..."):
                top_category = df_filtered.groupby('product_category')['order_amount'].sum().idxmax()
                most_cancel = df_filtered['cancel_reason'].mode()[0] if not df_filtered['cancel_reason'].mode().empty else "Tidak ada data"
                
                prompt = f"""
                Anda adalah konsultan bisnis senior profesional. Berikan analisis mendalam terhadap data ini:
                - Total Omset: Rp {total_sales:,.0f}
                - Total Refund: Rp {total_refund:,.0f} (Refund Rate: {refund_rate:.1f}%)
                - Total Transaksi: {total_orders}
                - Kategori Terlaris: {top_category}
                - Alasan Pembatalan Utama: {most_cancel}
                
                Tolong berikan:
                1. Analisis Performa: Apakah refund rate ini wajar?
                2. Strategi Operasional: Apa yang harus dilakukan untuk mengurangi alasan pembatalan '{most_cancel}'?
                3. Perspektif Etika Bisnis (Syariah): Bagaimana mengelola pengembalian barang agar tetap adil bagi penjual dan pembeli (prinsip saling ridha)?
                """
                
                try:
                    response = model.generate_content(prompt)
                    st.success("Analisis AI Berhasil!")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gagal menghubungi AI: {e}")

    # Preview Data
    with st.expander("🔍 Lihat Detail Data Terfilter"):
        st.dataframe(df_filtered, use_container_width=True)
