import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Nomophobia Dashboard",
    page_icon="📱",
    layout="wide"
)
@st.cache_resource
def load_clustering_model():
    scaler = joblib.load("scaler_clustering.pkl")
    kmeans = joblib.load("kmeans_clustering.pkl")
    return scaler, kmeans

scaler, kmeans = load_clustering_model()
@st.cache_resource
def load_likert_cols():
    # likert_cols harus LIST kolom, bukan model
    return [
        "cek_setelah_bangun", "khawatir_baterai_habis", "panik_saat_kuota_internet_habis",
        "tidak tenang ketika sinyal hilang", "cemas_tidak_bisa_medsos", "takut_melewatkan_informasi",
        "pantau_berita_setiap_saat", "tidak_nyaman_tanpa_smartphone", "bingung_tanpa_smartphone",
        "nyaman_dekat_smartphone", "bawa_smartphone_kemana_mana", "panik_hp_tertinggal",
        "marah_smartphone_bermasalah", "kehilangan_kendali_tanpa_smartphone"
    ]

likert_cols = load_likert_cols()

skala_jawaban = {
    "Sangat Tidak Setuju": 1,
    "Tidak Setuju": 2,
    "Netral": 3,
    "Setuju": 4,
    "Sangat Setuju": 5,
}

label_pernyataan = {
    "cek_setelah_bangun": "Saya langsung mengecek smartphone setelah bangun tidur",
    "khawatir_baterai_habis": "Saya khawatir jika baterai smartphone saya hampir habis",
    "panik_saat_kuota_internet_habis": "Saya panik jika kuota internet saya habis",
    "tidak tenang ketika sinyal hilang": "Saya merasa tidak tenang ketika sinyal smartphone hilang",
    "cemas_tidak_bisa_medsos": "Saya cemas jika tidak bisa mengakses media sosial",
    "takut_melewatkan_informasi": "Saya takut melewatkan informasi penting tanpa smartphone",
    "pantau_berita_setiap_saat": "Saya sering memantau berita atau informasi melalui smartphone",
    "tidak_nyaman_tanpa_smartphone": "Saya merasa tidak nyaman ketika jauh dari smartphone",
    "bingung_tanpa_smartphone": "Saya merasa bingung ketika tidak memegang smartphone",
    "nyaman_dekat_smartphone": "Saya merasa lebih nyaman jika smartphone selalu dekat dengan saya",
    "bawa_smartphone_kemana_mana": "Saya selalu membawa smartphone kemanapun saya pergi",
    "panik_hp_tertinggal": "Saya panik ketika menyadari smartphone tertinggal",
    "marah_smartphone_bermasalah": "Saya merasa marah atau kesal ketika smartphone bermasalah",
    "kehilangan_kendali_tanpa_smartphone": "Saya merasa kehilangan kendali ketika tidak bisa menggunakan smartphone",
}

likert_cols = [c for c in likert_cols if c in label_pernyataan]

# Mapping Saran Spesifik untuk mengurangi kuantitas perilaku
saran_spesifik_db = {
    "khawatir_baterai_habis": "💡 Tips Baterai: Jangan bawa charger/powerbank untuk perjalanan singkat. Latih diri Anda merasa aman meski baterai di bawah 20%.",
    "panik_saat_kuota_internet_habis": "💡 Tips Kuota: Unduh lagu/film secara offline. Biasakan diri bahwa 'offline' bukan berarti terputus dari dunia.",
    "cemas_tidak_bisa_medsos": "💡 Detox Medsos: Jadwalkan 'jam bebas medsos' (misal: jam 8-10 malam). Ganti scrolling dengan membaca buku fisik.",
    "takut_melewatkan_informasi": "💡 Atasi FOMO: Berita penting pasti akan sampai ke Anda. Batasi cek portal berita cukup 2x sehari saja.",
    "panik_hp_tertinggal": "💡 Latihan Lepas: Coba tinggalkan HP di rumah saat pergi ke warung atau tempat dekat sebentar saja.",
    "marah_smartphone_bermasalah": "💡 Kendali Emosi: Saat HP lemot, tarik napas dalam. Sadari bahwa gadget hanyalah alat, bukan pengendali mood Anda.",
    "kehilangan_kendali_tanpa_smartphone": "💡 Zona Larangan: Buat aturan tegas: Dilarang pegang HP di meja makan atau kamar mandi.",
    "cek_setelah_bangun": "💡 Rutinitas Pagi: Beli jam weker fisik. Jangan jadikan HP sebagai benda pertama yang disentuh saat membuka mata.",
    "tidak tenang ketika sinyal hilang": "💡 Nikmati Momen: Saat sinyal hilang, lihat sekeliling. Itu tanda semesta menyuruh Anda istirahat sejenak.",
    "pantau_berita_setiap_saat": "💡 Filter Informasi: Matikan notifikasi aplikasi berita. Anda yang mencari info, bukan info yang mengejar Anda.",
    "tidak_nyaman_tanpa_smartphone": "💡 Hobi Analog: Cari kesibukan yang membutuhkan kedua tangan (memasak, merakit gundam, berkebun) agar tidak sempat pegang HP.",
    "bingung_tanpa_smartphone": "💡 Catatan Fisik: Selalu bawa buku catatan kecil dan pena. Jangan gantungkan ingatan 100% pada HP.",
    "nyaman_dekat_smartphone": "💡 Jaga Jarak: Saat bekerja/belajar, letakkan HP di ruangan lain atau setidaknya sejauh 3 meter dari jangkauan tangan.",
    "bawa_smartphone_kemana_mana": "💡 Lepas Berkala: Mulai dari hal kecil: Jangan bawa HP saat ke toilet atau saat mengambil minum di dapur."
}

rekomendasi_db = {
    "Nomophobia Tinggi": [
        {"icon": "🛑", "title": "Puasa Gadget Ekstrem", "desc": "Wajib: 1-2 jam sehari tanpa menyentuh HP sama sekali."},
        {"icon": "🛏️", "title": "Zona Bebas HP", "desc": "Larang HP masuk kamar tidur. Gunakan jam weker fisik."},
        {"icon": "🌳", "title": "Alam & Fisik", "desc": "Lakukan olahraga outdoor tanpa membawa HP."}
    ],
    "Nomophobia Sedang": [
        {"icon": "🔕", "title": "Matikan Notifikasi", "desc": "Nonaktifkan notifikasi medsos agar tidak terdistraksi."},
        {"icon": "⏳", "title": "Screen Time Limit", "desc": "Gunakan fitur pembatas waktu aplikasi harian (Digital Wellbeing)."},
        {"icon": "📚", "title": "Aktivitas Offline", "desc": "Membaca buku fisik atau memasak tanpa melihat tutorial HP."}
    ],
    "Nomophobia Rendah": [
        {"icon": "✅", "title": "Pertahankan", "desc": "Kebiasaan Anda sudah seimbang. Terus jaga pola ini."},
        {"icon": "🤝", "title": "Quality Time", "desc": "Fokus tatap muka saat bersama teman/keluarga."},
        {"icon": "🧠", "title": "Mindfulness", "desc": "Gunakan teknologi sebagai alat, bukan tuan."}
    ]
}

# Mapping kategori → key rekomendasi
kategori_rekom_map = {
    "Nomophobia Tinggi": "Nomophobia Tinggi",
    "Nomophobia Sedang": "Nomophobia Sedang",
    "Nomophobia Rendah": "Nomophobia Rendah",
    "Tinggi": "Nomophobia Tinggi",
    "Sedang": "Nomophobia Sedang",
    "Rendah": "Nomophobia Rendah",
    "tinggi": "Nomophobia Tinggi",
    "sedang": "Nomophobia Sedang",
    "rendah": "Nomophobia Rendah",
    "High": "Nomophobia Tinggi",
    "Medium": "Nomophobia Sedang",
    "Low": "Nomophobia Rendah",
}
# --- LOAD MODEL ---
@st.cache_resource
def load_xgb_model():
    return joblib.load("xgb_nomophobia_model.joblib")

model = load_xgb_model()

# --- 2. MENU NAVIGASI (SIDEBAR YANG DIPERBAIKI) ---
with st.sidebar:
    # A. Branding / Logo di Atas
    # Menggunakan gambar ilustrasi agar sidebar tidak kosong
    # st.image("https://img.freepik.com/free-vector/nomophobia-concept-illustration_114360-1296.jpg", use_column_width=True)
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #4CAF50; margin:0;">Nomophobia</h2>
            <p style="font-size: 14px; color: #666; margin:0;">Detector & Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # B. Menu Navigasi Utama
    selected = option_menu(
        menu_title=None,  # Kita sembunyikan judul default agar lebih bersih
        options=["Beranda", "Dataset & Statistik","Exploratory Data Analysis","Prediksi & Analisis", "Tentang", "Prediksi Score Nomophobia"], 
        icons=["house", "activity", "bar-chart-fill", "info-circle"], 
        menu_icon="cast", 
        default_index=0, 
        orientation="vertical",
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "transparent" # Transparan agar menyatu dengan sidebar
            },
            "icon": {
                "color": "#4CAF50", # Warna ikon hijau agar senada
                "font-size": "20px"
            }, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "5px 0px", # Jarak antar tombol
                "padding": "10px 15px",
                "border-radius": "8px", # Sudut melengkung
                "--hover-color": "#f0f2f6" # Warna saat mouse hover
            },
            "nav-link-selected": {
                "background-color": "#4CAF50", # Hijau utama
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0px 4px 6px rgba(0,0,0,0.1)" # Efek bayangan
            },
        }
    )
    
    # C. Footer / Copyright di Bawah
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 12px; color: #888;">
            <p>© 2024 Nomophobia Detector<br>v1.0 - Beta Release</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- 3. HALAMAN: BERANDA (ULTIMATE VERSION) ---
if selected == "Beranda":
    # ==========================================
    # 0. STYLE ENGINE
    # ==========================================
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
            max-width: 100%;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4CAF50, #00E5FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(76, 175, 80, 0.5);
        }
        .glass-card {
            background: rgba(30, 30, 40, 0.6);
            border-left: 4px solid #4CAF50;
            padding: 15px 20px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 15px;
            font-size: 0.9rem;
            color: #E0E0E0;
            width: 100%;
            display: flex;
            min-height: 140px !important;
            flex-direction: column;
            justify-content: center;
        }
        .highlight-text {
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<h1 class="hero-title">🚀 Dashboard Monitoring</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle" style="color:#b0bec5; margin-top:-5px;">Real-time Nomophobia Risk & Behavior Analysis</p>', unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="text-align:right; padding-top:10px;">
            <span style="background:#262730; padding:5px 10px; border-radius:20px; font-size:12px; border:1px solid #4CAF50; color:#4CAF50;">
                🟢 Kelompok 7
            </span>
        </div>
        """, unsafe_allow_html=True)

    # --- DATA ENGINE ---
    @st.cache_data
    def load_data():
        try:
            return pd.read_excel("dataset_bersih1.xlsx")
        except:
            return None

    df = load_data()
    
    if df is not None:
        df_filtered = df.copy()

        # ==========================================
        # 1. CALCULATION ENGINE
        # ==========================================
        
        # A. Hitung Modus Durasi
        if 'rentang_durasi_label' in df_filtered.columns:
            durasi_dominan = df_filtered['rentang_durasi_label'].mode()[0]
        else:
            durasi_dominan = "N/A"
            
        # B. Hitung Modus Kategori & Persentase
        if 'kategori_nomophobia' in df_filtered.columns:
            top_category = df_filtered['kategori_nomophobia'].mode()[0]
            count_top = df_filtered[df_filtered['kategori_nomophobia'] == top_category].shape[0]
            top_cat_pct = (count_top / len(df_filtered)) * 100
        else:
            top_category = "N/A"
            top_cat_pct = 0
            
        # C. Hitung Modus waktu intens
        if 'waktu_paling_intens' in df_filtered.columns:
            waktu_dominan = df_filtered['waktu_paling_intens'].mode()[0]
        else:
            waktu_dominan = "N/A"
        
        # D. Hitung Modus rentang & Persentase
        if 'rentang_usia_label' in df_filtered.columns:
            usia_dominan = df_filtered['rentang_usia_label'].mode()[0]
            count_usia = df_filtered[df_filtered['rentang_usia_label'] == usia_dominan].shape[0]
            usia_pct = (count_usia / len(df_filtered)) * 100
        else:
            usia_dominan = "N/A"
            usia_pct = 0

        # ==========================================
        # 2. KPI SECTION
        # ==========================================
        st.write("") 
        k1, k2, k3 = st.columns(3)

        with k1: st.metric("👥 Total Responden", f"{len(df_filtered)}")
        with k2: 
            avg_score = df_filtered['skor_nomophobia'].mean() if 'skor_nomophobia' in df_filtered.columns else 0
            st.metric("📊 Skor Rata-rata Nomophobia", f"{avg_score:.1f}")
        with k3: 
            st.metric("⏳ Modus Durasi Penggunaan Smartphone", f"{durasi_dominan}")

        # ==========================================
        # 3. CHART SECTION (3 KOLOM SEJAJAR)
        # ==========================================
        st.write("---")
        col_left, col_mid, col_right = st.columns(3)

        # --- KOLOM KIRI: DONUT CHART (KATEGORI) ---
        with col_left:
            st.markdown("##### 🍩 Distribusi Kategori")
            if 'kategori_nomophobia' in df_filtered.columns:
                pie_data = df_filtered['kategori_nomophobia'].value_counts().reset_index()
                pie_data.columns = ['Kategori', 'Jumlah']
                
                COLOR_MAP = {
                    "Nomophobia Berat": "#FF2B2B", "Berat": "#FF2B2B",
                    "Nomophobia Sedang": "#FF9F1C", "Sedang": "#FF9F1C",
                    "Nomophobia Ringan": "#2EC4B6", "Ringan": "#2EC4B6",
                    "Nomophobia Rendah": "#3A86FF", "Rendah": "#3A86FF"
                }

                fig_donut = px.pie(pie_data, values='Jumlah', names='Kategori', hole=0.5,
                                   color='Kategori', color_discrete_map=COLOR_MAP)
                
                fig_donut.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=300, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                
                # Angka Tengah Donut
                fig_donut.add_annotation(text=f"{len(df_filtered)}", showarrow=False, 
                                         font=dict(size=24, color="#FFFFFF", family="Arial Black"), yshift=5)
                
                st.plotly_chart(fig_donut, use_container_width=True)

        # --- KOLOM TENGAH: BOX PLOT (USIA vs SKOR) ---
        with col_mid:
            st.markdown("##### 📉 Skor per Usia")
            if 'rentang_usia_label' in df_filtered.columns and 'skor_nomophobia' in df_filtered.columns:
                urutan_usia = ["< 18 tahun", "18 – 25 tahun", "26 – 35 tahun", "36 – 45 tahun", "> 45 tahun"]
                
                fig_box = px.box(df_filtered, 
                                 x='rentang_usia_label', 
                                 y='skor_nomophobia',
                                 color='rentang_usia_label',
                                 category_orders={"rentang_usia_label": urutan_usia},
                                 points="outliers") 
                
                fig_box.update_layout(
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title=None,
                    yaxis_title="Skor",
                    margin=dict(t=10, b=30, l=0, r=0),
                    height=300,
                    xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_box, use_container_width=True)

        # --- KOLOM KANAN: TREEMAP (WAKTU AKTIVITAS) ---
        with col_right:
            st.markdown("##### 🕒 Peta Waktu Aktivitas")
            if 'waktu_paling_intens' in df_filtered.columns:
                # Mapping Label Waktu
                time_mapping = {
                    1: "Pagi (06.00 – 11.59)",
                    2: "Siang (12.00 – 17.59)",
                    3: "Malam (18.00 – 23.59)",
                    4: "Larut malam (00.00 – 05.59)",
                    5: "Sepanjang hari"
                }
                df_filtered['waktu_label'] = df_filtered['waktu_paling_intens'].map(time_mapping)
                waktu_intens=time_mapping.get(waktu_dominan, "N/A")
                
                waktu_counts = df_filtered['waktu_label'].value_counts().reset_index()
                waktu_counts.columns = ['Waktu', 'Jumlah']
                
                fig_tree = px.treemap(waktu_counts, 
                                      path=['Waktu'], 
                                      values='Jumlah',
                                      color='Jumlah',
                                      color_continuous_scale='RdBu') 
                
                fig_tree.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                fig_tree.data[0].textinfo = "label+percent entry"
                st.plotly_chart(fig_tree, use_container_width=True)

        # ==========================================
        # 4. INSIGHT SECTION (GLASS CARDS)
        # ==========================================
        st.write("") 
        
        # Kalkulasi Insight Teks
        insight_durasi = "Pola penggunaan normal."
        if 'durasi_penggunaan' in df_filtered.columns and 'skor_nomophobia' in df_filtered.columns:
            df_heavy = df_filtered[df_filtered['durasi_penggunaan'] >= 4] 
            df_light = df_filtered[df_filtered['durasi_penggunaan'] <= 2] 
            
            if not df_heavy.empty and not df_light.empty:
                gap = df_heavy['skor_nomophobia'].mean() - df_light['skor_nomophobia'].mean()
                if gap > 2:
                    insight_durasi = f"User durasi tinggi (>10 jam) skornya <b style='color:#FF4081'>+{gap:.1f} poin</b> lebih besar."
                elif gap < -2:
                    insight_durasi = f"Unik! User durasi rendah justru skornya lebih tinggi {abs(gap):.1f} poin."

        # Layout Insight
       # ==========================================
        # 5. INSIGHT SECTION (4 KARTU: 2 ATAS, 2 BAWAH)
        # ==========================================
        st.write("") 

        # --- LAYOUT VISUALISASI ---
        st.subheader("💡 Key Analysis")
        
        # BARIS 1 (Card 1 & 2)
        row1_col1, row1_col2 = st.columns(2)

        # Card 1
        with row1_col1:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size:16px; margin-bottom:5px;">📌 <b>Dominasi Nomophobia</b></div>
                Sebanyak <span class="highlight-text" style="font-size:18px;">{top_cat_pct:.1f}%</span> responden berada pada kategori 
                <span style="color:#FFF; background:#4CAF50; padding:2px 6px; border-radius:4px; font-size:12px;">
                    {top_category.upper()}
                </span>,
                yang merupakan kelompok dengan proporsi tertinggi.
            </div>
            """, unsafe_allow_html=True)


        # Card 2
        with row1_col2:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size:16px; margin-bottom:5px;">📌 <b>Dominasi Usia dan Nomophobia</b></div>
                Sebanyak <span class="highlight-text" style="font-size:18px;">{usia_pct:.1f}%</span> responden berada pada rentang usia 
                <span style="color:#FFF; background:#4CAF50; padding:2px 6px; border-radius:4px; font-size:12px;">
                    {usia_dominan.upper()}
                </span>
                dengan kategori Nomophobia yang paling dominan.
            </div>
            """, unsafe_allow_html=True)


        # BARIS 2 (Card 3 & 4)
        row2_col1, row2_col2 = st.columns(2)

        # Card 3:
        with row2_col1:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #FFD600;">
                <div style="font-size:16px; margin-bottom:6px;">⏱️ <b>Durasi Pemakaian Smartphone</b></div>
                Mayoritas responden durasi penggunaan smartphone pada rentang 
                <b style="color:#FFD600">{durasi_dominan}</b> per hari.
                {insight_durasi}
            </div>
            """, unsafe_allow_html=True)


        # Card 4
        with row2_col2:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #FF4081;">
                <div style="font-size:16px; margin-bottom:6px;">⏱️ <b>Waktu Penggunaan Paling Intens</b></div>
                Waktu penggunaan smartphone yang paling dominan terjadi pada 
                <b style="color:#FF4081">{waktu_intens}</b>.
                {insight_durasi}
            </div>
            """, unsafe_allow_html=True)

            
    else:
        st.error("Data tidak ditemukan! Pastikan file excel 'dataset_bersih1.xlsx' ada.")


# --- 3.HALAMAN: DATASET & STATISTIK (INTERAKTIF LOKAL) ---
if selected == "Dataset & Statistik": 
    st.title("📊 Eksplorasi Data Interaktif ")
    st.markdown("""
    Page ini memberikan pandangan menyeluruh terhadap **seluruh dataset**. 
    Anda dapat melakukan filter spesifik langsung pada panel visualisasi di bawah.
    """)

    # --- LOAD DATA ---
    @st.cache_data
    def load_data():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "Dataset Hubungan Penggunaan Smartphone dengan nemophobia  - Sheet1.csv")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            return None
        except Exception as e:
            st.error(f"Error membaca Excel: {e}")
            return None

    df = load_data() 

    if df is None:
        st.error("⚠️ File `Dataset Hubungan Penggunaan Smartphone dengan nemophobia  - Sheet1.csv` tidak ditemukan.")
    else:
        # -----------------------------------------------------------------------------
        # A. INFORMASI DATASET
        # -----------------------------------------------------------------------------
        st.header("Informasi Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Sumber Data**")
            st.markdown("🔗 **[Survey Kuesioner Online (Google Form)](https://docs.google.com/forms/d/e/1FAIpQLSfTtocuCjbC_5CtmpgOyH_4Mv0PxQ61Yo1ES0nLRBcGJUBnMg/viewform?usp=dialog)**")
            st.caption("Periode: November - Desemer 2025")
        
        with col2:
            st.info("**Dimensi Data**")
            st.metric(label="Jumlah Responden (Baris)", value=df.shape[0])
        with col3:
            st.info("**Dimensi Data**")
            st.metric(label="Jumlah Fitur (Kolom)", value=df.shape[1])
            
        st.markdown("---")

        # -----------------------------------------------------------------------------
        # B. TABEL DATASET
        # -----------------------------------------------------------------------------
        # Preview Data
        st.subheader("Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
       # -----------------------------------------------------------------------------
        # C. DESKRIPSI VARIABEL
        # -----------------------------------------------------------------------------
        st.header("C. Deskripsi Variabel")
        
        # Membuat DataFrame manual untuk Kamus Data
        data_dict = [
            {"Variabel": "Nama", "Deskripsi": "Nama responden", "Tipe": "String"},
            {"Variabel": "gender", "Deskripsi": "Jenis kelamin responden", "Tipe": "Categorical (Nominal)"},
            {"Variabel": "rentang_usia", "Deskripsi": "Kategori umur responden", "Tipe": "Categorical (Nominal)"},
            {"Variabel": "pekerjaan_utama", "Deskripsi": "Pekerjaan utama responden", "Tipe": "Categorical (Nominal)"},
            {"Variabel": "durasi_penggunaan", "Deskripsi": "Rata-rata penggunaan HP per hari", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "frekuensi_buka_smartphone", "Deskripsi": "Frekuensi buka HP dalam 1 jam", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "waktu_paling_intens", "Deskripsi": "Waktu paling sering menggunakan HP", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "aktivitas_utama", "Deskripsi": "Aktivitas utama di HP", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "frekuensi_cek_notifikasi", "Deskripsi": "Seberapa sering responden memeriksa notifikasi meskipun smartphone tidak berbunyi/getar.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "frekuensi_gunakan_saat_berbicara", "Deskripsi": "Seberapa sering responden tetap menggunakan smartphone ketika sedang berbicara dengan orang lain.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "cek_setelah_bangun", "Deskripsi": "Tingkat kebiasaan memeriksa smartphone segera setelah bangun tidur.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "khawatir_baterai_habis", "Deskripsi": "Tingkat kekhawatiran responden ketika baterai smartphone hampir habis.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "panik_saat_kuota_internet_habis ", "Deskripsi": "Tingkat kepanikan responden ketika kuota internet habis.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "tidak tenang ketika sinyal hilang", "Deskripsi": "Tingkat ketidaknyamanan/kecemasan responden ketika kehilangan sinyal seluler.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "cemas_tidak_bisa_medsos", "Deskripsi": "Tingkat kecemasan ketika responden tidak dapat mengakses media sosial.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "takut_melewatkan_informasi", "Deskripsi": "Tingkat ketakutan melewatkan informasi penting tanpa smartphone", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "pantau_berita_setiap_saat", "Deskripsi": "Seberapa besar kebutuhan responden untuk memantau berita/update kapan saja.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "tidak_nyaman_tanpa_smartphone", "Deskripsi": "Tingkat ketidaknyamanan berkomunikasi tanpa bantuan smartphone", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "bingung_tanpa_smartphone", "Deskripsi": "Tingkat kebingungan ketika tidak dapat menggunakan smartphone dalam aktivitas sehari-hari.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "nyaman_dekat_smartphone", "Deskripsi": "Tingkat kenyamanan jika smartphone berada dekat responden", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "bawa_smartphone_kemana_mana", "Deskripsi": "Kebiasaan membawa smartphone ke mana-mana meskipun tidak diperlukan.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "panik_hp_tertinggal", "Deskripsi": "Tingkat kepanikan ketika smartphone tertinggal di rumah.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "marah_smartphone_bermasalah", "Deskripsi": "Tingkat kemarahan ketika smartphone mengalami masalah/error.", "Tipe": "Categorical (Ordinal)"},
            {"Variabel": "kehilangan_kendali_tanpa_smartphone", "Deskripsi": "Tingkat perasaan kehilangan kendali saat smartphone berada di luar jangkauan", "Tipe": "Categorical (Ordinal)"},
        ]
        df_dict = pd.DataFrame(data_dict)
        
        with st.expander("Lihat Kamus Data Lengkap", expanded=True):
            # GANTI st.table DENGAN st.dataframe
            st.dataframe(
                df_dict, 
                use_container_width=True, 
                height=400,       # Tinggi dalam pixel, jika data lebih panjang maka muncul scroll
                hide_index=True   # Menyembunyikan index angka (0,1,2...) agar lebih rapi
            )
        
        st.markdown("---")        

# --- 4. HALAMAN: EXPLORATORY DATA ANALYSIS ---
if selected == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Analisis mendalam mengenai pola perilaku dan tingkat Nomophobia responden.")

    if 'df' not in locals() or df is None:
        try:
            df = pd.read_excel("dataset_bersih1.xlsx") # Sesuaikan nama file jika perlu
            # Preprocessing sederhana jika load mendadak
            order = ['Nomophobia Rendah', 'Nomophobia Sedang', 'Nomophobia Tinggi']
            df['kategori_nomophobia'] = pd.Categorical(df['kategori_nomophobia'], categories=order, ordered=True)
        except:
            st.error("Data tidak ditemukan. Harap pastikan file dataset tersedia.")
            st.stop()

    # --- MULAI VISUALISASI ---
    if df is not None:
        
        # 1️⃣ Distribusi Tingkat Nomophobia 
        st.subheader("1️⃣ Distribusi Tingkat Nomophobia")
        
        distribusi = df['kategori_nomophobia'].value_counts().reset_index()
        distribusi.columns = ['Kategori', 'Jumlah']
        distribusi['Persentase'] = (distribusi['Jumlah'] / distribusi['Jumlah'].sum()) * 100
        distribusi = distribusi.sort_values('Kategori') 

        tab1, tab2 = st.tabs(["📊 Bar Chart", "🍩 Pie Chart"])
        
        with tab1:
            fig_bar = px.bar(distribusi, x='Kategori', y='Jumlah', text_auto=True, 
                             color='Kategori', title="Jumlah Responden per Kategori")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with tab2:
            fig_pie = px.pie(distribusi, values='Jumlah', names='Kategori', hole=0.4,
                             color='Kategori', title="Proporsi Kategori")
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        top_cat = distribusi.sort_values('Jumlah', ascending=False).iloc[0]
        st.info(f"💡 **Insight:** Mayoritas responden berada pada kategori **{top_cat['Kategori']}** ({top_cat['Persentase']:.1f}%).")
        st.markdown("---")

        # 2️⃣ USia vs Skor Nomophobia
        st.subheader(" 2️⃣ Peta Risiko: Di Usia Berapa Nomophobia Paling Mengancam?")
        
        # 1. Agregasi Data Manual (Menghitung % sendiri agar Plotly tidak salah hitung)
        # Kelompokkan data berdasarkan Usia dan Kategori, lalu hitung jumlah orangnya
        df_agg = df.groupby(['rentang_usia_label', 'kategori_nomophobia']).size().reset_index(name='jumlah_orang')
        
        # Hitung total orang per kelompok usia untuk mencari persentase
        df_total_per_usia = df_agg.groupby('rentang_usia_label')['jumlah_orang'].transform('sum')
        df_agg['persentase'] = (df_agg['jumlah_orang'] / df_total_per_usia) * 100
        
        # 2. Konfigurasi Tampilan
        urutan_usia = ['< 18 tahun', '18 – 25 tahun', '26 – 35 tahun', '36 – 45 tahun', '> 45 tahun']
        urutan_kategori = ['Nomophobia Rendah', 'Nomophobia Sedang', 'Nomophobia Tinggi']
        color_map_premium = {
            'Nomophobia Rendah': '#00C49A',
            'Nomophobia Sedang': '#FFB703',
            'Nomophobia Tinggi': '#D90429'
        }
        
        # 3. Membuat Chart dengan px.bar (Bukan histogram)
        # Kita pakai 'y=persentase' yang sudah kita hitung di atas
        fig_usia_final = px.bar(
            df_agg, 
            x='rentang_usia_label', 
            y='persentase', 
            color='kategori_nomophobia',
            text_auto='.1f', # Menampilkan angka desimal 1 digit
            
            # Mengunci urutan
            category_orders={
                'rentang_usia_label': urutan_usia,
                'kategori_nomophobia': urutan_kategori
            },
            color_discrete_map=color_map_premium
        )
        
        # 4. Styling Akhir
        fig_usia_final.update_layout(
            title={
                'text': "<b>Komposisi Tingkat Kecemasan per Generasi</b>",
                'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            template="plotly_white",
            bargap=0.3,
            
            # Sumbu Y: Kita set max 100 agar rapi
            yaxis=dict(
                title="Persentase (%)",
                range=[0, 100], 
                showgrid=False
            ),
            xaxis=dict(title=None, showgrid=False),
            
            legend=dict(
                orientation="h", y=1.02, x=0.5, xanchor="center", title=None
            ),
            margin=dict(t=80, l=20, r=20, b=50)
        )
        
        # Mempercantik label angka
        fig_usia_final.update_traces(
            textfont_size=14, 
            textfont_color="white",
            textposition='inside',
            insidetextanchor='middle',
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: <b>%{y:.1f}%</b><extra></extra>"
        )
        
        # 5. Tampilkan
        st.plotly_chart(fig_usia_final, use_container_width=True)
        
        # 6. Insight
        st.info(f"💡 **Insight:** Grafik ini sekarang menunjukkan proporsi yang sebenarnya. Perhatikan seberapa besar balok Merah di setiap usia. Semakin tinggi balok merahnya, semakin besar persentase orang di usia tersebut yang mengalami Nomophobia Tinggi.")
        st.markdown("---")

        # 3️⃣ Durasi Penggunaan HP vs Skor Nomophobia
        st.subheader("3️⃣ Durasi Penggunaan vs Skor Nomophobia")
            
        # Order durasi agar urut
        urutan_durasi = ['< 3 jam', '3 – 6 jam', '7 – 10 jam', '11 – 14 jam', '> 14 jam']
        
        fig_box = px.box(df, x='rentang_durasi_label', y='skor_nomophobia', 
                         color='rentang_durasi_label', 
                         category_orders={'rentang_durasi_label': urutan_durasi},
                         title="Distribusi Skor Nomophobia berdasarkan Durasi Penggunaan")
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.info("💡 **Insight:** Berdasarkan data ini, intervensi untuk mengurangi Nomophobia sebaiknya tidak hanya fokus pada mengurangi jam penggunaan  (digital detox), karena pengguna durasi rendah pun banyak yang cemas. Fokus sebaiknya diarahkan pada kualitas interaksi dan melatih mental untuk merasa nyaman saat terpisah dari perangkat (separation anxiety management).")
        st.markdown("---")


        # 4️⃣ Wakt Pa  Intens vs Kategori Nomoph
        st.subheader("4️⃣ Durasi Paling Intens")
        
        # Mapping Waktu 
        time_mapping = {
            1: "< 3 jam", 2: "3 – 6 jam", 3: "7 – 10 jam", 
            4: "11 – 14 jam", 5: "> 14 jam"
        }
        # Cek apakah kolom sudah dimapping atau masih angka
        if df['durasi_penggunaan'].dtype == 'int64' or df['durasi_penggunaan'].dtype == 'float64':
             df['waktu_label'] = df['durasi_penggunaan'].map(time_mapping)
        else:
             df['waktu_label'] = df['durasi_penggunaan'] # Jika sudah string
        
        waktu_cat = df.groupby(['waktu_label', 'kategori_nomophobia']).size().reset_index(name='Jumlah')
        
        fig_group = px.bar(waktu_cat, x='waktu_label', y='Jumlah', color='kategori_nomophobia',
                           barmode='relative', title="Durasi vs Kategori Nomophobia")
        st.plotly_chart(fig_group, use_container_width=True)

        st.info("💡 **Insight:** Orang dengan penggunaan wajar (3-6 jam) atau bahkan sedikit (<3 jam) pun mayoritas mengalami Nomophobia level Sedang hingga Tinggi. Hal ini mengindikasikan bahwa ketergantungan pada gadget lebih berkaitan dengan pola pikir atau keterikatan emosional daripada sekadar durasi waktu penggunaan.")
        st.markdown("---")


        
        # 5️⃣ Aktivitas Utama Penggunaan HP
        st.subheader("5️⃣ Aktivitas Dominan Pengguna")
        
        kolom_aktivitas = ['Bekerja/Belajar', 'Belanja online', 'Bermain Media sosial', 
                           'Bermain game', 'Chattingan', 'Mendengar Hiburan lain', 'Streaming video']
        
        # Filter kolom yang benar-benar ada di dataset
        kolom_ada = [col for col in kolom_aktivitas if col in df.columns]
        
        if kolom_ada:
            aktivitas_sum = df[kolom_ada].sum().reset_index()
            aktivitas_sum.columns = ['Aktivitas', 'Jumlah']
            aktivitas_sum = aktivitas_sum.sort_values('Jumlah', ascending=True)
            
            fig_act = px.bar(aktivitas_sum, x='Jumlah', y='Aktivitas', orientation='h',
                             title="Frekuensi Aktivitas Smartphone", text_auto=True)
            st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.warning("Kolom aktivitas tidak ditemukan untuk visualisasi ini.")
        st.markdown("---")

        
        # ---------------------------------------------------------
        # 6️⃣ Analisis Korelasi (Indikator Perilaku vs Skor Total)
        # ---------------------------------------------------------
        st.subheader("6️⃣ Analisis Korelasi: Indikator Perilaku")
        
        # 1. Daftar Kolom Likert (Pertanyaan Kuesioner)
        likert_cols = [
            "cek_setelah_bangun",
            "khawatir_baterai_habis",
            "panik_saat_kuota_internet_habis",
            "tidak tenang ketika sinyal hilang",
            "cemas_tidak_bisa_medsos",
            "takut_melewatkan_informasi",
            "pantau_berita_setiap_saat",
            "tidak_nyaman_tanpa_smartphone",
            "bingung_tanpa_smartphone",
            "nyaman_dekat_smartphone",
            "bawa_smartphone_kemana_mana",
            "panik_hp_tertinggal",
            "marah_smartphone_bermasalah",
            "kehilangan_kendali_tanpa_smartphone"
        ]
        
        # 2. Tambahkan Target Variabel (Skor Nomophobia) untuk perbandingan
        # Agar kita tahu hubungan setiap pertanyaan dengan total skor
        cols_to_analyze = likert_cols + ['skor_nomophobia']
        
        # 3. Filter: Pastikan hanya mengambil kolom yang benar-benar ada di Excel/CSV
        # (Mencegah error jika ada typo nama kolom)
        valid_cols = [col for col in cols_to_analyze if col in df.columns]
        
        if len(valid_cols) > 1:
            # 4. Hitung Korelasi
            corr_matrix = df[valid_cols].corr()
        
            # 5. Visualisasi Heatmap
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto='.2f',       # Menampilkan angka 2 desimal
                aspect="auto",
                color_continuous_scale='RdBu_r', # Merah-Putih-Biru
                zmin=-1, zmax=1,       # Skala warna seimbang (-1 s/d 1)
                title="Heatmap Korelasi: Indikator Perilaku & Skor Total"
            )
        
            # 6. Merapikan Layout
            fig_corr.update_layout(
                height=800, # Tinggi grafik diperbesar agar label terbaca jelas
                xaxis_tickangle=-45, # Miringkan label bawah
                margin=dict(l=20, r=20, t=50, b=100)
            )
        
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Insight
            st.info("💡 **Tips:** Perhatikan baris/kolom **'skor_nomophobia'**. Kotak yang berwarna **Merah Paling Gelap** menunjukkan perilaku (pertanyaan) yang paling kuat menyumbang tingginya tingkat Nomophobia seseorang.")
        
        else:
            st.error("⚠️ Kolom yang diminta tidak ditemukan di dataset. Cek kembali penulisan nama kolom.")
        
        st.markdown("---")

        # 7️⃣ Visualisasi Clustering Pengguna
        st.subheader("7️⃣ Segmentasi Pengguna (Clustering)")
        
        if 'cluster_label' in df.columns and 'skor_intensitas' in df.columns:
            fig_cluster = px.scatter(df, x='skor_intensitas', y='skor_nomophobia',
                                     color='cluster_label', symbol='cluster_label',
                                     title="Peta Sebaran Cluster Pengguna",
                                     hover_data=['rentang_durasi_label'])
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.info("💡 **Tips:** Clustering menunjukkan adanya segmentasi pengguna yang jelas, mulai dari pengguna ringan hingga heavy user dengan skor Nomophobia tinggi, memperkuat bahwa ketergantungan smartphone tidak bersifat homogen.")
        else:
            st.info("Visualisasi Cluster memerlukan kolom 'cluster_label'.")

    else:
        st.warning("Data belum dimuat. Silakan cek file dataset Anda.")
        

# --- 6. HALAMAN: Prediksi  ---
if selected == "Prediksi & Analisis":
    # ==========================================
    # STYLE ENGINE (SAMA)
    # ==========================================
    st.markdown("""
    <style>
    /* =========================================================
       0) LAYOUT & APP CONTAINER
    ========================================================= */
    .block-container{
      padding-top: 1.5rem !important;
      padding-bottom: 0rem !important;
      max-width: 100%;
    }
    
    /* =========================================================
       1) HERO (TITLE + SUBTITLE)
    ========================================================= */
    .hero-title{
      font-size: 2.2rem;
      font-weight: 800;
      margin-bottom: 0;
    
      background: linear-gradient(90deg, #4CAF50, #00E5FF);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hero-subtitle{
      color: #b0bec5;
      margin-top: -5px;
    }
    
    /* =========================================================
       2) COMMON TEXT UTILITIES
    ========================================================= */
    .highlight-text{
      color: #4CAF50;
      font-weight: 700;
    }
    
    /* =========================================================
       3) SECTION ELEMENTS (HEADER / SUBHEADER / DIVIDER)
    ========================================================= */
    .section-divider{
      height: 3px;
      border-radius: 2px;
      margin: 24px 0 16px 0;
      background: linear-gradient(90deg, #4CAF50, #00E5FF);
    }
    .section-header{
      font-size: 1.3rem;
      font-weight: 700;
      color: #4CAF50;
      margin: 16px 0 8px 0;
    }
    .section-subheader{
      font-size: 0.95rem;
      color: #9fb0ba;
      margin-bottom: 16px;
    }
    
    /* =========================================================
       4) INPUT / SELECT STYLING (BASEWEB)
    ========================================================= */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"]  > div{
      border-radius: 10px !important;
    }
    
    /* =========================================================
       5) SPACING HELPERS
    ========================================================= */
    .spacer-8{ height: 8px; }
    .spacer-16{ height: 16px; }
    .spacer-24{ height: 24px; }
    
    /* =========================================================
       6) RESULT SECTION (TITLE + SUBTITLE)
       NOTE: ini dipakai pada area hasil prediksi/clustering
    ========================================================= */
    .sec-title{
      font-size: 20px;
      font-weight: 900;
      margin: 0;
      color: #111827;
    }
    .sec-sub{
      margin: 6px 0 0 0;
      color: #9AA5B1;
      font-size: 13px;
    }
    
    /* =========================================================
       7) PANEL / GLASS CARD
    ========================================================= */
    .panel{
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.10);
    }
    
    /* Glass card (tema gelap, untuk info/intro) */
    .glass-card{
      background: rgba(30, 30, 40, 0.6);
      border-left: 4px solid #4CAF50;
      border-radius: 0 10px 10px 0;
      padding: 15px;
      margin-bottom: 10px;
      font-size: 0.95rem;
      color: #E0E0E0;
    }
    
    /* =========================================================
       8) RESULT HEADER (TOP)
    ========================================================= */
    .result-head{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 12px;
      margin-top: 8px;
    }
    .result-title{
      font-size: 34px;
      font-weight: 900;
      margin: 0;
    
      background: linear-gradient(90deg, #4CAF50, #00E5FF);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .pill{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
    
      font-size: 12px;
      font-weight: 800;
      color: #4CAF50;
      background: rgba(76,175,80,0.10);
      border: 1px solid rgba(76,175,80,0.28);
    }
    
    /* =========================================================
       9) INFO CARD (METRICS)
    ========================================================= */
    .info-card{
      border-radius: 14px;
      padding: 16px;
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(0,0,0,0.06);
      box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    }
    .info-card h3{
      margin: 0 0 10px 0;
      font-size: 16px;
      font-weight: 900;
      color: #111827;
    }
    .info-card p{
      margin: 0;
      color: #5F6C7B;
      font-size: 14px;
      line-height: 1.6;
    }
    .info-left{
      border-left: 8px solid #4CAF50;
    }
    
    /* =========================================================
       10) GCARD (CONTENT CARD + ITEM)
    ========================================================= */
    .gcard{
      background: rgba(255,255,255,.92);
      border: 1px solid rgba(76,175,80,.18);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 22px rgba(0,0,0,.08);
    }
    .ghead{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 10px;
    }
    .gtitle{
      margin: 0;
      font-weight: 900;
      color: #111827;
      font-size: 17px;
    }
    .gsub{
      margin: 6px 0 0;
      color: #6B7280;
      font-size: 13px;
    }
    .gpill{
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(76,175,80,.10);
      border: 1px solid rgba(76,175,80,.22);
      color: #2E7D32;
      font-weight: 900;
      font-size: 12px;
      white-space: nowrap;
    }
    .gitem{
      background: rgba(76,175,80,.06);
      border: 1px solid rgba(76,175,80,.14);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      gap: 10px;
      align-items: flex-start;
      margin-bottom: 10px;
    }
    .gicon{
      width: 34px;
      height: 34px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
    
      background: rgba(76,175,80,.16);
      border: 1px solid rgba(76,175,80,.22);
      font-size: 18px;
    }
    .git{
      margin: 0;
      font-weight: 900;
      color: #111827;
      font-size: 14px;
    }
    .gid{
      margin: 4px 0 0;
      color: #5F6C7B;
      font-size: 13px;
      line-height: 1.5;
    }
    .gbadge{
      background: #4CAF50;
      color: #fff;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 900;
    }
    
    /* =========================================================
       11) TABS STYLING
    ========================================================= */
    .stTabs [data-baseweb="tab-list"]{
      gap: 8px;
    }
    .stTabs [data-baseweb="tab"]{
      border-radius: 10px 10px 0 0;
      padding: 12px 24px;
      font-weight: 600;
    }
    </style>

    """, unsafe_allow_html=True)

    # ==========================================
    # HEADER SECTION
    # ==========================================
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<h1 class="hero-title">📝 Prediksi & Analisis Smartphone</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle"> Pilih fitur yang ingin Anda gunakan</p>', unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="text-align:right; padding-top:10px;">
        </div>
        """, unsafe_allow_html=True)

    # Panel info
    st.markdown("""
    <div class="glass-card">
        <b>Pilih Fitur:</b> 
        <span class="highlight-text">Prediksi Nomophobia</span> untuk menilai ketergantungan psikologis, atau 
        <span class="highlight-text">Analisis Pola Penggunaan</span> untuk melihat intensitas penggunaan harian.
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # INISIALISASI SESSION STATE
    # ==========================================
    if "form_answers" not in st.session_state:
        st.session_state.form_answers = {}
    
    if "intensitas_answers" not in st.session_state:
        st.session_state.intensitas_answers = {}

    # ==========================================
    # TAB UNIFIED: ANALISIS DAN PREDIKSI
    # ==========================================
    st.markdown('<div class="section-header">Analisis Pola Penggunaan & Prediksi Nomophobia</div>', unsafe_allow_html=True)
    
    # =========================
    # Mapping untuk Intensitas
    # =========================
    durasi_map = {
        "< 3 jam": 1,
        "3 – 6 jam": 2,
        "7 – 10 jam": 3,
        "11 – 14 ja": 4,
        "> 14 jam": 5
    }
    
    intes_penggunaan_map = {
        "1 – 3 kali": 1,
        "4 – 6 kali": 2,
        "7 – 9 kali": 3,
        "10 – 12 kali": 4,
        "> 12 kali": 5
    }
    
    frekuensi_map = {
        "1 – 3 kali": 1,
        "4 – 6 kali": 2,
        "7 – 9 kali": 3,
        "10 – 12 kali": 4,
        "> 12 kali": 5
    }
    
    waktu_intens_map = {
        "Pagi (06.00 – 11.59)": 1,
        "Siang (12.00 – 17.59)": 2,
        "Malam (18.00 – 23.59)": 3,
        "Larut malam (00.00 – 05.59)": 4,
        "Sepanjang hari (pagi – malam)": 5
    }
    
    numerical_cols = [
        "durasi_penggunaan_harian",
        "frekuensi_buka_smartphone",
        "frekuensi_cek_notifikasi",
        "frekuensi_gunakan_saat_berbicara",
        "waktu_paling_intens"
    ]
    
    onehot_cols = [
        "Bekerja/Belajar",
        "Belanja online",
        "Bermain Media sosial",
        "Bermain game",
        "Chattingan",
        "Mendengar Hiburan lain",
        "Streaming video"
    ]
    
    # Likert options untuk form nomophobia
    likert_options = list(skala_jawaban.keys())
    likert_map = skala_jawaban
    
    # =========================
    # FORM UNIFIED (INTENSITAS + NOMOPHOBIA)
    # =========================
    with st.form("form_unified"):
        
        # =========================
        # BAGIAN 1: FORM INTENSITAS
        # =========================
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Bagian 1: Pola Penggunaan Smartphone</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subheader">Jawab pertanyaan tentang kebiasaan penggunaan smartphone Anda</div>', unsafe_allow_html=True)
        
        col_int1, col_int2 = st.columns(2)
        
        with col_int1:
            durasi_txt = st.selectbox(
                "Durasi penggunaan smartphone per hari",
                list(durasi_map.keys()),
                index=0,
                key="i_durasi_txt"
            )
            frek_buka_txt = st.selectbox(
                "Seberapa sering Anda membuka smartphone (per hari)",
                list(intes_penggunaan_map.keys()),
                index=0,
                key="i_frek_buka_txt"
            )
            waktu_intens_txt = st.selectbox(
                "Waktu paling intens menggunakan smartphone",
                list(waktu_intens_map.keys()),
                index=0,
                key="i_waktu_intens_txt"
            )
        
        with col_int2:
            cek_notif_txt = st.selectbox(
                "Seberapa sering Anda mengecek notifikasi (per Jam)",
                list(frekuensi_map.keys()),
                index=0,
                key="i_cek_notif_txt"
            )
            pakai_saat_ngobrol_txt = st.selectbox(
                "Seberapa sering memakai HP saat berbicara/berinteraksi",
                list(frekuensi_map.keys()),
                index=0,
                key="i_pakai_ngobrol_txt"
            )
            st.markdown("Aktivitas utama yang paling sering dilakukan di smartphone")
            st.caption("Boleh memilih lebih dari satu")
            
            aktivitas_pilihan = []
            
            cols = st.columns(2)  # tampil 2 kolom agar rapi (ubah ke 3 kalau mau)
            
            for i, aktivitas in enumerate(onehot_cols):
                with cols[i % 2]:
                    if st.checkbox(aktivitas, key=f"chk_{aktivitas}"):
                        aktivitas_pilihan.append(aktivitas)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # =========================
        # BAGIAN 2: FORM NOMOPHOBIA (LIKERT)
        # =========================
        st.markdown('<div class="section-header">Bagian 2: Ketergantungan Psikologis pada Smartphone</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subheader">Jawab setiap pernyataan dengan jujur sesuai kondisi Anda</div>', unsafe_allow_html=True)
        
        n = len(likert_cols)
        q1 = likert_cols[:n//3]
        q2 = likert_cols[n//3:2*n//3]
        q3 = likert_cols[2*n//3:]
        
        # Helper function untuk render pertanyaan likert
        def render_group(title, subtitle, questions):
            st.markdown('<div class="section-divider" style="margin-top:20px;"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-header" style="font-size:16px;">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-subheader" style="font-size:14px;">{subtitle}</div>', unsafe_allow_html=True)
            
            cols = st.columns(2, gap="large")
            idx = 0
            
            for kolom in questions:
                label = label_pernyataan.get(kolom, kolom)
                
                default_value = "Netral" if "Netral" in likert_options else likert_options[len(likert_options)//2]
                if kolom in st.session_state.form_answers:
                    saved_num = st.session_state.form_answers[kolom]
                    for k, v in likert_map.items():
                        if v == saved_num:
                            default_value = k
                            break
                
                with cols[idx % 2]:
                    st.write(f"**{label}**")
                    val = st.select_slider(
                        "Label hidden",
                        options=likert_options,
                        value=default_value,
                        key=f"nomophobia_{kolom}",
                        label_visibility="collapsed"
                    )
                    st.session_state.form_answers[kolom] = likert_map[val]
                    st.write("")
                idx += 1
            
            st.write("")
        
        # === BAGIAN LIKERT 1 ===
        render_group(
            "Ketergantungan Fisik",
            "Ketergantungan terhadap keberadaan smartphone",
            q1
        )
        
        # === BAGIAN LIKERT 2 ===
        render_group(
            "Kecemasan & Emosi",
            "Reaksi emosional dan kecemasan",
            q2
        )
        
        # === BAGIAN LIKERT 3 ===
        render_group(
            "Kebutuhan Konektivitas",
            "Kebutuhan akan koneksi dan informasi",
            q3
        )
        
        # TOMBOL SUBMIT UNIFIED
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_all = st.form_submit_button("🚀 Analisis & Prediksi Lengkap", use_container_width=True)
    
    # =========================
    # PROSES SEMUA PREDIKSI SETELAH SUBMIT
    # =========================
    max_score_intensitas = 32  

    if submit_all:
        # ================================
        # 1) HITUNG SKOR INTENSITAS
        # ================================
        intensitas_encoded = {
            "durasi_penggunaan_harian": durasi_map.get(durasi_txt, 0),
            "frekuensi_buka_smartphone": intes_penggunaan_map.get(frek_buka_txt, 0),
            "frekuensi_cek_notifikasi": frekuensi_map.get(cek_notif_txt, 0),
            "frekuensi_gunakan_saat_berbicara": frekuensi_map.get(pakai_saat_ngobrol_txt, 0),
            "waktu_paling_intens": waktu_intens_map.get(waktu_intens_txt, 0),
        }
    
        for col in onehot_cols:
            intensitas_encoded[col] = 1 if col in aktivitas_pilihan else 0
    
        X_int = pd.DataFrame([intensitas_encoded])
        needed_cols = list(numerical_cols) + list(onehot_cols)
        X_int = X_int.reindex(columns=needed_cols, fill_value=0)
        X_int = X_int.apply(pd.to_numeric, errors="coerce").fillna(0)
    
        if max_score_intensitas <= 0:
            st.error("max_score_intensitas harus > 0")
            st.stop()
    
        skor_intensitas = (X_int.sum(axis=1).iloc[0] / max_score_intensitas) * 100.0
        skor_intensitas = float(np.clip(skor_intensitas, 0, 100))
    
        st.session_state.skor_intensitas = skor_intensitas
        st.session_state.intensitas_encoded = intensitas_encoded
    
        # ================================
        # 2) HITUNG SKOR NOMOPHOBIA (LIKERT)
        # ================================
        jawaban_angka = st.session_state.form_answers
        if len(jawaban_angka) < len(likert_cols):
            st.error("⚠️ Mohon lengkapi semua pertanyaan skala likert.")
            st.stop()
    
        total_skor_nom = sum(float(jawaban_angka[c]) for c in likert_cols)
        max_skor_nom = len(likert_cols) * 5
        skor_nomophobia = (total_skor_nom / max_skor_nom) * 100.0
        skor_nomophobia = float(np.clip(skor_nomophobia, 0, 100))
        st.session_state.skor_nomophobia = skor_nomophobia
    
        # ================================
        # 3) CLUSTERING (MODEL IPYNB)
        # ================================
        X_user = np.array([[skor_intensitas, skor_nomophobia]], dtype=float)
        X_user_scaled = scaler.transform(X_user)
        cluster_id = int(kmeans.predict(X_user_scaled)[0])
    
        cluster_label_map = {
            0: "Pengguna Seimbang",
            1: "Pengguna Intens dan Bergantung Tinggi",
            2: "Pengguna Rendah namun Cemas Tinggi"
        }
        cluster_kategori = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")
    
        st.session_state.cluster_id = cluster_id
        st.session_state.cluster_kategori = cluster_kategori
    
        # ==========================================
        # 5. TAMPILKAN HASIL (FOKUS CLUSTERING)
        # ==========================================
        # Ambil hasil dari session_state (BIAR AMAN)
        cluster_id = st.session_state.get("cluster_id", None)
        cluster_kategori = st.session_state.get("cluster_kategori", "Belum ada hasil")
        skor_nomophobia = st.session_state.get("skor_nomophobia", None)
        skor_intensitas = st.session_state.get("skor_intensitas", None)
        jawaban_angka = st.session_state.get("form_answers", {})
        
        # Guard: kalau belum submit / belum ada hasil
        if cluster_id is None or skor_nomophobia is None or skor_intensitas is None:
            st.info("Silakan isi form dan klik Submit untuk melihat hasil clustering.")
            st.stop()
        
        skor_nomophobia = float(skor_nomophobia)
        skor_intensitas = float(skor_intensitas)
        
        # ===== Helper: Kategori Nomophobia (label sesuai rekomendasi_db) =====
        def get_nomophobia_category(score: float):
            """
            Return:
              nomo_cat: "Nomophobia Rendah/Sedang/Tinggi" (sesuai mapping rekomendasi_db)
              nomo_icon: emoji
              nomo_color: warna
              nomo_desc: deskripsi singkat
            """
            if score < 60:
                return "Nomophobia Rendah", "🟢", "#4CAF50", "Tidak ada masalah nomophobia yang signifikan."
            elif score < 80:
                return "Nomophobia Sedang", "🟡", "#FFA726", "Ada beberapa gejala nomophobia yang perlu diwaspadai."
            else:
                return "Nomophobia Tinggi", "🔴", "#FF4B4B", "Tingkat ketergantungan tinggi, perlu penanganan serius."
        
        nomo_cat, nomo_icon, nomo_color, nomo_desc = get_nomophobia_category(skor_nomophobia)
        
        # ===== Style Cluster =====
        if cluster_id == 0:
            ccol, cicon = "#4CAF50", "✨"
        elif cluster_id == 1:
            ccol, cicon = "#FF4B4B", "🔥"
        elif cluster_id == 2:
            ccol, cicon = "#FFA15A", "⚡"
        else:
            ccol, cicon = "#9CA3AF", "❓"
        
        # ===== Header =====
        st.markdown("""
        <div class="result-head">
          <div>
            <div class="result-title">🧩 Hasil Clustering Pola Penggunaan</div>
            <p class="sec-sub">Hasil ini berasal dari model KMeans (scaler + kmeans) yang sudah dilatih di ipynb.</p>
          </div>
        </div>
        <div class='spacer-16'></div>
        """, unsafe_allow_html=True)
        
        # ===== Ringkasan 3 metrik =====
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-card" style="text-align:center; border-left:8px solid #4CAF50;">
                <h3 style="color:#111827; margin:0; font-size:24px;">{skor_intensitas:.1f}</h3>
                <p style="margin:0; font-size:12px; color:#666;">Skor Intensitas</p>
                <p style="margin:4px 0 0 0; font-size:11px; color:#888;">(0–100)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card" style="text-align:center; border-left:8px solid {nomo_color};">
                <h3 style="color:#111827; margin:0; font-size:24px;">{skor_nomophobia:.1f}</h3>
                <p style="margin:0; font-size:12px; color:#666;">Skor Nomophobia</p>
                <p style="margin:4px 0 0 0; font-size:11px; color:#888;">Skala 0–100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-card" style="text-align:center; border-left:8px solid {ccol};">
                <h3 style="color:#111827; margin:0; font-size:24px;">{cicon} Cluster {cluster_id}</h3>
                <p style="margin:0; font-size:12px; color:#666;">Pola Penggunaan</p>
                <p style="margin:4px 0 0 0; font-size:11px; color:#888;">{cluster_kategori}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='spacer-24'></div>", unsafe_allow_html=True)
        
        # ===== Penjelasan cluster (HTML) =====
        cluster_explanations = {
            0: f"""
            <p style="margin:0; color:#374151; line-height:1.6; font-size:14px;">
              Kombinasi skor intensitas <b style="color:{ccol};">{skor_intensitas:.1f}</b> dan skor nomophobia
              <b style="color:{nomo_color};">{skor_nomophobia:.1f}</b> menunjukkan pola penggunaan yang relatif seimbang.
              Penggunaan smartphone Anda cenderung proporsional dan tidak terlalu mengganggu aktivitas lain.
            </p>
            """,
            1: f"""
            <p style="margin:0; color:#374151; line-height:1.6; font-size:14px;">
              Skor intensitas <b style="color:{ccol};">{skor_intensitas:.1f}</b> cenderung tinggi dan disertai skor nomophobia
              <b style="color:{nomo_color};">{skor_nomophobia:.1f}</b>. Ini mengarah pada penggunaan intens dan ketergantungan tinggi.
              Disarankan mulai menerapkan pembatasan durasi dan mengurangi pemicu kebiasaan.
            </p>
            """,
            2: f"""
            <p style="margin:0; color:#374151; line-height:1.6; font-size:14px;">
              Intensitas <b style="color:{ccol};">{skor_intensitas:.1f}</b> tidak terlalu tinggi, namun skor nomophobia
              <b style="color:{nomo_color};">{skor_nomophobia:.1f}</b> menunjukkan kecemasan/ketergantungan psikologis.
              Fokuskan pada pengelolaan kecemasan saat jauh dari smartphone.
            </p>
            """
        }
        cluster_html = cluster_explanations.get(cluster_id, "")
        
        # ===== Card Kategori Nomophobia + Profil Cluster =====
        col_summary1, col_summary2 = st.columns([1, 1], gap="large")
        
        with col_summary1:
            st.markdown(f"""
            <div class="info-card" style="border-left:8px solid {nomo_color};">
              <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                <h3 style="margin:0;color:#111827;font-weight:900;">{nomo_icon} Kategori Nomophobia</h3>
                <div style="padding:6px 16px;background:{nomo_color};color:white;border-radius:20px;font-weight:700;font-size:0.9rem;">
                  {nomo_cat}
                </div>
              </div>
              <p style="margin:0;color:#374151;line-height:1.6;font-size:14px;">
                Skor Nomophobia Anda <b>{skor_nomophobia:.1f}/100</b>. {nomo_desc}
              </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_summary2:
            st.markdown(f"""
            <div class="info-card" style="border-left:8px solid {ccol};">
              <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                <h3 style="margin:0;color:#111827;font-weight:900;">{cicon} Profil Pola Penggunaan</h3>
                <div style="padding:6px 16px;background:{ccol};color:white;border-radius:20px;font-weight:700;font-size:0.9rem;">
                  Cluster {cluster_id}
                </div>
              </div>
              <p style="margin:0;color:#374151;line-height:1.6;font-size:14px;">
                <b>{cluster_kategori}</b>
              </p>
              <div style="margin-top:10px;">
                {cluster_html}
              </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='spacer-24'></div>", unsafe_allow_html=True)
        
        # ===== REKOMENDASI (pakai rekomendasi_db + kategori_rekom_map) =====
        st.markdown('<div class="section-header">Rekomendasi</div>', unsafe_allow_html=True)
        
        key_rekom = kategori_rekom_map.get(nomo_cat, "Nomophobia Sedang")
        rekom_list = rekomendasi_db.get(key_rekom, [])
        
        if rekom_list:
            cols = st.columns(len(rekom_list))
            for idx, rec in enumerate(rekom_list):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="info-card" style="height:220px; display:flex; flex-direction:column; text-align:center; border-top:4px solid {nomo_color};">
                      <div style="font-size:32px; margin-bottom:12px;">{rec.get('icon','💡')}</div>
                      <h4 style="color:#111827; margin:0 0 12px 0; font-size:16px;">{rec.get('title','Rekomendasi')}</h4>
                      <p style="margin:0; color:#374151; font-size:14px; line-height:1.5; flex-grow:1;">{rec.get('desc','')}</p>
                      <div style="margin-top:16px; padding:6px 12px; background:{nomo_color}15; color:{nomo_color}; border-radius:20px; font-size:12px; font-weight:600;">
                        Prioritas {idx + 1}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Belum ada rekomendasi untuk kategori ini.")
        
        st.markdown("<div class='spacer-24'></div>", unsafe_allow_html=True)
        
        # ===== Fokus Perbaikan (pakai saran_spesifik_db berdasarkan jawaban 4–5) =====
        st.markdown('<div class="section-header">Fokus Perbaikan Prioritas</div>', unsafe_allow_html=True)
        
        fokus_perbaikan = []
        for kolom in likert_cols:
            nilai = float(jawaban_angka.get(kolom, 0))
            if nilai >= 4:
                pertanyaan = label_pernyataan.get(kolom, kolom)  # ✅ tampilkan pertanyaan, fallback nama kolom
                fokus_perbaikan.append({
                    "kolom": kolom,
                    "pertanyaan": pertanyaan,
                    "nilai": nilai,
                    "saran": saran_spesifik_db.get(kolom, "Belum ada saran spesifik untuk aspek ini.")
                })
        
        fokus_perbaikan.sort(key=lambda x: x["nilai"], reverse=True)
        fokus_perbaikan = fokus_perbaikan[:6]
        
        if fokus_perbaikan:
            for item in fokus_perbaikan:
                st.markdown(f"""
                <div class="info-card" style="border-left:8px solid {nomo_color}; margin-bottom:12px;">
                  <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                    <div style="font-weight:900; color:#111827; line-height:1.4;">
                      {item['pertanyaan']}
                    </div>
                    <div style="white-space:nowrap; padding:4px 10px; border-radius:16px; background:{nomo_color}; color:white; font-weight:800; font-size:12px;">
                      {item['nilai']}/5
                    </div>
                  </div>
                  <div style="margin-top:10px; color:#374151; line-height:1.6;">
                    {item['saran']}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Tidak ada aspek prioritas (tidak ada jawaban 4–5).")
        
        st.markdown("<div class='spacer-24'></div>", unsafe_allow_html=True)

# --- 6. HALAMAN: Prediksi  ---
if selected == "Prediksi Score Nomophobia":
    import streamlit as st
    import joblib
    import numpy as np

    st.title("🧠 Prediksi Intensitas Nomophobia")
    st.markdown("Silakan pilih jawaban yang paling sesuai dengan kondisi Anda.")

    @st.cache_resource
    def load_resources():
        # Load model dan scaler secara bersamaan
        model = joblib.load("xgb_nomophobia_model.joblib")
        scaler = joblib.load("scaler.pkl")
        return model, scaler

    model, scaler = load_resources()

    map_durasi = { "< 3 jam": 1, "3 – 6 jam": 2, "7 – 10 jam": 3, "11 – 14 jam": 4, "> 14 jam": 5 }
    map_frekuensi_count = { "1 – 3 kali": 1, "4 – 6 kali": 2, "7 – 9 kali": 3, "10 – 12 kali": 4, "> 12 kali": 5 }
    map_perilaku = { "Tidak pernah": 1, "Jarang": 2, "Cukup sering": 3, "Sering": 4, "Sangat sering": 5 }
    map_waktu_intes_penggunaan = {
        "Pagi (06.00 – 11.59)": 1, 
        "Siang (12.00 – 17.59)": 2, 
        "Malam (18.00 – 23.59)": 3, 
        "Larut malam (00.00 – 05.59)": 4, 
        "Sepanjang hari (pagi – malam)": 5
    }

    with st.form("kuesioner_nomophobia"):
        st.subheader("Bagian 1: Perilaku Penggunaan")

        durasi_label = st.selectbox("Berapa lama durasi penggunaan smartphone Anda dalam sehari?", list(map_durasi.keys()))
        buka_hp_label = st.selectbox("Berapa kali Anda membuka smartphone dalam sehari?", list(map_frekuensi_count.keys()))
        cek_notif_label = st.selectbox("Berapa kali Anda mengecek notifikasi dalam satu jam?", list(map_frekuensi_count.keys()))
        bicara_hp_label = st.selectbox("Seberapa sering Anda menggunakan smartphone saat sedang berbicara dengan orang lain?", list(map_perilaku.keys()))
        waktu_label = st.selectbox("Waktu penggunaan paling intens?", list(map_waktu_intes_penggunaan.keys()))

        st.subheader("Bagian 2: Kategori Aktivitas")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            act_belajar = st.checkbox("Bekerja/Belajar")
            act_belanja = st.checkbox("Belanja online")
            act_medsos = st.checkbox("Bermain Media sosial")
            act_game = st.checkbox("Bermain game")
        with col_act2:
            act_chat = st.checkbox("Chattingan")
            act_hiburan = st.checkbox("Mendengar Hiburan lain")
            act_streaming = st.checkbox("Streaming video")

        submit_btn = st.form_submit_button("🚀 Analisis & Prediksi")
    if submit_btn:
        # 1. Ambil data numerik (5 kolom pertama)
        num_input = np.array([[
            float(map_durasi[durasi_label]),
            float(map_frekuensi_count[buka_hp_label]),
            float(map_frekuensi_count[cek_notif_label]),
            float(map_perilaku[bicara_hp_label]),
            float(map_waktu_intes_penggunaan[waktu_label])
        ]])

        # 2. Ambil data kategori (7 kolom sisanya)
        cat_input = np.array([[
            1 if act_belajar else 0,
            1 if act_belanja else 0,
            1 if act_medsos else 0,
            1 if act_game else 0,
            1 if act_chat else 0,
            1 if act_hiburan else 0,
            1 if act_streaming else 0
        ]])

        try:
            # 3. TRANSFORMASI: Fitur numerik HARUS di-scale
            num_scaled = scaler.transform(num_input)

            # 4. GABUNGKAN: [Numerik Scaled + Kategori]
            input_final = np.hstack([num_scaled, cat_input])

            # 5. PREDIKSI
            prediksi_skor = model.predict(input_final)[0]
            
            # Logika Klasifikasi Berdasarkan Skor (Contoh)
            if prediksi_skor <= 10:
                cluster_name, cluster_color, cluster_desc = "Rendah", "green", "Tingkat ketergantungan Anda terhadap smartphone masih dalam batas wajar."
            elif prediksi_skor <= 18:
                cluster_name, cluster_color, cluster_desc = "Sedang", "orange", "Anda mulai menunjukkan gejala kecemasan tanpa smartphone. Cobalah digital detox."
            else:
                cluster_name, cluster_color, cluster_desc = "Tinggi", "red", "Waspada! Skor Anda menunjukkan indikasi kuat Nomophobia."

            st.divider()
            st.subheader("📊 Hasil Analisis Intensitas")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Skor Prediksi", f"{prediksi_skor:.2f}")
                st.markdown(f"Status: **:{cluster_color}[{cluster_name}]**")
            with res_col2:
                st.info(f"**Analisis:** {cluster_desc}")

            # Visualisasi Progress
            progress = min(max(prediksi_skor / 25.0, 0.0), 1.0)
            st.write("Posisi Skor dalam Rentang Intensitas:")
            st.progress(progress)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat proses prediksi: {e}")