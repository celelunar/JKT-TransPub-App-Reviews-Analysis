import streamlit as st
from models import load_all_models

# --- Page Config ---
st.set_page_config(
    page_title="Welcome!",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Models ---
if "models" not in st.session_state:
    st.session_state.models = load_all_models()

# --- Brief Explanation ---
st.title("🚌 Analisis Ulasan Aplikasi Transportasi Publik")
st.markdown("""
*Website* ini menampilkan implementasi model *Natural Language Processing* (NLP) menggunakan **IndoBERT** dan **BERTopic** untuk menganalisis ulasan pengguna terhadap aplikasi transportasi publik di Jakarta yang tersedia pada *platform* Google Play Store. Aplikasi-aplikasi tersebut mencakup: **Access by KAI**, **Jak Lingko App**, **MyMRTJ**, dan **TJ: Transjakarta**.
""")

# --- Main Features ---
st.subheader("Fitur Utama")
st.markdown("""
* **🚦 Prediksi Sentimen:** Mengklasifikasikan ulasan ke dalam kategori sentimen (positif atau negatif) menggunakan model BERT yang telah dilatih.
            
* **🔍 Identifikasi Topik:** Menetapkan setiap ulasan ke salah satu topik yang telah diidentifikasi oleh model BERTopic yang telah dilatih.
""")

# --- Directory ---
st.subheader("Direktori *Website*")
st.info("""
**🏠 Halaman Utama:** Gambaran umum mengenai tujuan dan kemampuan website.

**📚 Data dan Metode:** Informasi rinci mengenai dataset serta teknik NLP yang digunakan dalam pengembangan model prediktif.

**⚙️ Penggunaan Model:** Antarmuka interaktif untuk memasukkan ulasan pengguna dan menjalankan model prediksi yang telah di-*deploy*.

**💡 Tutorial:** Panduan langkah demi langkah untuk membantu pengguna menggunakan halaman Penggunaan Model secara optimal.
""")

# Note for users
st.markdown("---")
st.warning("**👈 Gunakan *sidebar* untuk menavigasi antar halaman.**")