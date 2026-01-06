import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Tutorial",
    page_icon="💡",
    layout="wide"
)

# --- Title ---
st.title("💡 Tutorial Penggunaan Model")

st.markdown("""
Halaman ini berisi panduan langkah demi langkah untuk menggunakan fitur **Analisis Sentimen** dan 
**Pemodelan Topik** pada website ini.  
Pengguna dapat menganalisis ulasan aplikasi transportasi publik di Jakarta secara interaktif 
menggunakan model *Natural Language Processing* yang telah di-*deploy*.
""")

st.markdown("---")


# STEP 1
st.header("1️⃣ Memilih Metode Input Data")

st.markdown("""
Pada halaman **Penggunaan Model**, pengguna dapat memilih salah satu dari tiga metode input data berikut:
""")

st.subheader("✏️ A. Ketik Teks")
st.markdown("""
- Pilih opsi **Ketik Teks**
- Masukkan ulasan pengguna secara manual
- **Satu baris merepresentasikan satu ulasan**
- Cocok untuk analisis cepat atau pengujian individual
""")

st.code(
"""Aplikasi Transjakarta sangat membantu perjalanan saya.
Aplikasi sering error saat jam sibuk.""",
language="text"
)

st.subheader("📂 B. Unggah File CSV")
st.markdown("""
- Pilih opsi **Unggah CSV**
- Unggah file dengan format **.csv**
- File **wajib memiliki kolom bernama `Text`**
- Setiap baris akan diproses sebagai satu ulasan
""")

st.subheader("🧪 C. Teks Contoh")
st.markdown("""
- Pilih opsi **Teks Contoh**
- Sistem akan menggunakan ulasan contoh bawaan
- Cocok untuk demonstrasi, presentasi, atau validasi sistem
""")

st.markdown("---")

# STEP 2
st.header("2️⃣ Menjalankan Analisis")

st.markdown("""
Setelah data berhasil dimasukkan:
1. Klik tombol **🚀 Jalankan Analisis**
2. Sistem akan secara otomatis:
   - Memprediksi **sentimen** setiap ulasan
   - Menentukan **topik** berdasarkan hasil sentimen
""")

st.info("⏳ Waktu pemrosesan bergantung pada jumlah ulasan yang dianalisis.")

st.markdown("---")

# STEP 3
st.header("3️⃣ Membaca Hasil Analisis Sentimen")

st.markdown("""
Hasil analisis sentimen ditampilkan dalam bentuk tabel yang berisi:
- **Text** → Ulasan asli pengguna  
- **Sentiment** → Kategori sentimen (*Positif* atau *Negatif*)  
- **Confidence** → Tingkat keyakinan model terhadap prediksi  
""")

st.markdown("""
Nilai *confidence* yang mendekati **1.0** menunjukkan bahwa model sangat yakin terhadap hasil prediksi.
""")

st.markdown("---")

# STEP 4
st.header("4️⃣ Membaca Hasil Pemodelan Topik")

st.markdown("""
Setelah sentimen ditentukan, ulasan akan diproses menggunakan model pemodelan topik yang sesuai.
""")

st.subheader("🟢 Topik Sentimen Positif")
st.markdown("""
- Berisi ulasan dengan sentimen **Positif**
- Setiap ulasan ditetapkan ke satu topik paling relevan
- Disertai nilai *confidence* sebagai ukuran kedekatan dengan topik
""")

st.subheader("🔴 Topik Sentimen Negatif")
st.markdown("""
- Berisi ulasan dengan sentimen **Negatif**
- Topik merepresentasikan keluhan, kendala, atau masalah pengguna
""")

st.markdown("---")

# STEP 5
st.header("5️⃣ Mengunduh Hasil Analisis")

st.markdown("""
Pengguna dapat mengunduh hasil analisis dalam format **CSV** untuk keperluan lanjutan, seperti:
- Analisis tambahan
- Dokumentasi laporan
- Visualisasi eksternal
""")

st.markdown("""
File yang dapat diunduh:
- `hasil_sentimen.csv`
- `hasil_topik_positif.csv`
- `hasil_topik_negatif.csv`
""")

st.markdown("---")

# --- NOTES ---
st.header("📌 Catatan Penting")

st.markdown("""
- Model **tidak membuat topik baru**, tetapi menetapkan ulasan ke topik yang telah dipelajari sebelumnya
- Ulasan yang tidak relevan dengan topik mana pun dapat diberi label **Topic = -1**
- Disarankan menggunakan teks berbahasa Indonesia untuk hasil yang optimal
""")

st.success("🎉 Anda telah menyelesaikan tutorial penggunaan model!")
