import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import io
import re
import os

# --- FUNCTIONS INITIALIZATION ---
STAT_FILE_PATH = 'Dataset/data.xlsx'
DATA_FILE_PATH = 'Dataset/no_emoji.csv'
STOPWORDS_PATH = 'Dataset/tala-stopwords-indonesia.txt'
SENTIMENT = 'sentiment'
TEXT = 'cleaned_content'

# --- Remove Stopwords ---
def load_stopwords(filepath):
    stopwords = set()

    # From tala
    if not os.path.exists(filepath):
        print(f"❌ Error: Stopwords file not found at {filepath}.")
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            tala_stopwords = [line.strip() for line in f if line.strip()]
            stopwords.update(tala_stopwords)
    
    # From Sastrawi
    try:
        factory = StopWordRemoverFactory()
        sastrawi_stopwords = factory.get_stop_words()
        stopwords.update(sastrawi_stopwords)
    except Exception as e:
        print(f"❌ Error loading Sastrawi stopwords: {e}")

    # Additional stopwords
    manual_stopwords = ['nya', 'ya', 'ap', 'ok', 'sih', 'deh', 'tau', 'gue', 'kak', 'eh', 'gua', 'tuh', 'lu', 'the', 'by', 'hadeh', 'ku', 'jis', 'an', 'dah', 'mah', 'loh', 'iya', 'you', 'ayo', 'wow', 'jos', 'sip', 'aduh', 'anjir', 'and', 'apatu', 'ah', 'si', 'duh', 'mbak', 'kah', 'amin', 'this', 'mu', 'baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya', 'banget', 'pas', 'kayak', 'oke']
    stopwords.update(manual_stopwords)

    return list(stopwords)

CUSTOM_STOPWORDS = load_stopwords(STOPWORDS_PATH)

def remove_stopwords(text, stopwords):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

# --- Load Data ---
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path)
        df[TEXT] = df[TEXT].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: Dataset not found at '{path}'")
        return pd.DataFrame() # empty DataFrame
    
def load_excel(path, sheet_name):
    try:
        df = pd.read_excel(
            path, 
            sheet_name=sheet_name, 
            engine='openpyxl'
        )
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: Excel file not found at '{path}'")
        return pd.DataFrame()
    except ValueError:
        st.error(f"❌ Error: Sheet '{sheet_name}' not found in the Excel file.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        return pd.DataFrame()
    
# --- Generate Bar Chart ---
def generate_bar_chart(df, x_col, y_col, is_sentiment=False):   
    if is_sentiment:
        df_plot = df[df[x_col].str.lower() != 'neutral'].copy()
    else:
        df_plot = df.copy()
    
    cmap = plt.get_cmap('berlin') 
    colors = cmap(np.linspace(0.2, 0.8, len(df_plot)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_plot[x_col], df_plot[y_col], color=colors)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    ax.tick_params(axis='x', rotation=0)

    for i, v in enumerate(df_plot[y_col]):
        ax.text(i, v + (max(df_plot[y_col]) * 0.01), str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# --- Generate Word Cloud ---
def generate_word_cloud(df, sentiment=None, colormap='viridis'):   
    if sentiment:
        df_filtered = df[df[SENTIMENT].str.lower() == sentiment.lower()]
        title = f"Sentimen {sentiment.capitalize()}"
    else:
        df_filtered = df
        title = "Dataset"

    text_corpus = " ".join(
        df_filtered[TEXT].apply(lambda x: remove_stopwords(x, CUSTOM_STOPWORDS)).tolist()
    )

    if not text_corpus.strip():
        st.write(f"*(No data found for {title})*")
        return

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap=colormap 
    ).generate(text_corpus) 

    st.markdown(f"**{title}**")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off") # Hide axes
    
    st.pyplot(fig)

# --- Generate Resample Dist Bar Chart ---
def generate_resam_chart(df):  
    cmap = plt.get_cmap('berlin')
        
    color_neg = cmap(0.8)  # Color for Negative
    color_pos = cmap(0.2)  # Color for Positive

    labels = df['Sampling Type']
    neg_counts = df['Amount of Negative Sentiment']
    pos_counts = df['Amount of Positive Sentiment']
    
    x = np.arange(len(labels))  
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 3))
    
    rects1 = ax.bar(x - width/2, pos_counts, width, label='Positif', color=color_pos)
    rects2 = ax.bar(x + width/2, neg_counts, width, label='Negatif', color=color_neg)

    max_val = max(max(pos_counts), max(neg_counts))
    ax.set_ylim(0, max_val * 1.15)

    ax.set_ylabel('Jumlah Sentimen')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', frameon=True)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# --- INTERFACE ---
st.set_page_config(
    page_title="Data dan Metodologi",
    page_icon="📊",
    layout='wide'
)
st.title("📊 Data dan Metodologi")

tab_data, tab_resampling, tab_sa, tab_tm = st.tabs([
    "📊 Data",
    "⚖️ *Resampling*",
    "😄 Analisis Sentimen",
    "📰 Pemodelan Topik"
])

df = load_csv(DATA_FILE_PATH)
pos_sam = load_excel(STAT_FILE_PATH, sheet_name='pos_sam')   
neg_sam = load_excel(STAT_FILE_PATH, sheet_name='neg_sam')   
app_dist = load_excel(STAT_FILE_PATH, sheet_name='app_dist')
sen_dist = load_excel(STAT_FILE_PATH, sheet_name='sen_dist')
train_dist = load_excel(STAT_FILE_PATH, sheet_name='train_dist')
pos_topics_df = pd.read_excel(STAT_FILE_PATH, sheet_name="pos_lab")
neg_topics_df = pd.read_excel(STAT_FILE_PATH, sheet_name="neg_lab")

if df.empty or pos_sam.empty or neg_sam.empty or app_dist.empty or sen_dist.empty or train_dist.empty:
    st.stop()

# --- Data Tab ---
with tab_data:
    st.header("Gambaran Umum Dataset")

    # Brief Explanation 
    st.markdown(
        """
    Data yang digunakan dalam penelitian ini berupa ulasan pengguna dari empat aplikasi transportasi publik di Jakarta.
    Data dikumpulkan dari *Google Play Store* menggunakan *library* Python `google_play_scraper`.
    """)

    apps = [
        {
            "name": "Access by KAI",
            "url": "https://play.google.com/store/apps/details?id=com.kai.kaiticketing",
            "logo": "https://play-lh.googleusercontent.com/oMCaFu3rpx2r1kWwwg2c3onakLO9_A6gE5ncYf7X7wwyfkM06SvVc-a2YeD0-5kRjR6nnIv0Uvfa7Dcj_Hd2=w480-h960-rw"
        },
        {
            "name": "Jak Lingko App",
            "url": "https://play.google.com/store/apps/details?id=com.jaklingkoindonesia.app",
            "logo": "https://play-lh.googleusercontent.com/exKSYuJVIPdRoagsTGqEih5iHsU1FH0TJRXTvj0F8AiykPvoUqhZll1it_ShgbfGmSg=w480-h960-rw"
        },
        {
            "name": "MyMRTJ",
            "url": "https://play.google.com/store/apps/details?id=com.mrt.jakarta",
            "logo": "https://play-lh.googleusercontent.com/2Xzw9b9jLnRMbcbUFOJ1ASnTryuFbu2UTCCm_uw2xKonLwQVBHmrVIOMGJHljvIuh0kK8BGceNDqjqBX6wyoIw=w480-h960-rw"
        },
        {
            "name": "TJ: Transjakarta",
            "url": "https://play.google.com/store/apps/details?id=com.transjakmobile",
            "logo": "https://play-lh.googleusercontent.com/V8T7kMSOym7BEJSl2SpWPMmTYO9FbCjeLNCF8lBbJ0Ixbfrj74xxW_v7LkCuB1q_0Q=w480-h960-rw"
        },
    ]

    col1, col2, col3, col4 = st.columns(4)
    for col, app in zip([col1, col2, col3, col4], apps):
        with col:
            st.image(app['logo'], width=50)
            st.markdown(f"**{app['name']}**")
            st.link_button("Go to App", app['url'])

    # Sample Data
    st.subheader("Sampel Data")
    col1, col2 = st.columns(2)
    try:
        with col1:
            st.markdown("**Sampel Sentimen Positif**")
            st.dataframe(pos_sam,use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Sampel Sentimen Negatif**")
            st.dataframe(neg_sam, use_container_width=True, hide_index=True)
    except KeyError:
        st.error(f"❌ Error: '{SENTIMENT}' column not found.")

    # Bar Chart
    st.subheader("Distribusi Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribusi Sentimen**")
        st.write("Grafik ini menunjukkan bahwa secara umum persepsi pengguna terhadap aplikasi transportasi publik Jakarta cenderung positif, di mana jumlah ulasan dengan sentimen positif (8.905) jauh lebih banyak dibandingkan ulasan dengan sentimen negatif (5.257) dan netral (733).")
        generate_bar_chart(sen_dist, 'Sentiment', 'Amount of Review', is_sentiment=True)
    with col2:
        st.markdown("**Distribusi Aplikasi**")
        st.write("Grafik ini menunjukkan bahwa Access by KAI mendominasi dataset dengan total 9.000 ulasan. Peringkat kedua ditempati oleh TJ: Transjakarta dengan 3.313 ulasan, diikuti oleh MyMRTJ sebanyak 1.405 ulasan dan Jak Lingko App sebanyak 1.177 ulasan.")
        generate_bar_chart(app_dist, 'Application', 'Amount of Review')

    # WC
    st.subheader("*Word Clouds*")
    st.write("Visualisasi kata-kata yang paling sering muncul pada seluruh dataset serta pada masing-masing kategori sentimen. Semalin besar ukuran sebuah kata pada grafik, maka semakin sering kemunculan kata tersebut.")
    col_full, col_pos, col_neg = st.columns(3)

    # try:
    #     with col_full:
    #         generate_word_cloud(df, colormap='berlin')
    #     with col_pos:
    #         generate_word_cloud(df, sentiment='positive', colormap='winter')
    #     with col_neg:
    #         generate_word_cloud(df, sentiment='negative', colormap='autumn')
    # except Exception as e:
    #     st.error(f"An error occurred while generating word clouds. Error: {e}")

with tab_resampling:
    st.header("Strategi *Resampling*")

    # Brief Explanation
    st.write("Eksplorasi awal dataset menunjukkan adanya ketidakseimbangan kelas yang cukup signifikan,di mana jumlah ulasan positif jauh lebih banyak dibandingkan ulasan negatif. Ketidakseimbangan ini berpotensi menyebabkan bias model terhadap kelas mayoritas.")

    # Technique Explanation
    st.subheader("Teknik *Resampling* yang Diterapkan")
    st.markdown("""
                Untuk mengatasi permasalahan tersebut, dua strategi *resampling* diterapkan **hanya pada data pelatihan**
                guna mencegah terjadinya *data leakage*, yaitu:

                * **Random Over-Sampling (ROS):**  
                Sampel dari kelas minoritas (sentimen negatif) diduplikasi hingga jumlah kedua kelas menjadi seimbang.

                * **Random Over-Sampling + Neighborhood Cleaning Rule (ROS-NCL):**  
                Pendekatan hibrida yang mengombinasikan *over-sampling* dan *under-sampling*.  
                Setelah ROS diterapkan, NCL digunakan untuk menghapus data yang bersifat *noise* atau tumpang tindih
                di sekitar batas antar kelas, sehingga menghasilkan dataset yang lebih bersih namun tetap relatif seimbang.
    """)

    # Result Chart
    st.subheader("**Distribusi Hasil *Resampling***")
    st.markdown("""
                Grafik perbandingan berikut menggambarkan perubahan distribusi data dari kondisi awal yang tidak seimbang hingga dua kondisi alternatif hasil *resampling*.

                * Metode ROS menghasilkan keseimbangan sempurna dengan masing-masing kelas berjumlah 6.233 data.

                * Metode ROS-NCL menghasilkan distribusi yang mendekati keseimbangan sempurna dengan 5.711 sentimen positif dan 4.853 sentimen negatif.

                Perbandingan ini menunjukkan bagaimana teknik *resampling* yang berbeda dapat digunakan untuk mengurangi ketidakseimbangan kelas dan meningkatkan akurasi pelatihan model.

    """)
    generate_resam_chart(train_dist)


with tab_sa:
    st.header("Metode Analisis Sentimen")

    # Brief Explanation
    st.write("Tujuan analisis sentimen dalam penelitian ini adalah untuk mengklasifikasikan ulasan pengguna aplikasi transportasi publik Jakarta ke dalam kategori sentimen **positif** atau **negatif**. Pendekatan ini memungkinkan pemahaman terhadap persepsi pengguna dan tingkat kepuasan layanan berdasarkan umpan balik tekstual dalam skala besar.")

    # Models
    st.subheader("Model yang Dievaluasi")
    st.markdown("""
                Penelitian ini mengevaluasi model berbasis *Transformer* serta model *machine learning* tradisional
                untuk tugas klasifikasi sentimen ulasan aplikasi transportasi publik di Jakarta.

                * **IndoBERT:**  
                Model berbasis *Transformer* yang dilatih pada korpus bahasa Indonesia berskala besar dengan bahasa formal dan terstandarisasi. IndoBERT mampu menangkap konteks dan struktur gramatikal dengan baik, sehingga sangat sesuai untuk analisis sentimen teks terstruktur. 4 varian yang dievaluasi dalam penelitian ini adalah `indobert-base-p1`, `indobert-lite-base-p1`, `indobert-base-p2`, dan `indobert-lite-base-p2`.

                * **NusaBERT:**  
                Model *Transformer* multibahasa yang dilatih pada korpus bahasa Indonesia serta 12 bahasa daerah lainnya. Pendekatan ini memungkinkan penanganan variasi linguistik, termasuk bahasa informal dan ekspresi lokal yang sering muncul dalam ulasan pengguna. Varian yang digunakan dalam penelitian ini adalah `NusaBERT-base`.

                * **Support Vector Machine (SVM):**  
                Algoritma *machine learning* tradisional yang mengklasifikasikan teks dengan mempelajari batas keputusan optimal pada ruang fitur berdimensi tinggi. Dalam penelitian ini, SVM digunakan sebagai model pembanding untuk mengevaluasi peningkatan kinerja yang dicapai oleh pendekatan berbasis *Transformer*.
    """)


    # Best Model (Deployed Model)
    st.subheader("Model yang Di-*Deploy*")
    st.write("Berdasarkan hasil eksperimen, model `indobert-base-p1` yang dilatih menggunakan dataset ROS-NCL menunjukkan performa terbaik. Model tersebut berhasil mencapai ***weighted F1-score*** sebesar **96,16%** pada data pelatihan yang menunjukkan kemampuan generalisasi yang baik pada data yang diteliti. Oleh karena itu, model inilah yang di-deploy pada *website* ini.")


with tab_tm:
    st.header("Metode Pemodelan Topik")

    # Brief Explanation
    st.write("Tujuan pemodelan topik dalam penelitian ini adalah untuk mengelompokkan ulasan pengguna ke dalam tema-tema yang bermakna, sehingga dapat mengidentifikasi isu utama serta pola diskusi yang muncul dalam ulasan aplikasi transportasi publik di Jakarta.")

    # Method
    st.subheader("BERTopic")
    st.write("**BERTopic** merupakan metode pemodelan topik yang mengombinasikan representasi embedding dari BERT dengan teknik klasterisasi untuk mengelompokkan teks ke dalam topik-topik yang koheren secara semantik serta merepresentasikannya melalui kata kunci yang paling relevan.")

    # Deployed Model
    st.subheader("Model yang Di-*Deploy*")
    st.markdown("""
                Dua model BERTopic yang berbeda di-*deploy*, masing-masing dilatih menggunakan ulasan dari kategori sentimen tertentu
                untuk menghasilkan topik yang lebih spesifik dan terfokus.

                * **Model Sentimen Negatif:**  
                Terdiri dari **9 topik berbeda** dengan **nilai koherensi topik** sebesar **71,63%**, yang menunjukkan
                tema diskusi yang jelas dan konsisten terkait keluhan serta permasalahan pengguna.

                * **Model Sentimen Positif:**  
                Terdiri dari **14 topik berbeda** dengan **nilai koherensi topik** sebesar **62,69%**, yang mencerminkan
                beragam pengalaman positif dan tingkat kepuasan pengguna terhadap layanan.
    """)


    # Topics
    pos_topics_df.index.name = "ID"
    neg_topics_df.index.name = "ID"

    st.subheader("Ringkasan Topik")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Topik Sentimen Positif**")
        st.dataframe(pos_topics_df[["Label"]], use_container_width=True)
    with col2:
        st.markdown("**🔴 Topik Sentimen Negatif**")
        st.dataframe(neg_topics_df[["Label"]], use_container_width=True)

    





