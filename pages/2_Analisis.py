import streamlit as st 
import pandas as pd 
import torch 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity 
from io import BytesIO
import re
import unicodedata
from unidecode import unidecode
from emoji import demojize
import matplotlib.pyplot as plt
from indoNLP.preprocessing import replace_slang, replace_word_elongation, emoji_to_words


# --- FUNCTIONS INITIALIZATION ---
SENTIMENT_REPO = "chimons-academy/indobert-jkt-transpub-app-review"
TOPIC_POS_REPO = "chimons-academy/bertopic-jkt-transpub-app-pos-review"
TOPIC_NEG_REPO = "chimons-academy/bertopic-jkt-transpub-app-neg-review"
STAT_FILE_PATH = 'Dataset/data.xlsx'

LABEL_MAP = {
    "LABEL_0": "Negatif",
    "LABEL_1": "Positif"
}

SAMPLE_TEXTS = [
    "Aplikasi Transjakarta sangat membantu untuk perjalanan sehari-hari karena informasi rute dan halte sangat jelas, serta fitur pelacak busnya memudahkan saya agar tidak menunggu terlalu lama.",
    "Pemesanan tiket sekarang sangat mudah dan cepat; saya hanya butuh beberapa klik dalam 1 menit langsung dapat tiket tanpa harus antre di loket stasiun.",
    "Sangat terkesan dengan petugas stasiun MRT yang sangat ramah dan sigap membantu, bahkan sampai menawarkan bantuan pengisian daya HP dan hotspot saat saya kesulitan 👍🏻",
    "Aplikasi Akses by KAI sekarang jauh lebih simpel dan praktis untuk beli tiket kereta api, pembayarannya pun bisa pakai KAI Pay yang sangat mempermudah bagi yang tidak punya m-banking.",
    "Fitur real-time tracking di aplikasi sangat akurat, saya bisa memantau posisi bus secara langsung sehingga rencana perjalanan jadi lebih efisien dan tidak menebak-nebak lagi.",
    "Sangat mengecewakan 😔 saya sudah bayar pakai KAI Pay dan saldo sudah terpotong, tapi status tiket tidak ter-update dan dana tidak otomatis kembali.",
    "Aplikasi sering sekali error dan loading lama padahal sinyal bagus; saat mau beli tiket lokal di jam sibuk, jadwal sering tidak muncul atau langsung habis dalam hitungan menit.",
    "Proses verifikasi menjadi member basic sangat berbelit-belit dan selalu muncul pesan 'ID sudah ada', padahal saya baru mencoba mendaftar.",
    "Sistem pembayarannya payah karena hanya tersedia e-wallet tertentu yang jarang dipakai orang, tolong kembalikan pilihan pembayaran populer seperti GoPay, OVO, atau Dana.",
    "Server sering sekali down atau mengalami internal server error, terutama di hari libur atau saat jam pemesanan tiket lokal dibuka, benar-benar menyusahkan 😡😤"
]

# --- Get Models ---
models = st.session_state.models

tokenizer = models["tokenizer"]
sa_mod = models["sa_mod"]
pos_mod = models["pos_mod"]
neg_mod = models["neg_mod"]


# --- Pre-process ---

def normalize_font(text):
    text = unidecode(str(text))
    text = unicodedata.normalize("NFKC", text)
    return text

def emoji_alias(text):
    text = emoji_to_words(text, delimiter = (" ", " "))
    return " ".join(word.replace("_", " ") for word in text.split())

def remove_repetitive_symbols(text):
    return re.sub(r'([^\w\s])\1+', r'\1', text)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text)

def cleaning(text):
    if not isinstance(text, str):
        return text

    text = text.lower()
    text = replace_slang(text)
    text = replace_word_elongation(text)
    text = emoji_alias(text)
    text = normalize_font(text)
    text = remove_repetitive_symbols(text)
    text = remove_extra_spaces(text)
    text = text.strip()

    return text

def preprocess_texts(texts):
    return [cleaning(t) for t in texts]

# --- Predict Sentiment ---
def predict_sentiment(texts):
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = sa_mod(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confs, preds = torch.max(probs, dim=1)

    sentiments = [
        LABEL_MAP[sa_mod.config.id2label[p.item()]] for p in preds
    ]

    return sentiments, confs.numpy()

# --- Predict Topic ---
def predict_topic(topic_model, text):
    text = f"passage: {text}"  # E5 best practice
    topic_id, _ = topic_model.transform([text])
    topic_id = topic_id[0]

    if topic_id == -1:
        return -1, 0.0

    emb = topic_model.embedding_model.embed([text])
    topic_emb = topic_model.topic_embeddings_[topic_id].reshape(1, -1)
    conf = cosine_similarity(emb, topic_emb)[0][0]

    return topic_id, float(conf)

def format_topic_label(topic_id, label_map):
    if topic_id in label_map:
        return f"{topic_id} ({label_map[topic_id]})"
    return f"{topic_id}"

# --- Download Button ---
def download_button(df, filename):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label=f"Download {filename}",
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv"
    )

def generate_sentiment_bar_chart(sent_counts):
    cmap = plt.get_cmap("berlin")

    color_pos = cmap(0.2)   # Hijau kebiruan
    color_neg = cmap(0.8)   # Merah keunguan

    labels = sent_counts.index.tolist()
    values = sent_counts.values.tolist()

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(6, 3))

    bars = ax.bar(
        x,
        values,
        width,
        color=[color_pos if l == "Positif" else color_neg for l in labels]
    )

    # --- Axis & Title ---
    ax.set_ylabel("Jumlah Ulasan")
    ax.set_title("Distribusi Sentimen Ulasan Pengguna", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    max_val = max(values)
    ax.set_ylim(0, max_val * 1.2)

    # --- Label Angka di Atas Bar ---
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)

# --- INTERFACE ---
# --- Page Config ---
# st.set_page_config(
#     page_title="Penggunaan Model",
#     page_icon="⚙️",
#     layout='wide'
# )

# --- session state ---
if "df_sent" not in st.session_state:
    st.session_state.df_sent = None

if "df_pos" not in st.session_state:
    st.session_state.df_pos = None

if "df_neg" not in st.session_state:
    st.session_state.df_neg = None

if "cleaned_texts" not in st.session_state:
    st.session_state.cleaned_texts = None

pos_lab = pd.read_excel(STAT_FILE_PATH, sheet_name="pos_lab")
neg_lab = pd.read_excel(STAT_FILE_PATH, sheet_name="neg_lab")

pos_label_map = dict(zip(pos_lab["Topic"], pos_lab["Label"]))
neg_label_map = dict(zip(neg_lab["Topic"], neg_lab["Label"]))

# --- Brief Explanation ---
st.title("⚙️ Penggunaan Model")

# --- User Input ---
st.subheader("📝 Input Data")

input_mode = st.radio(
    "Pilih metode input data:",
    ["Ketik Teks", "Unggah CSV", "Teks Contoh"],
    horizontal=True
)

texts = []

if input_mode == 'Ketik Teks':
    text_input = st.text_area(
        "Masukkan teks ulasan pengguna (1 ulasan per baris)",
        height=200
    )
    if text_input.strip():
        texts = [t.strip() for t in text_input.split("\n") if t.strip()]

elif input_mode == "Unggah CSV":
    up_file = st.file_uploader(
        "Unggah file CSV (harus memiliki kolom bernama 'Text')",
        type=["csv"]
    )
    if up_file:
        df_input = pd.read_csv(up_file)
        if "Text" not in df_input.columns:
            st.error("File CSV harus memiliki kolom bernama 'Text'")
        else:
            texts = df_input["Text"].dropna().astype(str).tolist()

elif input_mode == "Teks Contoh":
    st.info("Menggunakan teks contoh bawaan.")
    texts = SAMPLE_TEXTS

    with st.expander("**Teks contoh yang digunakan:**"):
        st.write(pd.DataFrame({"Text": texts})) 

# --- Run Analysis ---
with st.form("analysis_form", border=False):
    run_clicked = st.form_submit_button("🚀 Run")

if run_clicked and texts:
    cleaned_texts = preprocess_texts(texts)

    # --- Sentiment ---
    sentiments, confs = predict_sentiment(cleaned_texts)

    df_sent = pd.DataFrame({
        "Text": texts,
        "Sentiment": sentiments,
        "Confidence": confs
    })

    st.session_state.df_sent = df_sent
    st.session_state.cleaned_texts = cleaned_texts

        # --- Topic ---
    pos_rows = []
    neg_rows = []

    for text, cleaned_text, sent in zip(texts, cleaned_texts, sentiments):
        if sent == 'Positif':
            topic, conf = predict_topic(
                pos_mod, cleaned_text
            )
            pos_rows.append({
                "Text": text,
                "Topic": topic,
                "Confidence": conf
            })
        else:
            topic, conf = predict_topic(
                neg_mod, cleaned_text
            )
            neg_rows.append({
                "Text": text,
                "Topic": topic,
                "Confidence": conf
            })

    st.session_state.df_pos = pd.DataFrame(pos_rows) if pos_rows else None
    st.session_state.df_neg = pd.DataFrame(neg_rows) if neg_rows else None
    
elif run_clicked:
    st.warning("Silakan masukkan teks ulasan atau unggah file CSV terlebih dahulu.")

if st.session_state.df_sent is not None:

    with st.expander("🔍 Lihat Hasil Pre-processing"):
        st.dataframe(
            pd.DataFrame({
                "Original Text": st.session_state.df_sent["Text"],
                "Cleaned Text": st.session_state.cleaned_texts
            }),
            use_container_width=True
        )

    st.subheader("📊 Hasil Analisis Sentimen")

    df_sent = st.session_state.df_sent
    df_pos = st.session_state.df_pos
    df_neg = st.session_state.df_neg

    total_reviews = len(df_sent)
    sent_counts = df_sent["Sentiment"].value_counts()

    pos_count = sent_counts.get("Positif", 0)
    neg_count = sent_counts.get("Negatif", 0)
    avg_conf = df_sent["Confidence"].mean()

    if pos_count > neg_count:
        sentiment_summary = "didominasi oleh sentimen positif"
    elif neg_count > pos_count:
        sentiment_summary = "didominasi oleh sentimen negatif"
    else:
        sentiment_summary = "memiliki distribusi sentimen yang relatif seimbang"

    st.write(
        f"Dari total {total_reviews} ulasan yang dianalisis, "
        f"hasil menunjukkan bahwa distribusi sentimen {sentiment_summary}. "
        "Nilai confidence rata-rata yang relatif tinggi mengindikasikan bahwa model memiliki tingkat keandalan yang baik dalam mengklasifikasikan sentimen ulasan pengguna."
    )
  

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        sent_counts = st.session_state.df_sent["Sentiment"].value_counts()
        generate_sentiment_bar_chart(sent_counts)
    with col2:
        st.metric("Total Ulasan", total_reviews)       
        st.metric("Rata-rata *Confidence*", f"{avg_conf:.2f}")
    with col3:
        st.metric("Positif", pos_count, f"{pos_count/total_reviews:.1%}")
        st.metric("Negatif", neg_count, f"{neg_count/total_reviews:.1%}", delta_color="inverse")

    with st.expander("🔍 Lihat Detail Hasil"):
        st.dataframe(
            st.session_state.df_sent,
            use_container_width=True
        )

    download_button(
        st.session_state.df_sent,
        "hasil_sentimen.csv"
    )

has_pos = (
    st.session_state.df_pos is not None
    and not st.session_state.df_pos.empty
)

has_neg = (
    st.session_state.df_neg is not None
    and not st.session_state.df_neg.empty
)

# JIKA POSITIF & NEGATIF ADA
if has_pos and has_neg:

    col1, col2 = st.columns(2)

    # ---------- POSITIVE ----------
    with col1:
        st.subheader("🟢 Topik Sentimen Positif")

        df_pos = st.session_state.df_pos
        topic_pos_counts = df_pos["Topic"].value_counts()

        cmap = plt.get_cmap("berlin")
        colors = cmap(np.linspace(0.2, 0.8, len(topic_pos_counts)))

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(
            topic_pos_counts.index.astype(str),
            topic_pos_counts.values,
            color=colors
        )

        ax.set_xlabel("ID Topik")
        ax.set_ylabel("Jumlah Ulasan")
        ax.set_title("Distribusi Topik Positif")
        ax.set_ylim(0, topic_pos_counts.max() * 1.2)

        for i, v in enumerate(topic_pos_counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)

        dominant_topic = topic_pos_counts.idxmax()
        dominant_count = topic_pos_counts.max()

        topic_label = format_topic_label(
            dominant_topic,
            pos_label_map
        )

        st.markdown("**📌 Interpretasi Topik Positif:**")
        st.write(
            f"Topik {topic_label} merupakan topik paling dominan "
            f"pada sentimen positif dengan {dominant_count} ulasan. "
            "Hal ini menunjukkan aspek layanan tersebut menjadi sumber utama kepuasan pengguna."
        )

        with st.expander("🔍 Lihat Detail"):
            st.dataframe(df_pos, use_container_width=True)

        download_button(df_pos, "hasil_topik_positif.csv")

    # ---------- NEGATIVE ----------
    with col2:
        st.subheader("🔴 Topik Sentimen Negatif")

        df_neg = st.session_state.df_neg
        topic_neg_counts = df_neg["Topic"].value_counts()

        cmap = plt.get_cmap("berlin")
        colors = cmap(np.linspace(0.8, 0.2, len(topic_neg_counts)))

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(
            topic_neg_counts.index.astype(str),
            topic_neg_counts.values,
            color=colors
        )

        ax.set_xlabel("ID Topik")
        ax.set_ylabel("Jumlah Ulasan")
        ax.set_title("Distribusi Topik Negatif")
        ax.set_ylim(0, topic_neg_counts.max() * 1.2)

        for i, v in enumerate(topic_neg_counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)

        dominant_topic = topic_neg_counts.idxmax()
        dominant_count = topic_neg_counts.max()

        topic_label = format_topic_label(
            dominant_topic,
            neg_label_map
        )

        st.markdown("**📌 Interpretasi Topik Negatif:**")
        st.write(
            f"Topik {topic_label} paling sering muncul "
            f"pada sentimen negatif dengan {dominant_count} ulasan. "
            "Hal ini menunjukkan permasalahan utama yang menjadi sumber ketidakpuasan pengguna."
        )

        with st.expander("🔍 Lihat Detail"):
            st.dataframe(df_neg, use_container_width=True)

        download_button(df_neg, "hasil_topik_negatif.csv")



# JIKA HANYA POSITIF
elif has_pos:

    st.subheader("🟢 Topik Sentimen Positif")

    df_pos = st.session_state.df_pos
    topic_pos_counts = df_pos["Topic"].value_counts()

    cmap = plt.get_cmap("berlin")
    colors = cmap(np.linspace(0.2, 0.8, len(topic_pos_counts)))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(
        topic_pos_counts.index.astype(str),
        topic_pos_counts.values,
        color=colors
    )

    ax.set_xlabel("ID Topik")
    ax.set_ylabel("Jumlah Ulasan")
    ax.set_title("Distribusi Topik Sentimen Positif")

    for i, v in enumerate(topic_pos_counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

    dominant_topic = topic_pos_counts.idxmax()
    dominant_count = topic_pos_counts.max()

    topic_label = format_topic_label(
        dominant_topic,
        pos_label_map
    )

    st.markdown("**📌 Interpretasi:**")
    st.write(
        f"Topik {topic_label}  menjadi topik dominan "
        f"dalam sentimen positif dengan {dominant_count} ulasan."
    )

    with st.expander("🔍 Lihat Detail"):
        st.dataframe(df_pos, use_container_width=True)

    download_button(df_pos, "hasil_topik_positif.csv")



# JIKA HANYA NEGATIF

elif has_neg:

    st.subheader("🔴 Topik Sentimen Negatif")

    df_neg = st.session_state.df_neg
    topic_neg_counts = df_neg["Topic"].value_counts()

    cmap = plt.get_cmap("berlin")
    colors = cmap(np.linspace(0.8, 0.2, len(topic_neg_counts)))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(
        topic_neg_counts.index.astype(str),
        topic_neg_counts.values,
        color=colors
    )

    ax.set_xlabel("ID Topik")
    ax.set_ylabel("Jumlah Ulasan")
    ax.set_title("Distribusi Topik Sentimen Negatif")

    for i, v in enumerate(topic_neg_counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

    dominant_topic = topic_neg_counts.idxmax()
    dominant_count = topic_neg_counts.max()

    topic_label = format_topic_label(
        dominant_topic,
        neg_label_map
    )

    st.markdown("**📌 Interpretasi:**")
    st.write(
        f"Topik {topic_label} paling dominan "
        f"dalam sentimen negatif dengan {dominant_count} ulasan."
    )

    with st.expander("🔍 Lihat Detail"):
        st.dataframe(df_neg, use_container_width=True)

    download_button(df_neg, "hasil_topik_negatif.csv")
