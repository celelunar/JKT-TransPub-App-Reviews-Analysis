import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from data.sample_texts import SAMPLE_TEXTS
from helper.preprocessing import preprocess_batch
from helper.predict_sentiment import predict_sentiment
from helper.predict_topic import predict_topic
from helper.charts import sentiment_bar_chart, topic_bar_chart
from helper.interpret import topic_interpretation, compute_sentiment_metrics, sentiment_interpretation
from helper.download import download_csv

# --- Label Map ---
STAT_FILE_PATH = 'data/data.xlsx'

pos_lab = pd.read_excel(STAT_FILE_PATH, sheet_name="pos_lab")
neg_lab = pd.read_excel(STAT_FILE_PATH, sheet_name="neg_lab")

pos_label_map = dict(zip(pos_lab["Topic"], pos_lab["Label"]))
neg_label_map = dict(zip(neg_lab["Topic"], neg_lab["Label"]))

# --- Get Models ---
models = st.session_state.models

tokenizer = models["tokenizer"]
sa_mod = models["sa_mod"]
pos_mod = models["pos_mod"]
neg_mod = models["neg_mod"]

# --- INTERFACE ---
# --- Page Config ---
st.set_page_config(
    page_title="Penggunaan Model",
    page_icon="⚙️",
    layout='wide'
)
# --- session state ---
if "df_sent" not in st.session_state:
    st.session_state.df_sent = None

if "df_pos" not in st.session_state:
    st.session_state.df_pos = None

if "df_neg" not in st.session_state:
    st.session_state.df_neg = None

if "cleaned_texts" not in st.session_state:
    st.session_state.cleaned_texts = None


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
    cleaned_texts = preprocess_batch(texts)

    # --- Sentiment ---
    sentiments, confs = predict_sentiment(cleaned_texts, tokenizer, sa_mod)

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

    st.subheader("🚦 Hasil Analisis Sentimen")

    df_sent = st.session_state.df_sent
    df_pos = st.session_state.df_pos
    df_neg = st.session_state.df_neg

    metrics = compute_sentiment_metrics(df_sent)
    
    st.write(sentiment_interpretation(metrics))
     
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        sentiment_bar_chart(metrics["counts"])
    with col2:
        st.metric("Total Ulasan", metrics["total"])
        st.metric("Rata-rata *Confidence*", f"{metrics['avg_conf']:.2f}")
    with col3:
        st.metric(
            "Positif",
            metrics["pos"],
            f"{metrics['pos']/metrics['total']:.1%}"
        )
        st.metric(
            "Negatif",
            metrics["neg"],
            f"{metrics['neg']/metrics['total']:.1%}",
            delta_color="inverse"
        )

    with st.expander("🔍 Lihat Detail Hasil"):
        st.dataframe(
            st.session_state.df_sent,
            use_container_width=True
        )

    download_csv(
        st.session_state.df_sent,
        "hasil_sentimen.csv",
        "Hasil Prediksi Sentimen"
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

        fig = topic_bar_chart(topic_pos_counts, "Distribusi Topik Positif")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**📌 Interpretasi Topik Positif:**")
        st.write(topic_interpretation(topic_pos_counts, pos_label_map, "positif"))

        with st.expander("🔍 Lihat Detail"):
            st.dataframe(df_pos, use_container_width=True)

        download_csv(df_pos, "hasil_topik_positif.csv", "Hasil Topik Positif")
    # ---------- NEGATIVE ----------
    with col2:
        st.subheader("🔴 Topik Sentimen Negatif")

        df_neg = st.session_state.df_neg
        topic_neg_counts = df_neg["Topic"].value_counts()

        fig = topic_bar_chart(topic_neg_counts, "Distribusi Topik Negatif")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**📌 Interpretasi Topik Negatif:**")
        st.write(topic_interpretation(topic_neg_counts, neg_label_map, "negatif"))

        with st.expander("🔍 Lihat Detail"):
            st.dataframe(df_neg, use_container_width=True)

        download_csv(df_neg, "hasil_topik_negatif.csv", "Hasil Topik Negatif")
# JIKA HANYA POSITIF
elif has_pos:
    st.subheader("🟢 Topik Sentimen Positif")

    df_pos = st.session_state.df_pos
    topic_pos_counts = df_pos["Topic"].value_counts()

    fig = topic_bar_chart(topic_pos_counts, "Distribusi Topik Positif")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**📌 Interpretasi Topik Positif:**")
    st.write(topic_interpretation(topic_pos_counts, pos_label_map, "positif"))

    with st.expander("🔍 Lihat Detail"):
        st.dataframe(df_pos, use_container_width=True)

    download_csv(df_pos, "hasil_topik_positif.csv", "Hasil Topik Positif")
# JIKA HANYA NEGATIF
elif has_neg:
    st.subheader("🔴 Topik Sentimen Negatif")

    df_neg = st.session_state.df_neg
    topic_neg_counts = df_neg["Topic"].value_counts()

    fig = topic_bar_chart(topic_neg_counts, "Distribusi Topik Negatif")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**📌 Interpretasi Topik Negatif:**")
    st.write(topic_interpretation(topic_neg_counts, neg_label_map, "negatif"))

    with st.expander("🔍 Lihat Detail"):
        st.dataframe(df_neg, use_container_width=True)

    download_csv(df_neg, "hasil_topik_negatif.csv", "Hasil Topik Negatif")
