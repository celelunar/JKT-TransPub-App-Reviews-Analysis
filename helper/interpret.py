def compute_sentiment_metrics(df_sent):
    total_reviews = len(df_sent)
    sent_counts = df_sent["Sentiment"].value_counts()

    pos_count = sent_counts.get("Positif", 0)
    neg_count = sent_counts.get("Negatif", 0)
    avg_conf = df_sent["Confidence"].mean()

    return {
        "total": total_reviews,
        "pos": pos_count,
        "neg": neg_count,
        "avg_conf": avg_conf,
        "counts": sent_counts
    }

def sentiment_interpretation(metrics):
    pos = metrics["pos"]
    neg = metrics["neg"]

    if pos > neg:
        summary = "didominasi oleh sentimen positif"
    elif neg > pos:
        summary = "didominasi oleh sentimen negatif"
    else:
        summary = "memiliki distribusi sentimen yang relatif seimbang"

    return (
        f"Dari total {metrics['total']} ulasan yang dianalisis, "
        f"hasil menunjukkan bahwa distribusi sentimen {summary}. "
        "Nilai confidence rata-rata yang relatif tinggi mengindikasikan bahwa "
        "model memiliki tingkat keandalan yang baik dalam mengklasifikasikan "
        "sentimen ulasan pengguna."
    )

def format_topic_label(topic_id, label_map):
    if topic_id in label_map:
        return f"{topic_id} ({label_map[topic_id]})"
    return f"{topic_id}"


def topic_interpretation(topic_counts, label_map, sentiment):
    dominant_topic = topic_counts.idxmax()
    dominant_count = topic_counts.max()

    topic_label = format_topic_label(dominant_topic, label_map)

    if sentiment == "positif":
        return(
            f"Topik {topic_label} merupakan topik paling dominan pada sentimen positif dengan {dominant_count} ulasan. Hal ini menunjukkan aspek layanan tersebut menjadi sumber utama kepuasan pengguna."
        )
    else:
        return(
            f"Topik {topic_label} merupakan topik paling dominan pada sentimen negatif dengan {dominant_count} ulasan. Hal ini menunjukkan permasalahan utama yang menjadi sumber ketidakpuasan pengguna."
        )