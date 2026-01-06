import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def bar_chart(df, x_col, y_col, is_sentiment=False):   
    if is_sentiment:
        df_plot = df[df[x_col].str.lower() != 'netral'].copy()
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

def resample_chart(df):  
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

def sentiment_bar_chart(sent_counts):
    cmap = plt.get_cmap("berlin")

    color_pos = cmap(0.2)
    color_neg = cmap(0.8) 

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

def topic_bar_chart(topic_counts, title, cmap_range=(0.2, 0.8), figsize=(5,3)):
    cmap = plt.get_cmap("berlin")

    colors = cmap(np.linspace(*cmap_range, len(topic_counts)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        topic_counts.index.astype(str),
        topic_counts.values,
        color=colors
    )

    ax.set_xlabel("ID Topik")
    ax.set_ylabel("Jumlah Ulasan")
    ax.set_title(title)
    ax.set_ylim(0, topic_counts.max() * 1.2)

    for i, v in enumerate(topic_counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    return fig  