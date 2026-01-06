from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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