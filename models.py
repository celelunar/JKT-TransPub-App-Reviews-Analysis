import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic

SENTIMENT_REPO = "chimons-academy/indobert-jkt-transpub-app-review"
TOPIC_POS_REPO = "chimons-academy/bertopic-jkt-transpub-app-pos-review"
TOPIC_NEG_REPO = "chimons-academy/bertopic-jkt-transpub-app-neg-review"


@st.cache_resource
def load_all_models():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_REPO)
    sa_mod = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_REPO
    )
    sa_mod.eval()

    pos_mod = BERTopic.load(TOPIC_POS_REPO)
    neg_mod = BERTopic.load(TOPIC_NEG_REPO)

    return {
        "tokenizer": tokenizer,
        "sa_mod": sa_mod,
        "pos_mod": pos_mod,
        "neg_mod": neg_mod,
    }