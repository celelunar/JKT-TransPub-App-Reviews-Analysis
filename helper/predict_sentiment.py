import torch

LABEL_MAP = {
    "LABEL_0": "Negatif",
    "LABEL_1": "Positif"
}

@st.cache_data(show_spinner=False)
def predict_sentiment(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confs, preds = torch.max(probs, dim=1)

    sentiments = [
        LABEL_MAP[model.config.id2label[p.item()]] for p in preds
    ]

    return sentiments, confs.numpy()