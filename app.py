import streamlit as st

# --- Configure Page Settings ---
st.set_page_config(
    page_title="BERT & BERTopic Deployment App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Welcome Page Content ---
st.title("🤖 Welcome to the Sentiment & Topic Analyzer App")
st.markdown("""
This application demonstrates the deployment of advanced Natural Language Processing (NLP) models. 
You can use it to analyze custom datasets for sentiment and topics, leveraging the power of Transformer-based models.
""")

st.header("What You Can Do Here")
st.markdown("""
* **Predict Sentiment:** Classify text data into specific sentiment categories (e.g., Positive, Negative).
* **Discover Topics:** Automatically extract and label the main themes and topics within your text data.
""")

st.subheader("App Directory")

# A directory/short explanation for each page
st.info("""
**🏠 Welcome Page (This Page):** An introduction to the application's capabilities.

**📚 Data and Method:** Provides a deep dive into the dataset used for model training and the underlying NLP methodologies, including a comparison of models like IndoBERT, NusaBERT, and BERTopic.

**⚙️ Model Usage (To be built):** The interactive page where you can upload your own data (`.csv` file) and run the deployed Sentiment Analysis and Topic Modelling models.

**💡 Tutorial (To be built):** A guide on how to effectively use the 'Model Usage' page.
""")

# Note for users
st.markdown("---")
st.warning("**👈 Please use the sidebar to navigate between pages.**")

# Placeholder for data/model loading (good practice for main file)
# In a real app, you would load models and data here using @st.cache_resource
# and share them across pages using st.session_state or a utility module.
# st.session_state["data"] = pd.read_csv("your_data.csv")
# st.session_state["sentiment_model"] = load_sentiment_model("saved_sentiment_model.joblib")