import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Welcome!",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Brief Explanation ---
st.title("🚌 Public Transportation Application Reviews Analyzer")
st.markdown("""
This website showcases the deployment of advanced Natural Language Processing (NLP) models using **IndoBERT**, **NusaBERT**, and **BERTopic**. It is designed to analyze user reviews from Jakarta's public transportation applications on Google Play, which are: Access by KAI, Jak Lingko App, MyMRTJ, and TJ: Transjakarta.
""")

# --- Main Features ---
st.subheader("What You Can Do in This Web")
st.markdown("""
* **🚦 Predict Sentiment:** Classify reviews into sentiment categories (positive or negative) using the pre-trained BERT model.
            
* **🔍 Discover Topics:** Assign reviews to one of the existing topics identified by the pre-trained BERTopic model.
""")

# --- Directory ---
st.subheader("Web Directory")
st.info("""
**🏠 Welcome Page:** An overview of the website's purpose and capabilities.

**📊 Data and Method:** Detailed information about the dataset and NLP techniques used to build the predictive models.

**⚙️ Model Usage:** An interactive interface where you can input your own review and run the deployed predictive models.

**💡 Tutorial:** A step-by-step guide to help you use the Model Usage page effectively.
""")

# Note for users
st.markdown("---")
st.warning("**👈 Use the sidebar to naivage through the pages.**")