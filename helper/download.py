from io import BytesIO
import streamlit as st

def download_csv(df, filename, nameplate):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label=f"Download {nameplate}",
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv"
    )