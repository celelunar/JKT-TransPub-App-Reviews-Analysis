import pandas as pd
import streamlit as st

@st.cache_data    
def load_excel(path, sheet_name):
    try:
        df = pd.read_excel(
            path, 
            sheet_name=sheet_name, 
            engine='openpyxl'
        )
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: Excel file not found at '{path}'")
        return pd.DataFrame()
    except ValueError:
        st.error(f"❌ Error: Sheet '{sheet_name}' not found in the Excel file.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        return pd.DataFrame()