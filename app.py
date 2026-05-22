import os
import streamlit as st
import pandas as pd

st.title("🏗️ HDB Dashboard Diagnostic Mode")
st.write("If you can see this message, the server connection is working perfectly!")

# Diagnostic check
if os.path.exists("hdb_resale_flats.csv"):
    st.success("✅ Found hdb_resale_flats.csv in root folder!")
    # Show structural size metrics
    try:
        df_test = pd.read_csv("hdb_resale_flats.csv", nrows=5)
        st.write("Data snippet preview:", df_test)
    except Exception as e:
        st.error(f"Could not read CSV file parser: {e}")
else:
    st.error("❌ Cannot find hdb_resale_flats.csv in root folder.")