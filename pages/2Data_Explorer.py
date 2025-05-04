import streamlit as st
import pandas as pd
import utils.data_preprocessing as dp
import utils.data_visualization as dv
import dotenv
import os
# dotenv.load_dotenv()

st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Always load the original data for Data Explorer page
data_path = "data/ta_dataset.csv"
# Store original data in session state if not already there
if 'original_df' not in st.session_state:
    st.session_state.original_df = dp.data_loader(data_path)

# Use original data for this page
df = st.session_state.original_df

st.title("Eksplorasi Data")


dataframe = st.dataframe(df, use_container_width=True)

# Age Distribution Visualization
st.header("Visualisasi Data")

# Select the column to visualize
df_copy = df.copy()

# Remove non-numeric columns for visualization
col_to_remove = ['Email', 'Nama', 'Timestamp', 'email', 'nama', 'timestamp']
for col in col_to_remove:
    if col in df_copy.columns:
        df_copy.drop(col, axis=1, inplace=True)

selected_column = st.selectbox(
    "Pilih kolom yang akan divisualisasikan", df_copy.columns.tolist())

# Display the distribution of the selected column
if selected_column:
    dv.distribution_by_request(df_copy, selected_column)
