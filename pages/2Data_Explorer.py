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

data_path = "data/ta_dataset.csv"
if 'original_df' not in st.session_state:
    st.session_state.original_df = dp.data_loader(data_path)
    st.session_state.original_df = st.session_state.original_df[(st.session_state.original_df['Umur'] > 17) & (st.session_state.original_df['Umur'] <= 23)]  # Filter data for age between 18 and 23

df = st.session_state.original_df

st.title("Eksplorasi Data")


dataframe = st.dataframe(df, use_container_width=True)

# Age Distribution Visualization
st.header("Visualisasi Data")

# Select the column to visualize
df_copy = df.copy()

col_to_remove = ['Email', 'Nama', 'Timestamp', 'email', 'nama', 'timestamp']
for col in col_to_remove:
    if col in df_copy.columns:
        df_copy.drop(col, axis=1, inplace=True)

selected_column = st.selectbox(
    "Pilih kolom yang akan divisualisasikan", df_copy.columns.tolist())

# Display the distribution of the selected column
if selected_column:
    dv.distribution_by_request(df_copy, selected_column)
