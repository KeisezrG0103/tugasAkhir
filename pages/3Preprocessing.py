import streamlit as st
import pandas as pd
import utils.data_preprocessing as dp
import utils.data_visualization as dv
import dotenv
import os
# dotenv.load_dotenv()
from pathlib import Path

st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'df' not in st.session_state:
    try:
        # Use direct path instead of trying to get it from environment variables
        data_path = "data/ta_dataset.csv"

        # Check if file exists
        if not Path(data_path).exists():
            st.error(f"File not found: {data_path}")
            st.info("Please make sure the data file exists in the correct location.")
            st.stop()

        # Load data using the path
        df = dp.data_loader(data_path)
        st.success(f"Data loaded successfully from {data_path}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using the direct path as fallback")

        # Fallback to the hardcoded path that's used later in the code
        try:
            df = dp.data_loader("data/ta_dataset.csv")
            st.success("Data loaded successfully from fallback path")
        except Exception as e2:
            st.error(f"Failed to load data from fallback path: {str(e2)}")
            st.stop()
else:
    df = st.session_state.df


st.title("Data Preprocessing")

st.markdown(
    """
    1. Drop semua kolom yang tidak diperlukan
    2. Likerd Reverse Coding (Melakukan reverse coding pada kolom yang diperlukan)
    3. Memilih kolom yang diperlukan
    4. Melihat multicollinearity
    5. Melakukan skalasi data
    """
)

st.write("## Data sebelum preprocessing")
original_df = dp.data_loader("data/ta_dataset.csv")
st.dataframe(original_df)
df = dp.Preprocess_Data(df)
st.write("## Data setelah preprocessing")
st.dataframe(df)


st.session_state.df = df

threshold = st.slider(
    "Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
)

multicollinearity = dp.pairwise_multicollinearity(df, threshold)
if len(multicollinearity) > 0:
    st.error(
        f"Membutuhkan penggunaan PCA")
    st.session_state.multicollinearity = multicollinearity
else:
    st.info(f"Tidak ada fitur yang memiliki korelasi lebih dari {threshold}.")
    st.session_state.multicollinearity = []

st.write("## Multicollinearity")
dp.plot_correlation_matrix(df, threshold)
dp.graph_pairwise_correlation(df, threshold)
