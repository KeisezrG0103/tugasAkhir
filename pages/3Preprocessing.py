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

if 'df' not in st.session_state:
    # Load environment variables from .env file
    # Get data path from environment variables
    data_path = os.getenv("data/ta_dataset.csv")
    # Load data using the path
    df = dp.data_loader(data_path)
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
