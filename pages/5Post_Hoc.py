import streamlit as st
import os
import numpy as np
import utils.data_preprocessing as dp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import utils.Kmeans as kmeans
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import utils.Hierarchical_clustering as hc
import utils.post_hoc as ph
st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

domain_prefix = {
    "fa": "Financial Attitude",
    "fb": "Financial Behavior",
    "fk": "Financial Knowledge",
    "m": "Materialism"
}


if 'df_with_clusters' in st.session_state:
    st.write("df_with_clusters exists, shape:",
             st.session_state.df_with_clusters.shape)
# Enhanced validation for clustered data
if 'clustered_data' not in st.session_state or st.session_state.clustered_data is None:
    st.error("Silakan lakukan clustering terlebih dahulu pada halaman sebelumnya.")
    # st.switch_page("pages/4Training.py")
    st.stop()
else:
    df_clustered = st.session_state.clustered_data
    df = st.session_state.df
    # Check if the DataFrame is empty
    if df_clustered.empty or df.empty:
        st.error("DataFrame is empty. Please check the clustering process.")
        st.stop()
    else:
        st.success("DataFrame loaded successfully.")
        st.write("## Data setelah clustering")
        st.dataframe(df_clustered)
        st.write("## Data asli")
        st.dataframe(df)


anova_results, domain_dict = ph.run_anova_tukey_analysis(
    df, domain_prefix, df_clustered)

ph.Interactive_analysis(anova_results, domain_dict)


# st.write("DEBUG")
# st.write("Session state keys:", list(st.session_state.keys()))
# st.write(st.session_state.scaler)
# st.dataframe(st.session_state.clustered_data)
# st.dataframe(st.session_state.anova_results)
# st.dataframe(st.session_state.domain_cluster_results)
# st.dataframe(st.session_state.cluster_pattern_counts)
