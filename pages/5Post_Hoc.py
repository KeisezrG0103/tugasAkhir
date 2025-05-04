import streamlit as st
import os
import numpy as np
import utils.data_preprocessing as dp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import utils.Gaussian_Mixture as gmm
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


if 'clustered_data' not in st.session_state:
    st.error("Silakan lakukan clustering terlebih dahulu pada halaman sebelumnya.")
    st.switch_page("pages/4Training.py")
    st.stop()
else:
    df = st.session_state.clustered_data

if 'current_algorithm' not in st.session_state and 'scaler' not in st.session_state:
    st.error("Silakan lakukan clustering terlebih dahulu pada halaman sebelumnya.")
    st.switch_page("pages/4Training.py")
    st.stop()
else:
    current_algorithm = st.session_state.current_algorithm
    scaler = st.session_state.scaler


st.title("Post Hoc Analysis")
st.write("Halaman ini digunakan untuk melakukan analisis lanjutan setelah melakukan clustering pada data yang telah diproses sebelumnya.")
st.info(f"Algoritma clustering yang digunakan: {current_algorithm}")
st.info(f"Menggunakan Scaler: {scaler}")

all_features = [col for col in df.columns if col not in [
    'cluster', 'label', 'umur', 'cluster']]

anova_features = st.multiselect(
    "Pilih fitur untuk ANOVA",
    options=all_features,
    default=all_features[:2]
)

anova_results = []

cluster_col = 'Cluster' if 'Cluster' in df.columns else 'cluster'


for feature in all_features:
    # Group data by cluster for this feature
    result = ph.run_anova_analysis(df, cluster_col, feature)

    anova_results.append(result)
    st.session_state.anova_results = anova_results


# Display ANOVA results in a table
if anova_results:
    st.subheader("Hasil One-Way ANOVA")

    # Create summary table
    anova_df = pd.DataFrame([{
        "Feature": r["feature"],
        "F-statistic": r['f_statistic'],  # Store raw numeric values
        "p-value": r['p_value'],          # Store raw numeric values
        "Signifikan": "Ya ✅" if r["significant"] else "Tidak ❌"
    } for r in anova_results])

    st.dataframe(anova_df.style.format({
        "F-statistic": "{:.4f}",
        "p-value": "{:.4f}"
    }))
    st.write("5 Fitur paling signifikan:")
    st.dataframe(anova_df.nlargest(5, "F-statistic").style.format({
        "F-statistic": "{:.4f}",
        "p-value": "{:.4f}"
    }))
    st.write("5 Fitur paling tidak signifikan:")
    st.dataframe(anova_df.nsmallest(5, "F-statistic").style.format({
        "F-statistic": "{:.4f}",
        "p-value": "{:.4f}"
    }))


st.subheader("Detail Analisis per Fitur")


for result in anova_results:
    if result["feature"] in anova_features:

        with st.expander(f"Detail {result['feature']}", expanded=False):
            ph.Run_Detail_Analysis(df, cluster_col, result)

# Tambahkan ringkasan ANOVA untuk semua fitur
if anova_results:
    st.subheader("Ringkasan Analisis ANOVA")

    # Hitung jumlah fitur signifikan
    n_significant = sum(r["significant"] for r in anova_results)

    if n_significant > 0:
        st.success(
            f"{n_significant} dari {len(anova_results)} fitur menunjukkan perbedaan signifikan antar cluster.")

        # Interpretasi hasil clustering
        st.write("#### Interpretasi Hasil Clustering:")
        st.write("""
        Berdasarkan hasil ANOVA, cluster yang terbentuk menunjukkan perbedaan karakteristik pada fitur-fitur tersebut. 
        Hal ini mengindikasikan bahwa algoritma clustering berhasil membentuk kelompok dengan karakteristik yang berbeda secara statistik.
        
        Untuk analisis lebih mendalam, perhatikan hasil analisis per fitur dan Tukey HSD untuk melihat pasangan cluster mana yang paling berbeda.
        """)
    else:
        st.warning(
            "Tidak ada fitur yang menunjukkan perbedaan signifikan antar cluster.")
        st.write("""
        Meskipun clustering telah membentuk kelompok, tidak ada perbedaan statistik yang signifikan pada fitur-fitur yang dianalisis.
        Hal ini bisa mengindikasikan bahwa:
        - Cluster yang terbentuk memiliki karakteristik yang mirip
        - Fitur yang dipilih mungkin tidak cukup diskriminatif
        - Jumlah cluster mungkin perlu disesuaikan
        
        Pertimbangkan untuk:
        - Mencoba algoritma clustering lain
        - Menambah atau mengubah fitur yang dianalisis
        - Mengubah jumlah cluster
        """)

# Simpan hasil ANOVA ke session state untuk digunakan di page lain
st.session_state.anova_results = anova_results
