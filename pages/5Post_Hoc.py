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


# Enhanced validation for clustered data
if 'clustered_data' not in st.session_state or st.session_state.clustered_data is None:
    st.error("Silakan lakukan clustering terlebih dahulu pada halaman sebelumnya.")
    st.switch_page("pages/4Training.py")
    st.stop()
else:
    df = st.session_state.clustered_data

    # Double-check that df is valid and has columns
    if df is None or not hasattr(df, 'columns'):
        st.error("Data clustering tidak valid. Silakan lakukan clustering ulang.")
        st.switch_page("pages/4Training.py")
        st.stop()

# Check for other necessary session state variables
required_vars = ['current_algorithm', 'scaler']
missing_vars = [
    var for var in required_vars if var not in st.session_state or st.session_state[var] is None]

if missing_vars:
    st.error(
        f"Silakan lakukan clustering terlebih dahulu. Data berikut hilang: {', '.join(missing_vars)}")
    st.switch_page("pages/4Training.py")
    st.stop()
else:
    current_algorithm = st.session_state.current_algorithm
    scaler = st.session_state.scaler

# Now it's safe to proceed with the analysis
st.title("Post Hoc Analysis")
st.write("Halaman ini digunakan untuk melakukan analisis lanjutan setelah melakukan clustering pada data yang telah diproses sebelumnya.")
st.info(f"Algoritma clustering yang digunakan: {current_algorithm}")
st.info(f"Menggunakan Scaler: {scaler}")

# Safer extraction of features
try:
    exclude_columns = ['cluster', 'label', 'umur', 'Cluster']
    all_features = [col for col in df.columns if col not in exclude_columns]

    if not all_features:
        st.warning(
            "Tidak ditemukan fitur untuk analisis. Pastikan data clustering berhasil dibuat.")
        st.stop()

except Exception as e:
    st.error(f"Error saat mengekstrak fitur: {str(e)}")
    st.stop()


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


# Add domain-based analysis after your existing ANOVA results display
st.subheader("Domain-Based Analysis")

# Define domains and their descriptions
domains = {
    "fa": "Financial Attitude",
    "fb": "Financial Behavior",
    "fk": "Financial Knowledge",
    "m": "Materialism"
}

# Group features by domain
domain_features = {domain: [] for domain in domains}
domain_results = {domain: [] for domain in domains}

# Categorize features and results by domain
for result in anova_results:
    feature = result["feature"]
    # Extract domain prefix (first 1-2 characters before the number)
    domain_prefix = ''.join([c for c in feature if not c.isdigit()]).lower()

    if domain_prefix in domains:
        domain_features[domain_prefix].append(feature)
        domain_results[domain_prefix].append(result)

# Display domain-based summaries
for domain_prefix, domain_name in domains.items():
    if domain_features[domain_prefix]:
        domain_results_list = domain_results[domain_prefix]

        # Calculate domain stats
        n_features = len(domain_results_list)
        n_significant = sum(r["significant"] for r in domain_results_list)
        avg_f_stat = np.mean([r["f_statistic"] for r in domain_results_list])

        # Create a domain results dataframe
        domain_df = pd.DataFrame([{
            "Feature": r["feature"],
            "F-statistic": r['f_statistic'],
            "p-value": r['p_value'],
            "Signifikan": "Ya ✅" if r["significant"] else "Tidak ❌"
        } for r in domain_results_list])

        # Display domain summary
        with st.expander(f"{domain_name} ({domain_prefix}) - {n_significant}/{n_features} fitur signifikan", expanded=True):
            # Display domain significance percentage
            significance_pct = (n_significant / n_features) * \
                100 if n_features > 0 else 0

            # Color-coded metric based on significance percentage
            if significance_pct > 75:
                st.success(
                    f"**{significance_pct:.1f}%** fitur dalam domain ini signifikan berbeda antar cluster")
            elif significance_pct > 50:
                st.info(
                    f"**{significance_pct:.1f}%** fitur dalam domain ini signifikan berbeda antar cluster")
            elif significance_pct > 25:
                st.warning(
                    f"**{significance_pct:.1f}%** fitur dalam domain ini signifikan berbeda antar cluster")
            else:
                st.error(
                    f"**{significance_pct:.1f}%** fitur dalam domain ini signifikan berbeda antar cluster")

            # Domain-specific insights
            if domain_prefix == "fa":
                st.write(
                    "**Insight:** Domain ini mengukur sikap responden terhadap keuangan, seperti pentingnya menabung, perencanaan, dan anggaran.")
            elif domain_prefix == "fb":
                st.write(
                    "**Insight:** Domain ini mengukur perilaku aktual responden dalam mengelola keuangan, seperti mencatat pengeluaran dan menabung.")
            elif domain_prefix == "fk":
                st.write(
                    "**Insight:** Domain ini mengukur pengetahuan responden tentang konsep keuangan dasar.")
            elif domain_prefix == "m":
                st.write(
                    "**Insight:** Domain ini mengukur tingkat materialisme responden, seperti kecenderungan menilai kesuksesan berdasarkan kepemilikan barang.")

            # Display domain features table
            st.dataframe(domain_df.sort_values("F-statistic", ascending=False).style.format({
                "F-statistic": "{:.4f}",
                "p-value": "{:.4f}"
            }))

            # Create bar chart of F-statistics for domain features
            if len(domain_features[domain_prefix]) > 0:
                fig, ax = plt.subplots(
                    figsize=(10, max(4, len(domain_features[domain_prefix])*0.4)))

                sorted_indices = np.argsort(
                    [r["f_statistic"] for r in domain_results_list])
                sorted_features = [domain_results_list[i]
                                   ["feature"] for i in sorted_indices]
                sorted_f_stats = [domain_results_list[i]
                                  ["f_statistic"] for i in sorted_indices]
                significant = [domain_results_list[i]["significant"]
                               for i in sorted_indices]

                # Create horizontal bar chart
                bars = ax.barh(sorted_features, sorted_f_stats)

                # Color bars based on significance
                for i, sig in enumerate(significant):
                    bars[i].set_color('green' if sig else 'gray')

                ax.set_xlabel('F-statistic')
                ax.set_title(f'Signifikansi Fitur dalam Domain {domain_name}')
                ax.axvline(x=3.0, color='red', linestyle='--',
                           label='Threshold umum signifikansi')
                ax.legend()

                st.pyplot(fig)

# Store domain-specific results in session state
domain_summary = {domain: {
    "name": domains[domain],
    "n_features": len(domain_results[domain]),
    "n_significant": sum(r["significant"] for r in domain_results[domain]),
    "avg_f_statistic": np.mean([r["f_statistic"] for r in domain_results[domain]]) if domain_results[domain] else 0,
    "significant_pct": (sum(r["significant"] for r in domain_results[domain]) / len(domain_results[domain]) * 100) if domain_results[domain] else 0
} for domain in domains}

st.session_state.domain_analysis = domain_summary

# Highlight domain with highest significance
if any(domain_summary.values()):
    most_significant_domain = max(domain_summary.items(
    ), key=lambda x: x[1]["significant_pct"] if x[1]["n_features"] > 0 else 0)

    st.subheader("Kesimpulan Analisis Domain")
    st.info(f"""
    Domain **{most_significant_domain[1]['name']}** menunjukkan perbedaan paling signifikan antar cluster 
    ({most_significant_domain[1]['significant_pct']:.1f}% fitur signifikan).
    
    Hal ini mengindikasikan bahwa clustering yang dihasilkan paling baik dalam membedakan responden berdasarkan 
    aspek **{most_significant_domain[1]['name'].lower()}** mereka.
    """)
