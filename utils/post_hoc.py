import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


def Run_Detail_Analysis(df, cluster_col, result):
    col1, col2 = st.columns([2, 1])

    with col1:
        # Statistik deskriptif per cluster
        stats_df = pd.DataFrame({
            cluster: {
                "Jumlah": stats["count"],
                "Rata-rata": stats["mean"],
                "Std. Deviasi": stats["std"],
                "Minimum": stats["min"],
                "Maximum": stats["max"]
            } for cluster, stats in result["group_stats"].items()
        }).T  # Transpose untuk format yang lebih baik

        st.write("#### Statistik Deskriptif per Cluster:")
        st.dataframe(stats_df.style.format({
            "Rata-rata": "{:.2f}",
            "Std. Deviasi": "{:.2f}",
            "Minimum": "{:.2f}",
            "Maximum": "{:.2f}"
        }))

    with col2:
        # Informasi ANOVA
        st.write("#### Hasil ANOVA:")
        st.metric("F-statistic", f"{result['f_statistic']:.4f}")
        st.metric("p-value", f"{result['p_value']:.4f}")

        if result["significant"]:
            st.success("Terdapat perbedaan signifikan antar cluster")
        else:
            st.info("Tidak ada perbedaan signifikan antar cluster")

            # Visualisasi boxplot perbandingan
    st.write("#### Perbandingan Distribusi per Cluster:")

    # Persiapkan data untuk boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data menggunakan boxplot
    boxplot_data = [df[df[cluster_col] == cluster][result["feature"]]
                    for cluster in sorted(df[cluster_col].unique())]

    # Buat boxplot dengan warna
    bp = ax.boxplot(boxplot_data, patch_artist=True)

    # Warna untuk boxplot
    colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

        # Kustomisasi plot
    ax.set_title(f'Distribusi {result["feature"]} per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(result["feature"])
    ax.set_xticklabels(
        [f'Cluster {i}' for i in sorted(df[cluster_col].unique())])
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tampilkan plot
    st.pyplot(fig)

    # Bar chart untuk perbandingan rata-rata
    st.write("#### Perbandingan Rata-rata per Cluster:")

    # Persiapkan data rata-rata untuk bar chart
    means = [stats["mean"]
             for _, stats in sorted(result["group_stats"].items())]
    clusters = [cluster for cluster, _ in sorted(
                result["group_stats"].items())]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.bar(clusters, means, color=colors)

    # Tambahkan nilai di atas bar
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(means),
                 f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')

    ax2.set_title(f'Rata-rata {result["feature"]} per Cluster')
    ax2.set_ylabel(result["feature"])
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig2)

    # Tampilkan hasil Tukey test jika signifikan
    if result["significant"] and "tukey" in result:
        st.write("#### Hasil Post-Hoc Test (Tukey HSD)")

        # Ambil objek tukey
        tukey = result["tukey"]["tukey_object"]

    # Create all pairwise combinations manually
        pairs = []
        for i, grp1 in enumerate(tukey.groupsunique):
            for j, grp2 in enumerate(tukey.groupsunique):
                if j > i:  # Only keep unique pairs (no duplicates)
                    pairs.append((grp1, grp2))

                # Create DataFrame from Tukey results
        tukey_data = []
        for idx, (grp1, grp2) in enumerate(pairs):
            lower_ci, upper_ci = tukey.confint[idx]
            tukey_data.append({
                'Cluster 1': grp1,
                'Cluster 2': grp2,
                'Perbedaan Mean': tukey.meandiffs[idx],
                'CI Lower': lower_ci,
                'CI Upper': upper_ci,
                'p-value': tukey.pvalues[idx],
                'Signifikan': 'Ya ✅' if tukey.reject[idx] else 'Tidak ❌'
            })

        tukey_df = pd.DataFrame(tukey_data)

        # Visualisasi Tukey HSD
        st.write("#### Visualisasi Tukey HSD:")

        # Plot menggunakan matplotlib
        fig3, ax3 = plt.subplots(
            figsize=(10, max(3, len(tukey_df)*0.3)))

        # Persiapkan data untuk plot
        pairs = [f"{row['Cluster 1']} vs {row['Cluster 2']}" for _,
                 row in tukey_df.iterrows()]
        reject = tukey_df['Signifikan'].map(
            {'Ya ✅': True, 'Tidak ❌': False}).values

        # Buat plot dengan warna berbeda berdasarkan signifikansi
        bars = ax3.barh(
            pairs,
            tukey_df['Perbedaan Mean'].values,
            color=[plt.cm.RdYlGn(0.8) if r else plt.cm.Blues(0.5)
                   for r in reject],
            xerr=tukey_df['CI Upper'] - tukey_df['Perbedaan Mean'],
            capsize=5
        )

        # Tambahkan garis vertikal di x=0
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Label pada plot
        ax3.set_title('Perbandingan Mean antar Cluster (Tukey HSD)')
        ax3.set_xlabel('Perbedaan Mean')
        ax3.set_ylabel('Perbandingan')

        # Tambahkan p-value di sebelah kanan bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p_value = tukey_df['p-value'].values[i]
            sign = "✅" if reject[i] else "❌"
            ax3.text(
                max(width + 0.05,
                    0.05) if width >= 0 else min(width - 0.05, -0.05),
                bar.get_y() + bar.get_height()/2,
                f'p={p_value:.4f} {sign}',
                va='center'
            )

            # Tampilkan plot
        st.pyplot(fig3)

        # Kesimpulan Tukey HSD
        st.write("#### Kesimpulan Tukey HSD:")

        # Hitung jumlah perbandingan signifikan
        n_significant = sum(tukey_df['Signifikan'] == 'Ya ✅')

        if n_significant == 0:
            st.info(
                "Meskipun ANOVA signifikan, tidak ada perbedaan signifikan antar pasangan cluster.")
        else:
            st.success(
                f"Ditemukan {n_significant} pasangan cluster yang berbeda secara signifikan.")

            # Daftar pasangan yang signifikan
            sig_pairs = tukey_df[tukey_df['Signifikan'] == 'Ya ✅']

            for _, row in sig_pairs.iterrows():
                st.write(
                    f"- **{row['Cluster 1']} vs {row['Cluster 2']}**: Perbedaan mean = {row['Perbedaan Mean']:.4f} (p = {row['p-value']:.4f})")


@st.cache_data
def run_anova_analysis(df, cluster_col, feature):
    groups = df.groupby(cluster_col)[feature].apply(list).values

    # Perform one-way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)

    # Store results
    result = {
        "feature": feature,
        "f_statistic": f_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "group_stats": {}
    }

    # Calculate descriptive statistics for each cluster
    for cluster in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster][feature]
        result["group_stats"][f"Cluster {cluster}"] = {
            "count": len(cluster_data),
            "mean": cluster_data.mean(),
            "std": cluster_data.std(),
            "min": cluster_data.min(),
            "max": cluster_data.max()
        }

    # Perform Tukey's HSD post-hoc test if ANOVA is significant
    if p_val < 0.05:
        # Prepare data for Tukey's test
        tukey_data = []
        tukey_groups = []

        tukey_analysis(df, cluster_col, feature,
                       result, tukey_data, tukey_groups)

    return result


def tukey_analysis(df, cluster_col, feature, result, tukey_data, tukey_groups):
    for cluster in sorted(df[cluster_col].unique()):
        values = df[df[cluster_col] == cluster][feature].values
        tukey_data.extend(values)
        tukey_groups.extend([f"Cluster {cluster}"] * len(values))

        # Perform Tukey's HSD
    tukey = pairwise_tukeyhsd(
        endog=tukey_data,
        groups=tukey_groups,
        alpha=0.05
    )

    # Add Tukey results to our results dict
    result["tukey"] = {
        "tukey_object": tukey,
        "comparison_data": list(zip(tukey.groupsunique, tukey.meandiffs, tukey.pvalues, tukey.reject))
    }
