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
import plotly.express as px


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


def analyze_domain_significance(df, domain_prefix, domain_name, anova_results):
    """
    Perform domain-specific analysis of feature significance.

    Parameters:
    - df: DataFrame with cluster assignments
    - domain_prefix: Prefix string that identifies features in this domain (e.g., 'fa')
    - domain_name: Human-readable name of the domain (e.g., 'Financial Attitude')
    - anova_results: List of ANOVA result dictionaries from run_anova_analysis

    Returns:
    - Dictionary of domain summary statistics
    """
    # Filter features and results for this domain
    domain_features = [f for f in df.columns if f.startswith(domain_prefix)]
    domain_results = [
        r for r in anova_results if r['feature'] in domain_features]

    # Calculate domain statistics
    n_features = len(domain_results)
    if n_features == 0:
        return {
            "domain": domain_prefix,
            "name": domain_name,
            "n_features": 0,
            "n_significant": 0,
            "significant_pct": 0,
            "avg_f_statistic": 0,
            "features": []
        }

    n_significant = sum(r["significant"] for r in domain_results)
    avg_f_stat = sum(r["f_statistic"] for r in domain_results) / n_features
    significant_pct = (n_significant / n_features) * 100

    # Sort features by F-statistic
    sorted_results = sorted(
        domain_results, key=lambda r: r["f_statistic"], reverse=True)

    return {
        "domain": domain_prefix,
        "name": domain_name,
        "n_features": n_features,
        "n_significant": n_significant,
        "significant_pct": significant_pct,
        "avg_f_statistic": avg_f_stat,
        "features": sorted_results
    }


def compare_domains(domain_analysis_results):
    """
    Compare significance across domains and generate insights.

    Parameters:
    - domain_analysis_results: Dictionary of domain analysis results

    Returns:
    - Dictionary with comparative insights
    """
    # Find most and least significant domains
    valid_domains = {
        k: v for k, v in domain_analysis_results.items() if v["n_features"] > 0}

    if not valid_domains:
        return {"most_significant": None, "least_significant": None, "insights": "No valid domains found"}

    most_significant = max(valid_domains.items(),
                           key=lambda x: x[1]["significant_pct"])
    least_significant = min(valid_domains.items(),
                            key=lambda x: x[1]["significant_pct"])

    # Generate insights
    insights = f"The domain '{most_significant[1]['name']}' shows the strongest differentiation " \
        f"({most_significant[1]['significant_pct']:.1f}% significant features), " \
        f"suggesting clusters are best separated by {most_significant[1]['name'].lower()} characteristics.\n\n"

    if most_significant[1]["significant_pct"] > 75:
        insights += "This is a very strong differentiation, indicating the clustering has captured " \
            f"meaningful patterns in {most_significant[1]['name']}.\n\n"

    insights += f"The domain '{least_significant[1]['name']}' shows the weakest differentiation " \
        f"({least_significant[1]['significant_pct']:.1f}% significant features)."

    return {
        "most_significant": most_significant[0],
        "least_significant": least_significant[0],
        "insights": insights
    }


def run_anova_tukey_analysis(df, domain_prefix, df_with_clusters):
    st.header("ANOVA dan Analisis Tukey's HSD")
    st.write(
        "Menganalisis perbedaan signifikan antara cluster untuk setiap fitur di setiap domain")

# Create container for ANOVA results
    anova_results = {}

# Create progress bar for analysis
    progress_bar = st.progress(0)
    progress_text = st.empty()

# Save the domain dictionary before it gets modified in loops
    domain_dict = domain_prefix.copy()

# Loop through each domain
    for i, (prefix_key, domain_name) in enumerate(domain_dict.items()):
        progress_text.text(f"Menganalisis domain {domain_name}...")

    # Get domain-specific features
        domain_features = [
            col for col in df.columns if col.startswith(prefix_key)]

    # Get cluster assignments for this domain
        cluster_column = f"{prefix_key}_cluster"

        if cluster_column not in df_with_clusters.columns:
            progress_text.text(
                f"Tidak ada kolom cluster untuk domain {domain_name}, melanjutkan...")
            continue

    # Initialize results for this domain
        domain_anova_results = []

    # Perform ANOVA for each feature in this domain
        run_anova_and_tukey(df_with_clusters, domain_features,
                            cluster_column, domain_anova_results)

    # Store results for this domain
        anova_results[prefix_key] = domain_anova_results

    # Update progress bar correctly
        progress_bar.progress(min(1.0, (i + 1) / len(domain_dict)))

# Complete progress
    progress_bar.progress(1.0)
    progress_text.text("Analisis statistik selesai!")

# Save ANOVA results to session state
    st.session_state.anova_results = anova_results
    return anova_results, domain_dict


def run_anova_and_tukey(df_with_clusters, domain_features, cluster_column, domain_anova_results):
    for feature in domain_features:
        groups = []
        for cluster in sorted(df_with_clusters[cluster_column].unique()):
            feature_values = df_with_clusters[df_with_clusters[cluster_column]
                                              == cluster][feature].values
            groups.append(feature_values)
        try:
            perform_anova_analysis(domain_anova_results, feature, groups)

        except Exception as e:
            st.warning(
                f"Error during ANOVA for feature {feature}: {str(e)}")


def perform_anova_analysis(domain_anova_results, feature, groups):
    f_stat, p_value = stats.f_oneway(*groups)

    # Check if result is significant
    is_significant = p_value < 0.05

    # Store result
    result = {
        "feature": feature,
        "f_statistic": f_stat,
        "p_value": p_value,
        "significant": is_significant,
        "groups": groups
    }

    # If significant, perform Tukey's test
    if is_significant:
        # Prepare data for Tukey's HSD
        data = []
        labels = []

        prepare_tukey_data(groups, result, data, labels)

    domain_anova_results.append(result)


def prepare_tukey_data(groups, result, data, labels):
    for cluster_idx, cluster_data in enumerate(groups):
        data.extend(cluster_data)
        labels.extend(
            [f"Cluster {cluster_idx}"] * len(cluster_data))

        # Run Tukey's test
    tukey_result = pairwise_tukeyhsd(data, labels, alpha=0.05)
    result["tukey_result"] = tukey_result
    result["tukey_data"] = data
    result["tukey_labels"] = labels


def Interactive_analysis(anova_results, domain_dict):
    st.subheader("Analisis Interaktif ANOVA dan Tukey's HSD")

# Find all domains with results
    domains_with_results = [k for k, v in anova_results.items() if v]

    if not domains_with_results:
        st.warning("Tidak ada hasil ANOVA yang tersedia untuk dianalisis")
    else:
        # Use a container to prevent refreshing issues
        analysis_container = st.container()

        with analysis_container:
            # Domain selector with multiselect
            selected_domains = st.multiselect(
                "Pilih Domain:",
                options=domains_with_results,
                format_func=lambda x: domain_dict.get(x, x),
                default=[domains_with_results[0]
                         ] if domains_with_results else [],
                key="domain_multiselect"
            )

        # If domains are selected, show the feature selector
            if selected_domains:
                domain_summary_tabs = st.tabs(
                    [domain_dict[domain] for domain in selected_domains])

                for i, domain in enumerate(selected_domains):
                    with domain_summary_tabs[i]:
                        domain_results = anova_results[domain]
                        if not domain_results:
                            st.info(
                                f"Tidak ada hasil ANOVA untuk domain {domain_dict[domain]}")
                            continue

                    # Create metrics for domain
                        significant_features, significant_pct = calculate_significance_metrics(
                            domain_results)

                    # Sort features by F-statistic
                        sorted_results = sorted(
                            domain_results, key=lambda r: r["f_statistic"], reverse=True)

                    # Create two columns for top/bottom 5
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### 5 Fitur Paling Signifikan")
                            top_5 = sorted_results[:5]
                            if top_5:
                                # Create DataFrame for display
                                top_5_df = pd.DataFrame([{
                                    "Fitur": r["feature"],
                                    "F-statistic": r["f_statistic"],
                                    "p-value": r["p_value"],
                                    "Signifikan": "Ya ✓" if r["significant"] else "Tidak ✗"
                                } for r in top_5])

                            # Display the table
                                st.dataframe(top_5_df.style.format({
                                    "F-statistic": "{:.3f}",
                                    "p-value": "{:.5f}"
                                }), use_container_width=True)

                            # Create bar chart for top 5
                                fig = px.bar(
                                    top_5_df,
                                    y="Fitur",
                                    x="F-statistic",
                                    color="Signifikan",
                                    color_discrete_map={
                                        "Ya ✓": "green", "Tidak ✗": "gray"},
                                    labels={"Fitur": "",
                                            "F-statistic": "F-statistic"},
                                    title="F-statistic untuk 5 Fitur Teratas"
                                )
                                fig.update_layout(
                                    yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Tidak cukup fitur untuk dianalisis")
                            with col2:
                                st.write(
                                    "#### 5 Fitur Paling Tidak Signifikan")
                                bottom_5 = sorted_results[-5:] if len(
                                    sorted_results) >= 5 else sorted_results
                                bottom_5.reverse()  # Show the least significant first
                                if bottom_5:
                                    # Create DataFrame for display
                                    bottom_5_df = pd.DataFrame([{
                                        "Fitur": r["feature"],
                                        "F-statistic": r["f_statistic"],
                                        "p-value": r["p_value"],
                                        "Signifikan": "Ya ✓" if r["significant"] else "Tidak ✗"
                                    } for r in bottom_5])

                                # Display the table
                                    st.dataframe(bottom_5_df.style.format({
                                        "F-statistic": "{:.3f}",
                                        "p-value": "{:.5f}"
                                    }), use_container_width=True)

                                # Create bar chart for bottom 5
                                    fig = px.bar(
                                        bottom_5_df,
                                        y="Fitur",
                                        x="F-statistic",
                                        color="Signifikan",
                                        color_discrete_map={
                                            "Ya ✓": "green", "Tidak ✗": "gray"},
                                        labels={"Fitur": "",
                                                "F-statistic": "F-statistic"},
                                        title="F-statistic untuk 5 Fitur Terbawah"
                                    )
                                    fig.update_layout(
                                        yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(
                                        fig, use_container_width=True)
                                else:
                                    st.info(
                                        "Tidak cukup fitur untuk dianalisis")
                    if significant_features > 0:
                        st.success(f"""
                        **Insight Domain {domain_dict[domain]}**: 
                        {significant_pct:.1f}% fitur menunjukkan perbedaan signifikan antar cluster.
                        Fitur dengan perbedaan paling signifikan adalah "{top_5[0]['feature']}" (F={top_5[0]['f_statistic']:.2f}).
                    """)
                    else:
                        st.warning(f"""
                        **Insight Domain {domain_dict[domain]}**: 
                        Tidak ada fitur yang menunjukkan perbedaan signifikan antar cluster.
                        Ini mengindikasikan bahwa clustering kurang efektif dalam membedakan responden berdasarkan domain ini.
                    """)

            # Gather all features from selected domains
                all_domain_features = []
                domain_feature_map = {}  # Map features to their domains

                accumulate_domain_features(
                    anova_results, domain_dict, selected_domains, all_domain_features, domain_feature_map)

            # Sort features by significance and F-statistic
                significant_features = [
                    f for f in all_domain_features if f[2]["significant"]]
                non_significant_features = [
                    f for f in all_domain_features if not f[2]["significant"]]

            # Sort each group by F-statistic
                significant_features.sort(
                    key=lambda x: x[2]["f_statistic"], reverse=True)
                non_significant_features.sort(
                    key=lambda x: x[2]["f_statistic"], reverse=True)

            # Create feature options for multiselect
                feature_options = [f[0]
                                   for f in significant_features + non_significant_features]

            # Feature multiselect
                selected_features = st.multiselect(
                    "Pilih Fitur untuk Analisis:",
                    options=feature_options,
                    default=[feature_options[0]] if feature_options else [],
                    key="feature_multiselect"
                )

            # Create tabs for comparing multiple features if more than one is selected
                if len(selected_features) > 0:
                    feature_tabs = st.tabs(selected_features)

                    for i, feature_display in enumerate(selected_features):
                        domain, feature_name, feature_result = domain_feature_map[feature_display]

                        with feature_tabs[i]:
                            # Display ANOVA results
                            st.write(f"#### Hasil ANOVA untuk {feature_name}")

                        # Use columns for metrics to avoid refresh issues
                            metric_cols = st.columns(3)
                            metric_cols[0].metric(
                                "F-statistic",
                                f"{feature_result['f_statistic']:.3f}"
                            )
                            metric_cols[1].metric(
                                "p-value",
                                f"{feature_result['p_value']:.5f}"
                            )
                            metric_cols[2].metric(
                                "Signifikan",
                                "Ya ✓" if feature_result["significant"] else "Tidak ✗"
                            )

                        # Get cluster column for this domain
                            cluster_column = f"{domain}_cluster"

                        # Create side-by-side visualizations
                            col1, col2 = st.columns(2)

                            with col1:
                                # Boxplot comparing clusters - with try/except to prevent errors
                                try:
                                    st.write("#### Distribusi per Cluster")
                                    fig = px.box(
                                        x=[f"Cluster {i}" for i, g in enumerate(
                                            feature_result['groups']) for _ in range(len(g))],
                                        y=[val for group in feature_result['groups']
                                           for val in group],
                                        points="all",
                                        title=f"Distribusi {feature_name} per Cluster",
                                        labels={"x": "Clusters",
                                                "y": feature_name}
                                    )
                                    st.plotly_chart(
                                        fig, use_container_width=True)
                                except Exception as e:
                                    st.error(
                                        f"Error creating boxplot: {str(e)}")

                            with col2:
                                # Means comparison - with try/except to prevent errors
                                try:
                                    st.write("#### Perbandingan Rata-rata")
                                    means = [np.mean(group)
                                             for group in feature_result['groups']]
                                    std_errs = [np.std(group) / np.sqrt(len(group))
                                                for group in feature_result['groups']]

                                    fig = px.bar(
                                        x=[f"Cluster {i}" for i in range(
                                            len(means))],
                                        y=means,
                                        error_y=std_errs,
                                        title=f"Rata-rata {feature_name} per Cluster",
                                        labels={"x": "Clusters",
                                                "y": f"Mean {feature_name}"}
                                    )
                                    st.plotly_chart(
                                        fig, use_container_width=True)
                                except Exception as e:
                                    st.error(
                                        f"Error creating mean comparison chart: {str(e)}")

                        # Display Tukey results if available
                            if "tukey_result" in feature_result:
                                with st.container():
                                    st.write(
                                        "#### Hasil Tukey's HSD (Perbedaan Antar Pasangan Cluster)")

                                    try:
                                        # Convert Tukey results to DataFrame
                                        tukey = feature_result["tukey_result"]
                                        tukey_df = pd.DataFrame(
                                            data=tukey._results_table.data[1:],
                                            columns=tukey._results_table.data[0]
                                        )

                                    # Display table in a way that prevents refresh
                                        st.dataframe(
                                            tukey_df, use_container_width=True)

                                    # Create visualization of pairwise differences
                                        st.write(
                                            "#### Visualisasi Perbedaan Antar Cluster")

                                    # Format for better display
                                        tukey_plot_df = tukey_df.copy()
                                        tukey_plot_df['pair'] = tukey_plot_df['group1'] + \
                                            ' vs ' + tukey_plot_df['group2']
                                        tukey_plot_df['meandiff'] = pd.to_numeric(
                                            tukey_plot_df['meandiff'])
                                        tukey_plot_df['significant'] = pd.to_numeric(
                                            tukey_plot_df['p-adj']) < 0.05

                                    # Create bar chart with different colors for significant differences
                                        fig = px.bar(
                                            tukey_plot_df,
                                            y='pair',
                                            x='meandiff',
                                            color='significant',
                                            color_discrete_map={
                                                True: 'green', False: 'gray'},
                                            labels={'pair': 'Pasangan Cluster',
                                                    'meandiff': 'Perbedaan Rata-rata'},
                                            title=f"Perbedaan Rata-rata Antar Cluster untuk {feature_name}"
                                        )

                                    # Add reference line at 0
                                        fig.add_vline(
                                            x=0, line_dash="dash", line_color="black")

                                    # Add labels for significance
                                        fig.update_traces(
                                            hovertemplate='<b>%{y}</b><br>Perbedaan: %{x:.3f}<extra></extra>',
                                        )

                                        st.plotly_chart(
                                            fig, use_container_width=True)

                                    # Add insights - wrap in try/except to prevent refresh issues
                                        try:
                                            significant_pairs = tukey_plot_df[tukey_plot_df['significant']]
                                            if not significant_pairs.empty:
                                                st.success(
                                                    f"Terdapat {len(significant_pairs)} pasangan cluster yang memiliki perbedaan signifikan.")

                                            # Find the largest difference
                                                max_diff_pair = significant_pairs.loc[significant_pairs['meandiff'].abs(
                                                ).idxmax()]
                                                st.info(
                                                    f"Perbedaan terbesar ditemukan antara {max_diff_pair['group1']} dan {max_diff_pair['group2']} "
                                                    f"dengan selisih rata-rata {abs(max_diff_pair['meandiff']):.2f}."
                                                )
                                            else:
                                                st.info(
                                                    "Tidak ada perbedaan signifikan antar pasangan cluster untuk fitur ini.")
                                        except Exception as e:
                                            st.error(
                                                f"Error analyzing results: {str(e)}")

                                    except Exception as e:
                                        st.error(
                                            f"Error processing Tukey results: {str(e)}")
                else:
                    st.info("Silakan pilih setidaknya satu fitur untuk analisis")
            else:
                st.info("Silakan pilih setidaknya satu domain untuk analisis")


def accumulate_domain_features(anova_results, domain_dict, selected_domains, all_domain_features, domain_feature_map):
    for domain in selected_domains:
        domain_results = anova_results[domain]
        for result in domain_results:
            feature_name = result["feature"]
            # Add domain prefix to feature name to prevent duplicates
            feature_display = f"{domain_dict[domain]}: {feature_name}"
            all_domain_features.append(
                (feature_display, feature_name, result))
            domain_feature_map[feature_display] = (
                domain, feature_name, result)


def calculate_significance_metrics(domain_results):
    total_features = len(domain_results)
    significant_features = sum(
        1 for r in domain_results if r["significant"])
    significant_pct = (significant_features /
                       total_features) * 100 if total_features > 0 else 0
    cols = st.columns(3)
    cols[0].metric("Total Fitur", total_features)
    cols[1].metric("Fitur Signifikan",
                   significant_features)
    cols[2].metric("Persentase", f"{significant_pct:.1f}%")
    return significant_features, significant_pct
