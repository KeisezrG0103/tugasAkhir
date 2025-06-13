from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import os
import numpy as np
import utils.data_preprocessing as dp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
import utils.Kmeans as kmeans
import utils.Hierarchical_clustering as hc
import utils.domain_based_clustering as dbc
st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

is_multicollinearity = False

if 'df' not in st.session_state:
    st.error("Tolong menuju ke halaman Preprocessing terlebih dahulu")
    st.switch_page("pages/3Preprocessing.py")
else:
    df = st.session_state.df


st.session_state.results = {
    'Silhouette Score': None,
    'Davies-Bouldin Index': None,
    'Calinski-Harabasz Index': None
}
st.session_state.clusters_number = None
st.session_state.clustered_data = None

if 'multicollinearity' in st.session_state and len(st.session_state.multicollinearity) > 0:
    st.info("Terdapat multikolinearitas pada data")
    is_multicollinearity = True


st.title("Training")
st.write("Data Hasil Preprocessing")
st.dataframe(df, use_container_width=True)

# Section for PCA configuration
st.subheader("Konfigurasi PCA")

print(len(df))

# milih scaler
scaler = st.selectbox("Pilih Scaler",
                      options=["StandardScaler",
                               "MinMaxScaler", "RobustScaler"],
                      index=0,
                      help="Pilih scaler untuk menormalkan data")
# Store the choice in session state only if it's different
if 'scaler' not in st.session_state or st.session_state.scaler != scaler:
    st.session_state.scaler = scaler
    st.info(f"menggunakan {scaler} untuk menormalkan data")


# Checkbox for PCA usage
use_pca = st.checkbox("Gunakan PCA",
                      value=is_multicollinearity,
                      help="Gunakan PCA untuk mengurangi dimensi data")

# Initialize session state for PCA settings
if 'use_pca' not in st.session_state:
    st.session_state.use_pca = is_multicollinearity

st.session_state.use_pca = use_pca

if use_pca:

    n_components = st.slider("Number of Components",
                             min_value=2,
                             max_value=min(len(df.columns), 10),
                             value=min(3, len(df.columns)),
                             help="Number of principal components to keep")

    scaled_data_original, pca_result_fa, pca_result_fb, pca_result_fk, pca_result_m, pca_result_fa_explained, pca_result_fb_explained, pca_result_fk_explained, pca_result_m_explained, cumulative_variance, \
        cumulative_variance_fa, cumulative_variance_fb, cumulative_variance_fk, cumulative_variance_m, explained_variance = dp.perform_pca(
            df, scaler, n_components)

    st.session_state.scaled_data = scaled_data_original
    scaled_data_fa = pca_result_fa
    scaled_data_fb = pca_result_fb
    scaled_data_fk = pca_result_fk
    scaled_data_m = pca_result_m

    with st.expander("Hasil PCA per Domain", expanded=True):
        # Summary metrics
        domain_names = ["Financial Attitude", "Financial Behavior",
                        "Financial Knowledge", "Materialism"]
        domain_keys = ["fa", "fb", "fk", "m"]

        # Create 4 columns for summary metrics
        cols = st.columns(4)
        for i, (name, key) in enumerate(zip(domain_names, domain_keys)):
            with cols[i]:
                # Get the appropriate cumulative variance
                cum_var = None
                if key == "fa":
                    cum_var = cumulative_variance_fa[-1]
                elif key == "fb":
                    cum_var = cumulative_variance_fb[-1]
                elif key == "fk":
                    cum_var = cumulative_variance_fk[-1]
                elif key == "m":
                    cum_var = cumulative_variance_m[-1]

                st.metric(f"{name}", f"{cum_var:.2%} explained")

        # Create tabs for each domain
        domain_tabs = st.tabs(domain_names)

        # Financial Attitude
        with domain_tabs[0]:
            # Chart of explained variance for FA
            fig_fa = px.bar(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_fa))],
                y=pca_result_fa_explained if 'pca_result_fa_explained' in locals(
                ) else explained_variance[:len(cumulative_variance_fa)],
                labels={"x": "Principal Component", "y": "Explained Variance"},
                title="Explained Variance by Component - Financial Attitude"
            )
            fig_fa.add_scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_fa))],
                y=cumulative_variance_fa,
                mode="lines+markers",
                name="Cumulative Variance"
            )
            st.plotly_chart(fig_fa, use_container_width=True)

            # PCA data preview for FA
            st.write("### Financial Attitude PCA Preview")
            pca_df_fa = pd.DataFrame(
                pca_result_fa,
                columns=[
                    f"FA_PC{i+1}" for i in range(min(n_components, pca_result_fa.shape[1]))]
            )
            st.dataframe(pca_df_fa.head(), use_container_width=True)

        # Financial Behavior
        with domain_tabs[1]:
            # Chart of explained variance for FB
            fig_fb = px.bar(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_fb))],
                y=pca_result_fb_explained if 'pca_result_fb_explained' in locals(
                ) else explained_variance[:len(cumulative_variance_fb)],
                labels={"x": "Principal Component", "y": "Explained Variance"},
                title="Explained Variance by Component - Financial Behavior"
            )
            fig_fb.add_scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_fb))],
                y=cumulative_variance_fb,
                mode="lines+markers",
                name="Cumulative Variance"
            )
            st.plotly_chart(fig_fb, use_container_width=True)

            # PCA data preview for FB
            st.write("### Financial Behavior PCA Preview")
            pca_df_fb = pd.DataFrame(
                pca_result_fb,
                columns=[
                    f"FB_PC{i+1}" for i in range(min(n_components, pca_result_fb.shape[1]))]
            )
            st.dataframe(pca_df_fb.head(), use_container_width=True)

        # Financial Knowledge
        with domain_tabs[2]:
            fig_fk = px.bar(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_fk))],
                y=pca_result_fk_explained if 'pca_result_fk_explained' in locals(
                ) else explained_variance[:len(cumulative_variance_fk)],
                labels={"x": "Principal Component", "y": "Explained Variance"},
                title="Explained Variance by Component - Financial Knowledge"
            )
            fig_fk.add_scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_fk))],
                y=cumulative_variance_fk,
                mode="lines+markers",
                name="Cumulative Variance"
            )
            st.plotly_chart(fig_fk, use_container_width=True)

            # PCA data preview for FK
            st.write("### Financial Knowledge PCA Preview")
            pca_df_fk = pd.DataFrame(
                pca_result_fk,
                columns=[
                    f"FK_PC{i+1}" for i in range(min(n_components, pca_result_fk.shape[1]))]
            )
            st.dataframe(pca_df_fk.head(), use_container_width=True)

        with domain_tabs[3]:
            fig_m = px.bar(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_m))],
                y=pca_result_m_explained if 'pca_result_m_explained' in locals(
                ) else explained_variance[:len(cumulative_variance_m)],
                labels={"x": "Principal Component", "y": "Explained Variance"},
                title="Explained Variance by Component - Materialism"
            )
            fig_m.add_scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance_m))],
                y=cumulative_variance_m,
                mode="lines+markers",
                name="Cumulative Variance"
            )
            st.plotly_chart(fig_m, use_container_width=True)

            st.write("### Materialism PCA Preview")
            pca_df_m = pd.DataFrame(
                pca_result_m,
                columns=[
                    f"M_PC{i+1}" for i in range(min(n_components, pca_result_m.shape[1]))]
            )
            st.dataframe(pca_df_m.head(), use_container_width=True)

    with st.expander("View Combined PCA Results"):
        all_pca_cols = []
        all_pca_data = []

        if pca_result_fa.shape[1] > 0:
            fa_cols = [
                f"FA_PC{i+1}" for i in range(min(n_components, pca_result_fa.shape[1]))]
            all_pca_cols.extend(fa_cols)
            all_pca_data.append(pd.DataFrame(
                pca_result_fa, columns=fa_cols))

        if pca_result_fb.shape[1] > 0:
            fb_cols = [
                f"FB_PC{i+1}" for i in range(min(n_components, pca_result_fb.shape[1]))]
            all_pca_cols.extend(fb_cols)
            all_pca_data.append(pd.DataFrame(
                pca_result_fb, columns=fb_cols))

        if pca_result_fk.shape[1] > 0:
            fk_cols = [
                f"FK_PC{i+1}" for i in range(min(n_components, pca_result_fk.shape[1]))]
            all_pca_cols.extend(fk_cols)
            all_pca_data.append(pd.DataFrame(
                pca_result_fk, columns=fk_cols))

        if pca_result_m.shape[1] > 0:
            m_cols = [
                f"M_PC{i+1}" for i in range(min(n_components, pca_result_m.shape[1]))]
            all_pca_cols.extend(m_cols)
            all_pca_data.append(pd.DataFrame(pca_result_m, columns=m_cols))

        # Combine all PCA results
        if all_pca_data:
            combined_pca = pd.concat(all_pca_data, axis=1)
            st.write("### Combined PCA Results")
            st.dataframe(combined_pca.head(), use_container_width=True)
else:
    if scaler == "StandardScaler":
        scaler = StandardScaler()
        st.session_state.scaler = "StandardScaler"
    elif scaler == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        st.session_state.scaler = "MinMaxScaler"
    elif scaler == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        st.session_state.scaler = "RobustScaler"
    scaled_data = scaler.fit_transform(df)

    st.write("### hasil normalisasi data")
    with st.expander("Hasil Normalisasi Data", expanded=False):
        st.dataframe(scaled_data, use_container_width=True)


use_algorithm = st.selectbox("Pilih Algoritma",
                             options=["Kmeans",
                                      "Hierarchical",
                                      ],
                             index=0,
                             help="Pilih algoritma yang akan digunakan untuk pelatihan model")


if 'current_algorithm' not in st.session_state:
    st.session_state.current_algorithm = use_algorithm
st.session_state.use_algorithm = use_algorithm

st.info(f"menggunakan {use_algorithm} untuk pelatihan model")

domain_prefix = {
    "fa": "Financial Attitude",
    "fb": "Financial Behavior",
    "fk": "Financial Knowledge",
    "m": "Materialism"
}


if use_algorithm == "Kmeans":
    st.subheader("K-Means")

    if use_pca:
        st.info("Running K-Means on PCA components for each domain")

        pca_data = {
            'fa': pca_result_fa,
            'fb': pca_result_fb,
            'fk': pca_result_fk,
            'm': pca_result_m
        }

        domain_cluster = dbc.domain_clustering(
            df,
            domain_prefix,
            scaled_data_original,
            algorithm='kmeans',
            random_state=42,
            use_pca=True,
            pca_data=pca_data
        )
    else:
        st.info("Running K-Means on scaled original features")
        domain_cluster = dbc.domain_clustering(
            df,
            domain_prefix,
            scaled_data,
            algorithm='kmeans',
            random_state=42,
            use_pca=False
        )

    dbc.plot_domain_clusters(domain_cluster)

    comparison_df = dbc.create_comparison_table(domain_cluster)
    st.dataframe(comparison_df)

    df_with_clusters, cross_tabs, crosstab_figs, pattern_counts = dbc.analyze_domain_clusters(
        df, domain_cluster
    )


if use_algorithm == "Hierarchical":
    st.subheader("Hierarchical Clustering")

    if use_pca:
        st.info("Running Hierarchical Clustering on PCA components for each domain")

        pca_data = {
            'fa': pca_result_fa,
            'fb': pca_result_fb,
            'fk': pca_result_fk,
            'm': pca_result_m
        }

        domain_cluster = dbc.domain_clustering(
            df,
            domain_prefix,
            scaled_data_original,
            algorithm='hierarchical',
            random_state=42,
            use_pca=True,
            pca_data=pca_data
        )
    else:
        st.info("Running Hierarchical Clustering on scaled original features")
        domain_cluster = dbc.domain_clustering(
            df,
            domain_prefix,
            scaled_data,
            algorithm='hierarchical',
            random_state=42,
            use_pca=False
        )

    dbc.plot_domain_clusters(domain_cluster)

    comparison_df = dbc.create_comparison_table(domain_cluster)
    st.dataframe(comparison_df)

    # For more detailed analysis
    df_with_clusters, cross_tabs, crosstab_figs, pattern_counts = dbc.analyze_domain_clusters(
        df, domain_cluster
    )


st.session_state.clustered_data = df_with_clusters
st.session_state.domain_cluster_results = domain_cluster
st.session_state.cluster_cross_tabs = cross_tabs
st.session_state.cluster_pattern_counts = pattern_counts

# Save algorithm-specific metrics
algorithm_metrics = {
    "algorithm": use_algorithm,
    "comparison_metrics": comparison_df.to_dict(),
    "use_pca": use_pca,
    "scaler": scaler
}
st.session_state.algorithm_metrics = algorithm_metrics

st.session_state.crosstab_figs = crosstab_figs

# Add this after storing pattern_counts in session_state

st.subheader("Visualisasi Statistik per Cluster per Domain")

# Tambahkan kode ini setelah baris 440, sebelum bagian domain_tabs

# Ringkasan cluster per domain
st.write("### Ringkasan Mean Per-Cluster Per-Domain")
st.write("Tabel berikut menampilkan rata-rata nilai untuk setiap cluster dalam setiap domain.")

# Membuat dataframe untuk ringkasan
cluster_summary_data = []

# Mengumpulkan data cluster dari semua domain
for domain in domain_prefix:
    domain_cluster_col = next((col for col in df_with_clusters.columns
                              if col.startswith(domain) and 'cluster' in col.lower()), None)

    if domain_cluster_col:
        domain_features = [col for col in df.columns if col.startswith(domain)]
        unique_clusters = sorted(df_with_clusters[domain_cluster_col].unique())

        for cluster_id in unique_clusters:
            cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col] == cluster_id]
            mean_value = cluster_data[domain_features].mean().mean()
            cluster_size = len(cluster_data)
            percentage = (cluster_size / len(df_with_clusters)) * 100

            cluster_summary_data.append({
                'Domain': domain_prefix[domain],
                'Cluster': f"Cluster {cluster_id}",
                'Mean Value': round(mean_value, 2),
                'Sample Count': cluster_size,
                'Percentage': f"{percentage:.1f}%"
            })

# Membuat dan menampilkan dataframe ringkasan
cluster_summary_df = pd.DataFrame(cluster_summary_data)

# Menampilkan dalam bentuk tabel interaktif
st.dataframe(
    cluster_summary_df,
    use_container_width=True,
    column_config={
        "Mean Value": st.column_config.NumberColumn(format="%.2f"),
        "Percentage": st.column_config.TextColumn("% dari Total")
    }
)

# Menampilkan dalam bentuk heatmap untuk perbandingan visual
pivot_summary = cluster_summary_df.pivot(
    index='Domain', columns='Cluster', values='Mean Value')
fig = px.imshow(
    pivot_summary,
    text_auto=".2f",
    labels=dict(x="Cluster", y="Domain", color="Mean Value"),
    color_continuous_scale="viridis",
    title="Heatmap Rata-rata Per-Cluster Per-Domain"
)
st.plotly_chart(fig, use_container_width=True)

# Create sections to display cluster statistics by domain
domain_tabs = st.tabs([domain_prefix[domain] for domain in domain_prefix])

# Prepare statistical data for each domain
for domain_idx, domain in enumerate(domain_prefix):
    with domain_tabs[domain_idx]:
        # Get the cluster column for this domain
        domain_cluster_col = next((col for col in df_with_clusters.columns
                                  if col.startswith(domain) and 'cluster' in col.lower()), None)

        if domain_cluster_col:
            # Get unique clusters for this domain
            unique_clusters = sorted(
                df_with_clusters[domain_cluster_col].unique())

            st.write(
                f"### {domain_prefix[domain]} memiliki {len(unique_clusters)} cluster")

            # Let user select which statistical measure to view
            stat_view = st.selectbox(
                f"Pilih visualisasi untuk {domain_prefix[domain]}",
                ["Radar Chart (Semua Cluster)", "Perbandingan Mean", "Perbandingan Std Dev",
                 "Box Plot", "Violin Plot"],
                key=f"stat_view_{domain}"
            )

            # Get all features for this domain
            domain_features = [
                col for col in df.columns if col.startswith(domain)]

            # Prepare data for visualization
            cluster_means = {}
            cluster_stds = {}

            for cluster_id in unique_clusters:
                cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col] == cluster_id]
                cluster_means[cluster_id] = cluster_data[domain_features].mean()
                cluster_stds[cluster_id] = cluster_data[domain_features].std()

            # Create visualizations based on selection
            if stat_view == "Radar Chart (Semua Cluster)":
                # Create radar chart data
                radar_data = []
                for cluster_id in unique_clusters:
                    for feature, value in cluster_means[cluster_id].items():
                        radar_data.append({
                            'Cluster': f"Cluster {cluster_id}",
                            'Feature': feature,
                            'Value': value
                        })

                radar_df = pd.DataFrame(radar_data)

                fig = px.line_polar(
                    radar_df,
                    r="Value",
                    theta="Feature",
                    color="Cluster",
                    line_close=True,
                    range_r=[radar_df['Value'].min()-0.5,
                             radar_df['Value'].max()+0.5],
                    title=f"Radar Chart - {domain_prefix[domain]} Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif stat_view == "Perbandingan Mean":
                # Prepare mean comparison data
                mean_data = []
                for cluster_id in unique_clusters:
                    for feature, value in cluster_means[cluster_id].items():
                        mean_data.append({
                            'Cluster': f"Cluster {cluster_id}",
                            'Feature': feature,
                            'Mean': value
                        })

                mean_df = pd.DataFrame(mean_data)

                # Create grouped bar chart
                fig = px.bar(
                    mean_df,
                    x="Feature",
                    y="Mean",
                    color="Cluster",
                    barmode="group",
                    title=f"Mean Comparison - {domain_prefix[domain]} Features"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Create a heatmap of means for easy comparison
                pivot_df = mean_df.pivot(
                    index="Feature", columns="Cluster", values="Mean")
                fig_heatmap = px.imshow(
                    pivot_df,
                    labels=dict(x="Cluster", y="Feature", color="Mean Value"),
                    title=f"Mean Heatmap - {domain_prefix[domain]}",
                    color_continuous_scale="viridis",
                    text_auto=True
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            elif stat_view == "Perbandingan Std Dev":
                # Prepare std comparison data
                std_data = []
                for cluster_id in unique_clusters:
                    for feature, value in cluster_stds[cluster_id].items():
                        std_data.append({
                            'Cluster': f"Cluster {cluster_id}",
                            'Feature': feature,
                            'Std Dev': value
                        })

                std_df = pd.DataFrame(std_data)

                # Create grouped bar chart
                fig = px.bar(
                    std_df,
                    x="Feature",
                    y="Std Dev",
                    color="Cluster",
                    barmode="group",
                    title=f"Standard Deviation Comparison - {domain_prefix[domain]} Features"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Create a heatmap of std devs
                pivot_df = std_df.pivot(
                    index="Feature", columns="Cluster", values="Std Dev")
                fig_heatmap = px.imshow(
                    pivot_df,
                    labels=dict(x="Cluster", y="Feature", color="Std Dev"),
                    title=f"Standard Deviation Heatmap - {domain_prefix[domain]}",
                    color_continuous_scale="viridis",
                    text_auto=True
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            elif stat_view == "Box Plot":
                # Let user select which feature to examine
                selected_feature = st.selectbox(
                    f"Pilih fitur {domain_prefix[domain]} untuk box plot",
                    domain_features,
                    key=f"boxplot_feature_{domain}"
                )

                # Create boxplot
                fig = px.box(
                    df_with_clusters,
                    x=domain_cluster_col,
                    y=selected_feature,
                    title=f"Box Plot: {selected_feature} by Cluster",
                    labels={
                        domain_cluster_col: "Cluster",
                        selected_feature: "Value"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show summary statistics
                st.write("### Summary Statistics")
                stats_df = df_with_clusters.groupby(domain_cluster_col)[
                    selected_feature].describe()
                st.dataframe(stats_df, use_container_width=True)

            elif stat_view == "Violin Plot":
                # Let user select which feature to examine
                selected_feature = st.selectbox(
                    f"Pilih fitur {domain_prefix[domain]} untuk violin plot",
                    domain_features,
                    key=f"violinplot_feature_{domain}"
                )

                # Create violin plot
                fig = px.violin(
                    df_with_clusters,
                    x=domain_cluster_col,
                    y=selected_feature,
                    box=True,
                    points="all",
                    title=f"Violin Plot: {selected_feature} by Cluster",
                    labels={
                        domain_cluster_col: "Cluster",
                        selected_feature: "Value"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show distribution statistics
                st.write("### Distribution Statistics")
                skew_df = pd.DataFrame({
                    'Cluster': [],
                    'Count': [],
                    'Mean': [],
                    'Median': [],
                    'Skewness': [],
                    'Kurtosis': []
                })

                for cluster_id in unique_clusters:
                    cluster_values = df_with_clusters[df_with_clusters[domain_cluster_col]
                                                      == cluster_id][selected_feature]
                    skew_df = pd.concat([skew_df, pd.DataFrame({
                        'Cluster': [cluster_id],
                        'Count': [len(cluster_values)],
                        'Mean': [cluster_values.mean()],
                        'Median': [cluster_values.median()],
                        'Skewness': [cluster_values.skew()],
                        'Kurtosis': [cluster_values.kurtosis()]
                    })], ignore_index=True)

                st.dataframe(skew_df, use_container_width=True)

            # Show summary statistics for all features in this domain
            with st.expander("Show Detailed Statistics for All Features"):
                cluster_select = st.selectbox(
                    "Select Cluster",
                    unique_clusters,
                    key=f"cluster_select_{domain}"
                )

                cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col]
                                                == cluster_select]
                stats_df = cluster_data[domain_features].describe().T
                stats_df['CV (%)'] = (stats_df['std'] /
                                      stats_df['mean'] * 100).round(2)
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning(f"No cluster column found for {domain_prefix[domain]}")

# Save statistics to session state
cluster_stats = {
    'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'domain_tabs': [domain_prefix[domain] for domain in domain_prefix],
    'feature_count': {domain: len([col for col in df.columns if col.startswith(domain)]) for domain in domain_prefix},
    # Add domain-specific statistics here
}

# Store domain-specific cluster means and statistics
for domain in domain_prefix:
    domain_cluster_col = next((col for col in df_with_clusters.columns
                               if col.startswith(domain) and 'cluster' in col.lower()), None)

    if domain_cluster_col:
        unique_clusters = sorted(df_with_clusters[domain_cluster_col].unique())
        domain_features = [col for col in df.columns if col.startswith(domain)]

        # Store cluster means with standard naming convention
        for cluster_id in unique_clusters:
            cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col] == cluster_id]

            # Calculate statistics
            mean_values = cluster_data[domain_features].mean()
            std_values = cluster_data[domain_features].std()
            domain_avg = mean_values.mean() if not mean_values.empty else 0

            # Store with consistent keys that AI Analysis can find
            key = f"{domain}_cluster_{cluster_id}_mean"
            cluster_stats[key] = domain_avg

            # Store detailed statistics
            cluster_stats[f"{domain}_cluster_{cluster_id}_stats"] = {
                "size": len(cluster_data),
                "mean": domain_avg,
                "feature_means": mean_values.to_dict(),
                "feature_stds": std_values.to_dict(),
                "percentage": f"{len(cluster_data) / len(df_with_clusters) * 100:.1f}%"
            }

# Save enhanced statistics to session state
st.session_state.cluster_statistics = cluster_stats
