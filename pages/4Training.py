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

# Check if preprocessing is completed
if 'df' not in st.session_state:
    st.error("Data tidak ditemukan. Silakan lakukan preprocessing terlebih dahulu.")
    st.switch_page("pages/3Preprocessing.py")
    st.stop()

if 'scaler' not in st.session_state:
    st.error("Data belum diskalakan. Silakan lakukan preprocessing terlebih dahulu.")
    st.switch_page("pages/3Preprocessing.py")
    st.stop()
    
if 'scaled_data_original' not in st.session_state:
    st.error("Data hasil scaling tidak ditemukan. Silakan lakukan preprocessing terlebih dahulu.")
    st.switch_page("pages/3Preprocessing.py")
    st.stop()

# Get preprocessed data
df = st.session_state.df
scaled_data_original = st.session_state.scaled_data_original


use_pca = st.session_state.get('use_pca', False)

st.title("Training Model Clustering")

# Display preprocessing summary
st.subheader("Ringkasan Data Preprocessing")
summary_cols = st.columns(4)

with summary_cols[0]:
    st.metric("Jumlah Sampel", len(df))
with summary_cols[1]:
    st.metric("Jumlah Fitur", len(df.columns))
with summary_cols[2]:
    st.metric("Scaler", st.session_state.scaler_type)
with summary_cols[3]:
    if use_pca:
        st.metric("PCA", f"{st.session_state.n_components} components")
    else:
        st.metric("PCA", "Tidak digunakan")

# Display data overview
with st.expander("Data Overview", expanded=False):
    st.write("### Data Hasil Preprocessing")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("### Data hasil robust scaling")
    st.dataframe(scaled_data_original, use_container_width=True)
    
    if use_pca:
        st.write("### PCA Results Summary")
        pca_summary_cols = st.columns(4)
        domain_names = ["Financial Attitude", "Financial Behavior", "Financial Knowledge", "Materialism"]
        domain_keys = ["fa", "fb", "fk", "m"]
        
        for i, (name, key) in enumerate(zip(domain_names, domain_keys)):
            with pca_summary_cols[i]:
                # Fix: Check if key exists and has data
                if (key in st.session_state.get('cumulative_variance', {}) and 
                    st.session_state.cumulative_variance[key] is not None and
                    len(st.session_state.cumulative_variance[key]) > 0):
                    cum_var = st.session_state.cumulative_variance[key][-1]
                    st.metric(f"{name}", f"{cum_var:.1%}")

# Initialize session state for results
st.session_state.results = {
    'Silhouette Score': None,
    'Davies-Bouldin Index': None,
    'Calinski-Harabasz Index': None
}
st.session_state.clusters_number = None
st.session_state.clustered_data = None

# Algorithm Selection
st.subheader("Pemilihan Algoritma Clustering")

use_algorithm = "Hierarchical"

# Store algorithm choice
if 'current_algorithm' not in st.session_state:
    st.session_state.current_algorithm = use_algorithm
st.session_state.use_algorithm = use_algorithm

st.info(f"Menggunakan **{use_algorithm}** untuk clustering")

# Domain configuration
domain_prefix = {
    "fa": "Financial Attitude",
    "fb": "Financial Behavior", 
    "fk": "Financial Knowledge",
    "m": "Materialism"
}

# Prepare data for clustering
if use_pca:
    st.info("Menggunakan data PCA untuk clustering")
    
    # Get PCA data from session state with safe checking
    pca_data = {}
    pca_available = True
    
    for domain in ['fa', 'fb', 'fk', 'm']:
        pca_key = f'pca_result_{domain}'
        if pca_key in st.session_state and st.session_state[pca_key] is not None:
            pca_data[domain] = st.session_state[pca_key]
        else:
            st.warning(f"PCA data untuk {domain} tidak ditemukan")
            pca_available = False
    
    if not pca_available:
        st.error("PCA data tidak lengkap. Silakan lakukan preprocessing ulang.")
        st.stop()
    
    # Display PCA summary
    with st.expander("PCA Summary", expanded=False):
        pca_cols = st.columns(4)
        for i, (domain, name) in enumerate(domain_prefix.items()):
            with pca_cols[i]:
                # Fix: Safe checking for cumulative variance
                if (domain in st.session_state.get('cumulative_variance', {}) and 
                    st.session_state.cumulative_variance[domain] is not None and
                    len(st.session_state.cumulative_variance[domain]) > 0):
                    cum_var = st.session_state.cumulative_variance[domain][-1]
                    n_components = len(st.session_state.cumulative_variance[domain])
                    st.metric(f"{name}", f"{cum_var:.1%}")
                    st.caption(f"{n_components} components")
                else:
                    st.metric(f"{name}", "No data")
else:
    st.info("Menggunakan data scaled original untuk clustering")
    pca_data = None

# Clustering execution
st.subheader(f"{use_algorithm} Clustering")

with st.spinner(f"Menjalankan {use_algorithm} clustering..."):
    try:
        if use_algorithm == "Kmeans":
            if use_pca:
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
                domain_cluster = dbc.domain_clustering(
                    df,
                    domain_prefix,
                    scaled_data_original,
                    algorithm='kmeans',
                    random_state=42,
                    use_pca=False
                )
        
        elif use_algorithm == "Hierarchical":
            if use_pca:
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
                domain_cluster = dbc.domain_clustering(
                    df,
                    domain_prefix,
                    scaled_data_original,
                    algorithm='hierarchical',
                    random_state=42,
                    use_pca=False
                )
    except Exception as e:
        st.error(f"Error dalam clustering: {str(e)}")
        st.stop()

# Display clustering results
st.subheader("Hasil Clustering")

# Plot domain clusters
try:
    dbc.plot_domain_clusters(domain_cluster)
except Exception as e:
    st.error(f"Error dalam plotting: {str(e)}")

# Create comparison table
try:
    comparison_df = dbc.create_comparison_table(domain_cluster)
    st.dataframe(comparison_df, use_container_width=True)
except Exception as e:
    st.error(f"Error dalam membuat comparison table: {str(e)}")

# Detailed analysis
with st.spinner("Melakukan analisis detail..."):
    try:
        df_with_clusters, cross_tabs, crosstab_figs, pattern_counts = dbc.analyze_domain_clusters(
            df, domain_cluster
        )
    except Exception as e:
        st.error(f"Error dalam analisis detail: {str(e)}")
        st.stop()

# Store results in session state
st.session_state.clustered_data = df_with_clusters
st.session_state.domain_cluster_results = domain_cluster
st.session_state.cluster_cross_tabs = cross_tabs
st.session_state.cluster_pattern_counts = pattern_counts
st.session_state.crosstab_figs = crosstab_figs

# Save algorithm-specific metrics
algorithm_metrics = {
    "algorithm": use_algorithm,
    "comparison_metrics": comparison_df.to_dict(),
    "use_pca": use_pca,
    "scaler": st.session_state.scaler_type,
    "n_components": st.session_state.get('n_components', None) if use_pca else None
}
st.session_state.algorithm_metrics = algorithm_metrics

# Cluster Analysis and Visualization
st.subheader("Analisis dan Visualisasi Cluster")

# Summary statistics
st.write("### Ringkasan Cluster per Domain")
cluster_summary_data = []

for domain in domain_prefix:
    domain_cluster_col = next((col for col in df_with_clusters.columns
                              if col.startswith(domain) and 'cluster' in col.lower()), None)
    
    if domain_cluster_col:
        domain_features = [col for col in df.columns if col.startswith(domain)]
        if len(domain_features) > 0:  # Fix: Check if domain has features
            unique_clusters = sorted(df_with_clusters[domain_cluster_col].unique())
            
            for cluster_id in unique_clusters:
                cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col] == cluster_id]
                if len(cluster_data) > 0:  # Fix: Check if cluster has data
                    mean_values = cluster_data[domain_features].mean()
                    if not mean_values.empty:  # Fix: Check if mean_values is not empty
                        mean_value = mean_values.mean()
                        cluster_size = len(cluster_data)
                        percentage = (cluster_size / len(df_with_clusters)) * 100
                        
                        cluster_summary_data.append({
                            'Domain': domain_prefix[domain],
                            'Cluster': f"Cluster {cluster_id}",
                            'Mean Value': round(mean_value, 2),
                            'Sample Count': cluster_size,
                            'Percentage': f"{percentage:.1f}%"
                        })

# Display summary table
if len(cluster_summary_data) > 0:
    cluster_summary_df = pd.DataFrame(cluster_summary_data)
    st.dataframe(
        cluster_summary_df,
        use_container_width=True,
        column_config={
            "Mean Value": st.column_config.NumberColumn(format="%.2f"),
            "Percentage": st.column_config.TextColumn("% dari Total")
        }
    )

    # Heatmap visualization
    try:
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
    except Exception as e:
        st.warning(f"Tidak dapat membuat heatmap: {str(e)}")
else:
    st.warning("Tidak ada data cluster yang valid untuk ditampilkan.")

# Detailed domain analysis
st.subheader("Visualisasi Statistik per Cluster per Domain")

domain_tabs = st.tabs([domain_prefix[domain] for domain in domain_prefix])

# Store cluster statistics
cluster_stats = {
    'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'algorithm': use_algorithm,
    'use_pca': use_pca,
    'scaler': st.session_state.scaler_type,
    'domain_tabs': [domain_prefix[domain] for domain in domain_prefix],
    'feature_count': {domain: len([col for col in df.columns if col.startswith(domain)]) for domain in domain_prefix}
}

for domain_idx, domain in enumerate(domain_prefix):
    with domain_tabs[domain_idx]:
        domain_cluster_col = next((col for col in df_with_clusters.columns
                                  if col.startswith(domain) and 'cluster' in col.lower()), None)
        
        if domain_cluster_col:
            unique_clusters = sorted(df_with_clusters[domain_cluster_col].unique())
            domain_features = [col for col in df.columns if col.startswith(domain)]
            
            if len(domain_features) == 0:
                st.warning(f"No features found for domain {domain}")
                continue
                
            st.write(f"### {domain_prefix[domain]} - {len(unique_clusters)} Clusters")
            
            # Visualization options
            stat_view = st.selectbox(
                f"Pilih visualisasi untuk {domain_prefix[domain]}",
                ["Radar Chart (Semua Cluster)", "Perbandingan Mean", "Perbandingan Std Dev",
                 "Box Plot", "Violin Plot"],
                key=f"stat_view_{domain}"
            )
            
            # Prepare data for visualization
            cluster_means = {}
            cluster_stds = {}
            
            for cluster_id in unique_clusters:
                cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col] == cluster_id]
                if len(cluster_data) > 0:
                    cluster_means[cluster_id] = cluster_data[domain_features].mean()
                    cluster_stds[cluster_id] = cluster_data[domain_features].std()
            
            # Create visualizations based on selection
            try:
                if stat_view == "Radar Chart (Semua Cluster)":
                    radar_data = []
                    for cluster_id in unique_clusters:
                        if cluster_id in cluster_means:
                            for feature, value in cluster_means[cluster_id].items():
                                if not pd.isna(value):  # Fix: Check for NaN values
                                    radar_data.append({
                                        'Cluster': f"Cluster {cluster_id}",
                                        'Feature': feature,
                                        'Value': value
                                    })
                    
                    if len(radar_data) > 0:
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
                    else:
                        st.warning("No valid data for radar chart")
                
                elif stat_view == "Perbandingan Mean":
                    mean_data = []
                    for cluster_id in unique_clusters:
                        if cluster_id in cluster_means:
                            for feature, value in cluster_means[cluster_id].items():
                                if not pd.isna(value):  # Fix: Check for NaN values
                                    mean_data.append({
                                        'Cluster': f"Cluster {cluster_id}",
                                        'Feature': feature,
                                        'Mean': value
                                    })
                    
                    if len(mean_data) > 0:
                        mean_df = pd.DataFrame(mean_data)
                        
                        fig = px.bar(
                            mean_df,
                            x="Feature",
                            y="Mean",
                            color="Cluster",
                            barmode="group",
                            title=f"Mean Comparison - {domain_prefix[domain]} Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Heatmap
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
                    else:
                        st.warning("No valid data for mean comparison")
                
                elif stat_view == "Perbandingan Std Dev":
                    std_data = []
                    for cluster_id in unique_clusters:
                        if cluster_id in cluster_stds:
                            for feature, value in cluster_stds[cluster_id].items():
                                if not pd.isna(value):  # Fix: Check for NaN values
                                    std_data.append({
                                        'Cluster': f"Cluster {cluster_id}",
                                        'Feature': feature,
                                        'Std Dev': value
                                    })
                    
                    if len(std_data) > 0:
                        std_df = pd.DataFrame(std_data)
                        
                        fig = px.bar(
                            std_df,
                            x="Feature",
                            y="Std Dev",
                            color="Cluster",
                            barmode="group",
                            title=f"Standard Deviation Comparison - {domain_prefix[domain]} Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Heatmap
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
                    else:
                        st.warning("No valid data for std dev comparison")
                
                elif stat_view == "Box Plot":
                    if len(domain_features) > 0:
                        selected_feature = st.selectbox(
                            f"Pilih fitur {domain_prefix[domain]} untuk box plot",
                            domain_features,
                            key=f"boxplot_feature_{domain}"
                        )
                        
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
                        
                        st.write("### Summary Statistics")
                        stats_df = df_with_clusters.groupby(domain_cluster_col)[
                            selected_feature].describe()
                        st.dataframe(stats_df, use_container_width=True)
                
                elif stat_view == "Violin Plot":
                    if len(domain_features) > 0:
                        selected_feature = st.selectbox(
                            f"Pilih fitur {domain_prefix[domain]} untuk violin plot",
                            domain_features,
                            key=f"violinplot_feature_{domain}"
                        )
                        
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
                        
                        st.write("### Distribution Statistics")
                        skew_data = []
                        
                        for cluster_id in unique_clusters:
                            cluster_values = df_with_clusters[df_with_clusters[domain_cluster_col]
                                                              == cluster_id][selected_feature]
                            if len(cluster_values) > 0:
                                skew_data.append({
                                    'Cluster': cluster_id,
                                    'Count': len(cluster_values),
                                    'Mean': cluster_values.mean(),
                                    'Median': cluster_values.median(),
                                    'Skewness': cluster_values.skew() if len(cluster_values) > 1 else 0,
                                    'Kurtosis': cluster_values.kurtosis() if len(cluster_values) > 1 else 0
                                })
                        
                        if len(skew_data) > 0:
                            skew_df = pd.DataFrame(skew_data)
                            st.dataframe(skew_df, use_container_width=True)
                
                # Show summary statistics for all features
                with st.expander("Show Detailed Statistics for All Features"):
                    if len(unique_clusters) > 0:
                        cluster_select = st.selectbox(
                            "Select Cluster",
                            unique_clusters,
                            key=f"cluster_select_{domain}"
                        )
                        
                        cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col]
                                                        == cluster_select]
                        if len(cluster_data) > 0 and len(domain_features) > 0:
                            stats_df = cluster_data[domain_features].describe().T
                            # Fix: Check for division by zero
                            valid_mean_mask = stats_df['mean'] != 0
                            stats_df['CV (%)'] = 0
                            stats_df.loc[valid_mean_mask, 'CV (%)'] = (
                                stats_df.loc[valid_mean_mask, 'std'] / 
                                stats_df.loc[valid_mean_mask, 'mean'] * 100
                            ).round(2)
                            st.dataframe(stats_df, use_container_width=True)
                        else:
                            st.warning("No data available for selected cluster")
                    else:
                        st.warning("No clusters available")
                        
            except Exception as e:
                st.error(f"Error dalam visualisasi {stat_view}: {str(e)}")
                
        else:
            st.warning(f"No cluster column found for {domain_prefix[domain]}")

# Store domain-specific cluster means and statistics
for domain in domain_prefix:
    domain_cluster_col = next((col for col in df_with_clusters.columns
                               if col.startswith(domain) and 'cluster' in col.lower()), None)

    if domain_cluster_col:
        unique_clusters = sorted(df_with_clusters[domain_cluster_col].unique())
        domain_features = [col for col in df.columns if col.startswith(domain)]

        if len(domain_features) > 0:  # Fix: Check if domain has features
            # Store cluster means with standard naming convention
            for cluster_id in unique_clusters:
                cluster_data = df_with_clusters[df_with_clusters[domain_cluster_col] == cluster_id]

                if len(cluster_data) > 0:  # Fix: Check if cluster has data
                    # Calculate statistics
                    mean_values = cluster_data[domain_features].mean()
                    std_values = cluster_data[domain_features].std()
                    
                    # Fix: Check if mean_values is not empty
                    if not mean_values.empty:
                        domain_avg = mean_values.mean()
                        
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

st.session_state.cluster_statistics = cluster_stats

st.info("Langkah selanjutnya: Lanjut ke halaman **Post-Hoc Analysis** untuk validasi statistik clustering.")