import streamlit as st
import os
import numpy as np
import utils.data_preprocessing as dp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import utils.Gaussian_Mixture as gmm
import utils.Kmeans as kmeans
import utils.Hierarchical_clustering as hc
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

# Store the choice in session state
st.session_state.use_pca = use_pca

# If PCA is selected, show additional options
if use_pca:

    # Create n_components slider
    n_components = st.slider("Number of Components",
                             min_value=2,
                             max_value=min(len(df.columns), 10),
                             value=min(3, len(df.columns)),
                             help="Number of principal components to keep")

    # Perform PCA
    scaled_data_original, pca_result, explained_variance, cumulative_variance = dp.perform_pca(
        df, scaler, n_components)

    # Store original scaled data for reference
    st.session_state.scaled_data = scaled_data_original
    # Store PCA result as the main data for clustering
    scaled_data = pca_result

    # Display PCA results in expander
    with st.expander("Hasil PCA", expanded=True):
        # Metrics for explained variance
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Explained Variance",
                      f"{cumulative_variance[-1]:.2%}")
        with col2:
            st.metric("Number of Components", n_components)

        # Chart of explained variance
        import plotly.express as px
        fig = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance,
            labels={"x": "Principal Component", "y": "Explained Variance"},
            title="Explained Variance by Component"
        )
        fig.add_scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
            y=cumulative_variance,
            mode="lines+markers",
            name="Cumulative Variance"
        )
        st.plotly_chart(fig, use_container_width=True)

        # PCA data preview
        st.write("### PCA Result Preview")
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        st.dataframe(pca_df.head(), use_container_width=True)
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
        # Apply scaling
    # Scale the data
    scaled_data = scaler.fit_transform(df)

    st.write("### hasil normalisasi data")
    with st.expander("Hasil Normalisasi Data", expanded=False):
        st.dataframe(scaled_data, use_container_width=True)


use_algorithm = st.selectbox("Pilih Algoritma",
                             options=["Kmeans",
                                      "Hierarchical",
                                      "Gaussian Mixture Model"],
                             index=0,
                             help="Pilih algoritma yang akan digunakan untuk pelatihan model")


# Store the choice in session state
if 'current_algorithm' not in st.session_state:
    st.session_state.current_algorithm = use_algorithm
st.session_state.use_algorithm = use_algorithm

st.info(f"menggunakan {use_algorithm} untuk pelatihan model")


if use_algorithm == "Kmeans":
    st.subheader("K-Means")
    n_cluster = kmeans.elbow_method(scaled_data)
    labels, kmeans_model = kmeans.kmeans_clustering(
        scaled_data, n_cluster, plot=True, original_data=df)

    st.session_state.clusters_number = n_cluster
    st.write(
        f"Jumlah cluster yang ditemukan: {n_cluster}"
    )

    kmeans.kmeans_summary(kmeans_model, scaled_data)
    cluster_df = df.copy()
    cluster_df['cluster'] = labels
    st.session_state.cluster_results_df = cluster_df

    st.write("### Hasil K-Means")
    st.dataframe(cluster_df, use_container_width=True)


if use_algorithm == "Hierarchical":
    st.subheader("Hierarchical Clustering")
    n_cluster = hc.find_elbow(scaled_data)
    st.write(
        f"Jumlah cluster yang ditemukan: {n_cluster}")

    # Pass the original DataFrame to save with cluster assignments
    labels, model = hc.hierarchical_clustering_with_options(
        scaled_data, n_cluster, original_data=df)

    st.session_state.clusters_number = n_cluster

    # Store current algorithm in session state for Post Hoc page
    st.session_state.current_algorithm = "Hierarchical"

    cluster_df = df.copy()
    cluster_df['cluster'] = labels
    st.session_state.cluster_results_df = cluster_df

    # Store results in session state if needed
    if labels is not None:
        st.session_state.clusters = labels
        # clustered_data is already saved in hierarchical_clustering_with_options

        st.success(
            f"Hierarchical clustering completed with {st.session_state.hc_clusters} clusters")
        hc.hc_summary(model, scaled_data, labels)


if use_algorithm == "Gaussian Mixture Model":
    st.subheader("Gaussian Mixture Model")
    bic = gmm.bic_calculation(data=scaled_data)
    gmm.plot_bic(bic)
    n_components = np.argmin(bic)+1
    st.write(
        f"Jumlah komponen optimal berdasarkan BIC: {n_components}")

    st.session_state.clusters_number = n_components

    gmm_result = gmm.gmm_fit(scaled_data, n_components, df)
    gmm.gmm_plot(scaled_data, n_components, gmm_result)

    # Get cluster labels and create DataFrame
    labels = gmm_result.predict(scaled_data)
    cluster_df = df.copy()
    cluster_df['cluster'] = labels
    st.session_state.cluster_results_df = cluster_df

    gmm.GMM_summary(gmm_result, scaled_data)


st.header("Domain Analysis")

# Only proceed if we have clustering results
if 'cluster_results_df' in st.session_state and st.session_state.cluster_results_df is not None:
    cluster_df = st.session_state.cluster_results_df

    # Define domains and their descriptions
    domains = {
        "fa": "Financial Attitude",
        "fb": "Financial Behavior",
        "fk": "Financial Knowledge",
        "m": "Materialism"
    }

    # Create tabs for domain analysis
    domain_tabs = st.tabs(list(domains.values()))

    # Prepare container for domain results to store in session_state
    domain_results = {}

    # Process each domain in its tab
    for i, (domain_prefix, domain_name) in enumerate(domains.items()):
        with domain_tabs[i]:
            st.subheader(f"{domain_name} Analysis")

            # Filter for columns from this domain (excluding 'cluster')
            domain_cols = [col for col in cluster_df.columns
                           if col.startswith(domain_prefix) and col != 'cluster']

            if domain_cols:
                # Get domain data
                domain_data = cluster_df[domain_cols + ['cluster']]

                # Show domain data
                st.write(
                    f"Features in {domain_name} domain: {len(domain_cols)}")
                st.dataframe(domain_data.head())

                # Calculate domain stats by cluster
                domain_means = domain_data.groupby(
                    'cluster').mean().reset_index()

                # Calculate domain overall score (mean of all domain features)
                domain_means[f'{domain_name}_Score'] = domain_means[domain_cols].mean(
                    axis=1)

                # Store in results
                domain_results[domain_prefix] = {
                    "features": domain_cols,
                    "cluster_means": domain_means.to_dict('records')
                }

                # Visualization: Radar chart of domain features by cluster
                import plotly.graph_objects as go

                fig = go.Figure()

                # Add traces for each cluster
                for cluster in domain_means['cluster'].unique():
                    cluster_data = domain_means[domain_means['cluster'] == cluster]

                    # Get feature values for this cluster
                    values = cluster_data[domain_cols].values.flatten(
                    ).tolist()
                    # Add the first value again to close the polygon
                    values.append(values[0])

                    # Get feature names
                    categories = domain_cols.copy()
                    categories.append(categories[0])

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )
                    ),
                    title=f"{domain_name} Features by Cluster",
                    showlegend=True
                )

                # Add unique key to plotly chart
                st.plotly_chart(fig, use_container_width=True,
                                key=f"radar_{domain_prefix}")

                # Bar chart of overall domain score by cluster
                import plotly.express as px

                fig = px.bar(
                    domain_means,
                    x='cluster',
                    y=f'{domain_name}_Score',
                    title=f"Average {domain_name} Score by Cluster",
                    labels={"cluster": "Cluster",
                            f"{domain_name}_Score": f"Avg {domain_name} Score"},
                    color='cluster'
                )

                # Add unique key to plotly chart
                st.plotly_chart(fig, use_container_width=True,
                                key=f"bar_{domain_prefix}")

                # Table with key stats
                st.write("#### Domain Statistics by Cluster")
                stats_df = domain_means[['cluster', f'{domain_name}_Score']]

                # Add highest and lowest feature
                highest_features = []
                lowest_features = []

                for _, row in domain_means.iterrows():
                    cluster = row['cluster']
                    feature_values = {col: row[col] for col in domain_cols}
                    highest_feature = max(
                        feature_values, key=feature_values.get)
                    lowest_feature = min(
                        feature_values, key=feature_values.get)
                    highest_features.append(
                        f"{highest_feature} ({feature_values[highest_feature]:.2f})")
                    lowest_features.append(
                        f"{lowest_feature} ({feature_values[lowest_feature]:.2f})")

                stats_df['Highest Feature'] = highest_features
                stats_df['Lowest Feature'] = lowest_features

                st.dataframe(stats_df)
            else:
                st.info(f"No features found for {domain_name} domain")

    # Add cross-domain comparison
    st.subheader("Cross-Domain Comparison")

    # Create a summary dataframe for all domains
    all_domains_summary = []

    for cluster in cluster_df['cluster'].unique():
        domain_scores = {"Cluster": f"Cluster {cluster}"}

        for domain_prefix, domain_name in domains.items():
            domain_cols = [col for col in cluster_df.columns
                           if col.startswith(domain_prefix) and col != 'cluster']

            if domain_cols:
                cluster_data = cluster_df[cluster_df['cluster'] == cluster]
                avg_score = cluster_data[domain_cols].mean().mean()
                domain_scores[domain_name] = avg_score

        all_domains_summary.append(domain_scores)

    all_domains_df = pd.DataFrame(all_domains_summary)

    # Plot comparison
    domains_to_plot = [domain for domain in domains.values()
                       if any(domain in col for col in all_domains_df.columns)]

    if domains_to_plot:
        fig = px.bar(
            all_domains_df,
            x="Cluster",
            y=domains_to_plot,
            title="Comparison of Domain Scores Across Clusters",
            barmode="group"
        )

        # Add unique key to plotly chart
        st.plotly_chart(fig, use_container_width=True, key="domain_comparison")

    # Save domain analysis to session state
    st.session_state.domain_analysis = domain_results

    # Overall insights based on domain analysis
    st.subheader("Domain Analysis Insights")

    # Find dominant domain for each cluster
    cluster_dominant_domains = {}

    for idx, row in all_domains_df.iterrows():
        cluster = row['Cluster']
        domain_values = {domain: row[domain]
                         for domain in domains.values() if domain in row}

        if domain_values:
            dominant_domain = max(domain_values, key=domain_values.get)
            cluster_dominant_domains[cluster] = {
                "dominant_domain": dominant_domain,
                "score": domain_values[dominant_domain]
            }

    # Display dominant domains
    for cluster, data in cluster_dominant_domains.items():
        st.info(
            f"{cluster}: Dominant domain is **{data['dominant_domain']}** with score {data['score']:.2f}")

    # Add domain interpretation section
    with st.expander("Interpretasi Domain Analysis", expanded=True):
        # Domain interpretations
        domain_interpretations = {
            "fa": {
                "high": "Pandangan positif tentang perencanaan keuangan, menabung, dan nilai manajemen keuangan yang baik",
                "low": "Sikap negatif atau apatis terhadap perencanaan keuangan dan penghematan",
                "impact": "Memengaruhi kecenderungan seseorang untuk menerapkan praktik keuangan positif"
            },
            "fb": {
                "high": "Secara konsisten menerapkan kebiasaan keuangan positif seperti menabung, membayar tepat waktu, dan perencanaan",
                "low": "Kesulitan menerapkan praktik keuangan positif secara konsisten",
                "impact": "Berkaitan langsung dengan kesehatan finansial jangka panjang"
            },
            "fk": {
                "high": "Pemahaman yang baik tentang konsep keuangan dasar seperti bunga, inflasi, dan perhitungan nilai",
                "low": "Pemahaman terbatas tentang konsep keuangan mendasar",
                "impact": "Memengaruhi kemampuan membuat keputusan keuangan yang tepat"
            },
            "m": {
                "high": "Kecenderungan menilai diri atau mengukur kebahagiaan berdasarkan kepemilikan materi",
                "low": "Fokus lebih rendah pada kepemilikan materi sebagai ukuran nilai atau kebahagiaan",
                "impact": "Berpotensi memengaruhi keputusan pembelian dan pola pengeluaran"
            }
        }

        # Display domain interpretations
        for prefix, domain in domains.items():
            st.markdown(f"### {domain}")
            st.markdown(f"""
            - **Skor Tinggi:** {domain_interpretations[prefix]['high']}
            - **Skor Rendah:** {domain_interpretations[prefix]['low']}
            - **Dampak:** {domain_interpretations[prefix]['impact']}
            """)

        # Cluster interpretations
        st.markdown("## Interpretasi Cluster")
        for cluster, data in cluster_dominant_domains.items():
            dominant_domain = data["dominant_domain"]
            score = data["score"]
            domain_prefix = next(
                k for k, v in domains.items() if v == dominant_domain)

            st.markdown(f"### {cluster}")
            st.markdown(
                f"**Domain Dominan:** {dominant_domain} (Skor: {score:.2f})")

            # Basic interpretation based on domain and score
            if domain_prefix == "fa":
                if score > 4:
                    st.markdown(
                        "Cluster ini memiliki sikap keuangan yang sangat positif, menghargai perencanaan dan pengelolaan keuangan yang baik.")
                else:
                    st.markdown(
                        "Cluster ini memiliki sikap moderat terhadap perencanaan keuangan.")
            elif domain_prefix == "fb":
                if score > 4:
                    st.markdown(
                        "Cluster ini menunjukkan perilaku keuangan yang sangat baik dalam praktik sehari-hari.")
                else:
                    st.markdown(
                        "Cluster ini menunjukkan perilaku keuangan yang cukup baik namun masih bisa ditingkatkan.")
            elif domain_prefix == "fk":
                if score > 0.7:
                    st.markdown(
                        "Cluster ini memiliki pengetahuan keuangan yang sangat baik.")
                else:
                    st.markdown(
                        "Cluster ini memiliki pengetahuan keuangan yang perlu ditingkatkan.")
            elif domain_prefix == "m":
                if score > 4:
                    st.markdown(
                        "Cluster ini memiliki kecenderungan materialisme yang tinggi, yang bisa mempengaruhi keputusan keuangan mereka.")
                else:
                    st.markdown(
                        "Cluster ini memiliki tingkat materialisme yang moderat atau rendah.")

    st.success("Domain analysis completed and saved to session state")
else:
    st.warning("Run clustering first to see domain analysis")
