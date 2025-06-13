import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from pandas.plotting import parallel_coordinates
import utils.Hierarchical_clustering as hc


def domain_clustering(df, domain_prefixes, scaled_data, algorithm='kmeans', random_state=42, use_pca=False, pca_data=None):
    """
    Performs clustering on multiple domains of features, with optimal cluster detection per domain
    """
    results = {}

    # Process each domain prefix
    for domain_prefix, domain_name in domain_prefixes.items():
        # Extract domain features
        domain_cols = [
            col for col in df.columns if col.startswith(domain_prefix)]
        if len(domain_cols) < 2:
            continue

        # Get domain data based on whether we're using PCA or not
        if use_pca and pca_data is not None and isinstance(pca_data, dict) and domain_prefix in pca_data:
            # Use PCA data for this domain
            domain_scaled_data = pca_data[domain_prefix]
        else:
            # Extract domain data from the scaled original features
            domain_indices = [df.columns.get_loc(col) for col in domain_cols]
            domain_scaled_data = scaled_data[:, domain_indices]

        n_clusters_range = range(2, 8)  # Try 2-7 clusters
        if algorithm == 'kmeans':
            # Find optimal number of clusters using elbow method for this domain
            sse = []

            for k in n_clusters_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=random_state).fit(
                    domain_scaled_data)
                sse.append(kmeans_temp.inertia_)

            # Try to find elbow point for this domain
            try:
                kn = KneeLocator(
                    list(n_clusters_range),
                    sse,
                    curve='convex',
                    direction='decreasing'
                )
                # If KneeLocator found an elbow, use it
                domain_optimal_k = kn.elbow if kn.elbow is not None else 2
            except Exception as e:
                # Default to 2 if can't find elbow
                domain_optimal_k = 2
                st.warning(
                    f"Error finding optimal clusters for {domain_name}: {str(e)}")

            # Ensure we have at least 2 clusters
            domain_optimal_k = max(2, domain_optimal_k)

            # Apply KMeans with optimal k for this domain
            kmeans = KMeans(
                n_clusters=domain_optimal_k,
                random_state=random_state,
                n_init=10  # Multiple initializations for better results
            ).fit(domain_scaled_data)

            labels = kmeans.labels_

            # Calculate validation metrics
            sil_score = silhouette_score(domain_scaled_data, labels)
            db_score = davies_bouldin_score(domain_scaled_data, labels)
            ch_score = calinski_harabasz_score(domain_scaled_data, labels)

            # Perform PCA for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(domain_scaled_data)
            explained_var = pca.explained_variance_ratio_

            # Create simple result dictionary matching the template you provided
            results[domain_prefix] = {
                "domain": domain_prefix,
                "domain_name": domain_name,
                "optimal_k": domain_optimal_k,
                "kmeans": kmeans,
                "labels": labels,
                "scaled_data": domain_scaled_data,
                "pca_data": pca_data,
                "explained_variance": explained_var,
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "domain_columns": domain_cols
            }

        if algorithm == 'hierarchical':
            elbow = hc.find_elbow(
                domain_scaled_data,
                plot=False,
                cluster_range=n_clusters_range
            )

            # Ensure elbow returns an integer value
            domain_optimal_k = int(elbow)

            # Apply hierarchical clustering with optimal k for this domain
            from sklearn.cluster import AgglomerativeClustering

            hierarchical_model = AgglomerativeClustering(
                n_clusters=domain_optimal_k,
                linkage='ward',
            )

            # Fit the model to get cluster labels
            labels = hierarchical_model.fit_predict(domain_scaled_data)

            # Calculate validation metrics
            sil_score = silhouette_score(domain_scaled_data, labels)
            db_score = davies_bouldin_score(domain_scaled_data, labels)
            ch_score = calinski_harabasz_score(domain_scaled_data, labels)

            # Perform PCA for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(domain_scaled_data)
            explained_var = pca.explained_variance_ratio_

            # Create result dictionary matching the KMeans format
            results[domain_prefix] = {
                "domain": domain_prefix,
                "domain_name": domain_name,
                "optimal_k": domain_optimal_k,
                "hierarchical_model": hierarchical_model,  # Store the model instead of kmeans
                "labels": labels,
                "scaled_data": domain_scaled_data,
                "pca_data": pca_data,
                "explained_variance": explained_var,
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "domain_columns": domain_cols
            }

            st.write(
                f"Optimal number of clusters for {domain_name} using hierarchical clustering: {domain_optimal_k}"
            )

        if algorithm == 'Gaussian Mixture Model':
            bic = gmm.bic_calculation(data=domain_scaled_data)
            n_components = np.argmin(bic)+1
            st.write(
                f"Optimal number of components for {domain_name} using GMM: {n_components}")

            # Fit GMM model with optimal number of components
            from sklearn.mixture import GaussianMixture

            gmm_model = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=random_state,
                n_init=10  # Multiple initializations for better results
            )

            # Fit the model and get cluster labels
           # Find line 188-193 where GMM predicts labels and calculates scores


# Fit the model and get cluster labels
            gmm_model.fit(domain_scaled_data)
            labels = gmm_model.predict(domain_scaled_data)

            # Add this check before calculating validation metrics
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                st.warning(
                    f"All data points in {domain_name} were assigned to a single cluster. Cannot calculate validation metrics.")
                sil_score = float('nan')  # Not a number as placeholder
                db_score = float('nan')
                ch_score = float('nan')
            else:
                # Calculate validation metrics
                sil_score = silhouette_score(domain_scaled_data, labels)
                db_score = davies_bouldin_score(domain_scaled_data, labels)
                ch_score = calinski_harabasz_score(domain_scaled_data, labels)
            # Perform PCA for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(domain_scaled_data)
            explained_var = pca.explained_variance_ratio_

            # Create result dictionary matching the format of other algorithms
            results[domain_prefix] = {
                "domain": domain_prefix,
                "domain_name": domain_name,
                "optimal_k": n_components,
                "gmm_model": gmm_model,
                "labels": labels,
                "scaled_data": domain_scaled_data,
                "pca_data": pca_data,
                "explained_variance": explained_var,
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "domain_columns": domain_cols,
                "bic_values": bic
            }

    return results


def plot_domain_clusters(domain_results):
    """
    Create a visualization grid for all domains

    Parameters:
    - domain_results: Dictionary with clustering results from domain_clustering()

    Returns:
    - fig: Matplotlib figure object for streamlit to display
    """
    if domain_results is None or len(domain_results) == 0:
        st.warning("No domain clustering results to display")
        return None

    # First show the elbow curves for each domain
    st.subheader("Optimal Cluster Selection (Elbow Method)")

    # Create tabs for elbow curves
    domain_names = {
        'fa': 'Financial Attitude',
        'fb': 'Financial Behavior',
        'fk': 'Financial Knowledge',
        'm': 'Materialism'
    }

    # Use tabs for elbow curves
    domain_tabs = st.tabs([domain_names.get(domain, domain)
                          for domain in domain_results.keys()])

    for i, (domain, tab) in enumerate(zip(domain_results.keys(), domain_tabs)):
        with tab:
            result = domain_results[domain]

            # Use columns for metrics and elbow curve
            col1, col2 = st.columns([3, 1])

            with col1:
                if "elbow_figure" in result:
                    st.pyplot(result["elbow_figure"])
                else:
                    # Create elbow curve on the fly if not stored
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if "sse_values" in result and "cluster_range" in result:
                        ax.plot(result["cluster_range"],
                                result["sse_values"], 'bo-')
                        ax.set_xlabel('Number of Clusters')
                        ax.set_ylabel('Sum of Squared Distances (Inertia)')
                        ax.set_title(
                            f'Elbow Method for {result["domain_name"]} Domain')
                        ax.axvline(x=result["optimal_k"],
                                   color='r', linestyle='--')
                        ax.text(result["optimal_k"] + 0.1, max(result["sse_values"]) * 0.9,
                                f'Optimal k = {result["optimal_k"]}', color='r')
                        ax.grid(True)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info(
                            f"Used {result['optimal_k']} clusters for this domain")

            with col2:
                st.metric("Optimal Clusters", result["optimal_k"])
                st.metric("Silhouette Score", f"{result['silhouette']:.3f}")
                st.metric("Davies-Bouldin", f"{result['davies_bouldin']:.3f}")

    # Now create the main cluster visualization
    st.subheader("Domain Cluster Visualization")

    # Determine grid size based on number of domains
    n_domains = len(domain_results)
    n_cols = min(2, n_domains)
    n_rows = (n_domains + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 14))
    if n_domains > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, (domain, result) in enumerate(domain_results.items()):
        # Plot clusters
        scatter = axs[i].scatter(
            result['pca_data'][:, 0],
            result['pca_data'][:, 1],
            c=result['labels'],
            cmap='viridis',
            alpha=0.8,
            s=80,
            edgecolor='k',
            linewidth=0.5
        )

        # Add cluster centroids in PCA space
        for k in range(result['optimal_k']):
            cluster_points = result['pca_data'][result['labels'] == k]
            if len(cluster_points) > 0:  # Make sure there are points in this cluster
                centroid = np.mean(cluster_points, axis=0)
                axs[i].scatter(
                    centroid[0],
                    centroid[1],
                    s=200,
                    marker='*',
                    c='red',
                    edgecolor='k',
                    linewidth=1.5,
                    zorder=10
                )
                axs[i].text(
                    centroid[0],
                    centroid[1],
                    f"C{k}",
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color='white',
                    zorder=11
                )

        # Set labels and title
        domain_name = result.get('domain_name', domain)
        axs[i].set_title(
            f'{domain_name} (k={result["optimal_k"]})\nSilhouette: {result["silhouette"]:.3f}', fontsize=14)
        axs[i].set_xlabel('PCA Component 1', fontsize=12)
        axs[i].set_ylabel('PCA Component 2', fontsize=12)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        legend = axs[i].legend(*scatter.legend_elements(), title="Clusters")
        axs[i].add_artist(legend)

    # Remove any unused subplots
    for i in range(n_domains, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.suptitle('Domain-Specific Clustering Results', fontsize=18, y=1.02)

    st.pyplot(fig)  # Automatically display in Streamlit
    return fig


def create_comparison_table(domain_results):
    """
    Create a comparison table of clustering results

    Parameters:
    - domain_results: Dictionary with clustering results from domain_clustering()

    Returns:
    - DataFrame containing comparison metrics
    """
    domain_names = {
        'fa': 'Financial Attitude',
        'fb': 'Financial Behavior',
        'fk': 'Financial Knowledge',
        'm': 'Materialism'
    }

    comparison_df = pd.DataFrame({
        'Domain': [domain_names.get(d, d) for d in domain_results.keys()],
        'Features': [len(result['domain_columns']) for result in domain_results.values()],
        'Clusters': [result['optimal_k'] for result in domain_results.values()],
        'Silhouette': [result['silhouette'] for result in domain_results.values()],
        'Davies-Bouldin': [result['davies_bouldin'] for result in domain_results.values()],
        'Calinski-Harabasz': [result['calinski_harabasz'] for result in domain_results.values()],
    })

    return comparison_df


def analyze_domain_clusters(df_original, domain_results):
    """
    Analyze characteristics of clusters within each domain and relationships between domains

    Parameters:
    - df_original: Original dataframe
    - domain_results: Dictionary with clustering results from domain_clustering()

    Returns:
    - df_with_clusters: DataFrame with cluster assignments
    - cross_tabs: Dictionary of cross-tabulations between domains
    - crosstab_figs: List of matplotlib figures for streamlit to display
    - parallel_fig: Matplotlib figure for parallel coordinates
    - pattern_counts: DataFrame of common cluster patterns
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import streamlit as st
    from pandas.plotting import parallel_coordinates

    # Check if we have valid domain results
    if domain_results is None or len(domain_results) == 0:
        st.warning("No valid domain clustering results to analyze")
        return None, None, None, None, None

    # Create a copy of the original dataframe to add cluster assignments
    df_with_clusters = df_original.copy()

    # Store cluster labels for each domain
    domain_names = {
        'fa': 'Financial Attitude',
        'fb': 'Financial Behavior',
        'fk': 'Financial Knowledge',
        'm': 'Materialism'
    }

    # Add cluster assignments from each domain to the dataframe
    for domain, result in domain_results.items():
        cluster_name = f"{domain}_cluster"
        df_with_clusters[cluster_name] = result['labels']

    # Get list of cluster column names
    cluster_cols = [f"{domain}_cluster" for domain in domain_results.keys()]

    # Create cross-tabulations between all pairs of domains
    cross_tabs = {}
    crosstab_figs = []

    # Only proceed if we have at least 2 domains
    if len(domain_results) >= 2:
        domains = list(domain_results.keys())

        # For each pair of domains
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                d1, d2 = domains[i], domains[j]
                cross_name = f"{d1}_vs_{d2}"

                # Calculate cross-tabulation with normalization by row
                cross_tabs[cross_name] = pd.crosstab(
                    df_with_clusters[f"{d1}_cluster"],
                    df_with_clusters[f"{d2}_cluster"],
                    normalize='index'
                )

                # Create visualization for this cross-tabulation
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    cross_tabs[cross_name],
                    annot=True,
                    fmt='.2f',
                    cmap='viridis',
                    ax=ax,
                    cbar_kws={'label': 'Proportion'}
                )

                # Set title and labels using domain names instead of prefixes
                d1_name = domain_names.get(d1, d1)
                d2_name = domain_names.get(d2, d2)
                ax.set_title(
                    f'Cluster Association: {d1_name} vs {d2_name}', fontsize=14)
                ax.set_xlabel(f'{d2_name} Cluster', fontsize=12)
                ax.set_ylabel(f'{d1_name} Cluster', fontsize=12)

                plt.tight_layout()
                crosstab_figs.append(fig)

        # Show preliminary results in Streamlit
        # if len(crosstab_figs) > 0:
        #     st.subheader("Cross-Domain Cluster Relationships")
        #     st.markdown("""
        #     These heatmaps show how clusters from different domains relate to each other.
        #     Higher values (darker colors) indicate a stronger association between clusters.
        #     """)

        #     # Create tabs for each cross-tabulation
        #     tab_names = [f"{domain_names.get(d1, d1)} vs {domain_names.get(d2, d2)}"
        #                  for d1, d2 in [cross_name.split('_vs_') for cross_name in cross_tabs.keys()]]

        #     tabs = st.tabs(tab_names)

        #     for i, tab in enumerate(tabs):
        #         with tab:
        #             st.pyplot(crosstab_figs[i])

    # # Create parallel coordinates plot for visualizing cluster assignments across domains
    # parallel_data = df_with_clusters[cluster_cols].copy()

    # # Sample the data if it's too large (for performance)
    # sample_size = min(200, len(df_with_clusters))
    # if len(df_with_clusters) > sample_size:
    #     parallel_sample = parallel_data.sample(sample_size, random_state=42)
    # else:
    #     parallel_sample = parallel_data

    # # Add a group column for coloring (all in same group for now)
    # parallel_sample['group'] = 'all'

    # # Create parallel coordinates plot
    # plt.figure(figsize=(12, 6))
    # parallel_fig = plt.figure(figsize=(12, 6))

    # # Rename columns for better readability
    # renamed_parallel = parallel_sample.copy()
    # renamed_parallel.columns = [domain_names.get(col.split('_')[0], col) if col != 'group' else col
    #                             for col in renamed_parallel.columns]

    # # Generate parallel coordinates plot
    # parallel_coordinates(renamed_parallel, 'group', colormap='viridis')
    # plt.title('Cluster Assignments Across Domains', fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()

    # Find most common cross-domain patterns
    pattern_counts = df_with_clusters[cluster_cols].value_counts(
    ).reset_index()
    pattern_counts.columns = list(pattern_counts.columns[:-1]) + ['count']
    pattern_counts['percentage'] = pattern_counts['count'] / \
        len(df_with_clusters) * 100

    # Create Sankey diagram for visualizing flows between domains
    st.subheader("Diagram Aliran Antar Kluster (Sankey)")
    st.markdown("""
    Diagram Sankey menunjukkan bagaimana responden terdistribusi di antara kluster dari domain yang berbeda.
    Semakin tebal aliran, semakin banyak responden yang tergolong dalam kedua kluster yang terhubung.
    """)

    sankey_fig = create_sankey_diagram(df_with_clusters, domain_results)
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)

    # # Display parallel coordinates plot
    # st.subheader("Parallel Coordinates Plot")
    # st.markdown("""
    # This plot shows how respondents are distributed across clusters in different domains.
    # Each line represents a respondent, and you can see their cluster assignments across all domains.
    # """)
    # st.pyplot(parallel_fig)

    # Display common patterns
    st.subheader("Most Common Cluster Patterns")
    st.markdown("""
    This table shows the most common combinations of cluster assignments across domains.
    """)

    # Format and display the pattern counts
    pattern_display = pattern_counts.head(10).copy()

    # Rename columns for readability
    pattern_display.columns = [domain_names.get(col.split('_')[0], col) if '_cluster' in col
                               else col for col in pattern_display.columns]

    # Display the table
    st.dataframe(pattern_display)

    return df_with_clusters, cross_tabs, crosstab_figs, pattern_counts


def run_domain_clustering_analysis(df, domain_prefixes=['fa', 'fb', 'fk', 'm']):
    """
    Main function to run domain-based clustering analysis in Streamlit

    Parameters:
    - df: DataFrame with the data
    - domain_prefixes: List of domain prefixes to analyze
    """
    st.title("Domain-Based Clustering Analysis")

    st.markdown("""
    This analysis groups respondents based on their responses within specific financial domains.
    Each domain (Financial Attitude, Financial Behavior, Financial Knowledge, and Materialism) is
    clustered separately to identify distinct response patterns.
    """)

    # Let user select which domains to analyze
    selected_domains = st.multiselect(
        "Select domains to analyze",
        domain_prefixes,
        default=domain_prefixes,
        format_func=lambda x: {
            'fa': 'Financial Attitude',
            'fb': 'Financial Behavior',
            'fk': 'Financial Knowledge',
            'm': 'Materialism'
        }.get(x, x)
    )

    if not selected_domains:
        st.warning("Please select at least one domain to analyze.")
        return

    # Set up progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Apply clustering to each domain
    domain_results = {}

    for i, prefix in enumerate(selected_domains):
        status_text.text(f"Analyzing {prefix} domain...")
        result = domain_clustering(df, prefix, scaler=MinMaxScaler())
        if result:
            domain_results[prefix] = result
            domain_name = {
                'fa': 'Financial Attitude',
                'fb': 'Financial Behavior',
                'fk': 'Financial Knowledge',
                'm': 'Materialism'
            }.get(prefix, prefix)

            st.subheader(f"Clustering Results for {domain_name}")

            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Clusters", result['optimal_k'])
            col2.metric("Silhouette Score", f"{result['silhouette']:.3f}")
            col3.metric("Davies-Bouldin Index",
                        f"{result['davies_bouldin']:.3f}")
            col4.metric("Calinski-Harabasz Index",
                        f"{result['calinski_harabasz']:.3f}")

            # Update progress
            progress_value = (i + 1) / len(selected_domains)
            progress_bar.progress(progress_value)

    if not domain_results:
        st.error("No valid domains found for clustering.")
        return

    status_text.text("Creating visualizations...")

    # Create and display domain cluster visualization
    fig = plot_domain_clusters(domain_results)
    st.pyplot(fig)

    # Display comparison table
    st.subheader("Domain Clustering Comparison")
    comparison_df = create_comparison_table(domain_results)
    st.dataframe(comparison_df.style.background_gradient(
        subset=['Silhouette', 'Calinski-Harabasz'], cmap='viridis'
    ).background_gradient(
        subset=['Davies-Bouldin'], cmap='viridis_r'
    ))

    # Cross-domain analysis
    st.subheader("Cross-Domain Analysis")

    if st.checkbox("Show detailed cross-domain analysis"):
        df_with_clusters, cross_tabs, crosstab_figs, parallel_fig, pattern_counts = analyze_domain_clusters(
            df, domain_results)

        # Show parallel coordinates plot
        st.subheader("Parallel Coordinates Plot")
        st.markdown(
            "This plot shows how respondents are distributed across different domain clusters.")
        st.pyplot(parallel_fig)

        # Show cross-tabulations
        st.subheader("Cross-Domain Associations")
        st.markdown(
            "These heatmaps show how clusters from different domains are associated with each other.")

        # Display each cross-tabulation figure
        for fig in crosstab_figs:
            st.pyplot(fig)

        # Show most common patterns
        st.subheader("Most Common Cross-Domain Patterns")
        st.dataframe(pattern_counts.head(10))

        # Option to download clustered data
        csv = df_with_clusters.to_csv(index=False)
        st.download_button(
            label="Download Clustered Data as CSV",
            data=csv,
            file_name='domain_clustered_data.csv',
            mime='text/csv',
        )

    status_text.text("Analysis complete!")
    progress_bar.progress(1.0)

    # Return the results for potential further use
    return domain_results


def interpret_domain_clusters(df_original, domain_results):
    """
    Menginterpretasikan karakteristik setiap kluster dalam masing-masing domain

    Parameters:
    - df_original: DataFrame asli dengan semua fitur
    - domain_results: Dictionary hasil clustering dari setiap domain

    Returns:
    - Dictionary berisi interpretasi untuk setiap kluster dalam setiap domain
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import streamlit as st

    domain_names = {
        'fa': 'Financial Attitude',
        'fb': 'Financial Behavior',
        'fk': 'Financial Knowledge',
        'm': 'Materialism'
    }

    interpretations = {}

    st.subheader("Interpretasi Karakteristik Kluster Per Domain")

    # Create container for all visualizations
    interpretation_container = st.container()

    # Untuk setiap domain
    for domain, result in domain_results.items():
        domain_cols = result['domain_columns']
        labels = result['labels']
        optimal_k = result['optimal_k']

        with interpretation_container:
            st.markdown(f"### {domain_names.get(domain, domain)}")

            # Tambahkan label kluster ke dataframe
            temp_df = df_original[domain_cols].copy()
            temp_df['cluster'] = labels

            # Hitung profil kluster (nilai rata-rata untuk setiap fitur per kluster)
            cluster_profiles = temp_df.groupby('cluster')[domain_cols].mean()

            # Hitung statistik global untuk perbandingan
            global_mean = temp_df[domain_cols].mean()
            global_std = temp_df[domain_cols].std()

            # Untuk standardisasi perbandingan, hitung z-score profil kluster
            z_profiles = pd.DataFrame()
            for col in domain_cols:
                if global_std[col] > 0:  # Hindari pembagian dengan nol
                    z_profiles[col] = (
                        cluster_profiles[col] - global_mean[col]) / global_std[col]
                else:
                    z_profiles[col] = 0

            # Tentukan fitur yang paling karakteristik untuk setiap kluster
            domain_interpretations = {}

            # Ukuran kluster untuk interpretasi representativitas
            cluster_sizes = temp_df['cluster'].value_counts().sort_index()
            cluster_pcts = (cluster_sizes / len(temp_df) * 100).round(1)

            # Visualisasi z-scores dalam heatmap untuk mendukung interpretasi
            fig_heatmap, ax_heatmap = plt.subplots(
                figsize=(max(10, len(domain_cols) * 0.8), optimal_k * 1.2))
            sns.heatmap(z_profiles, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                        linewidths=.5, cbar_kws={"label": "Z-Score"}, ax=ax_heatmap)
            plt.title(
                f'Profil Kluster {domain_names.get(domain, domain)}\n(Standardized Z-Scores)', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_heatmap)

            # Visualisasi profil kluster dengan nilai absolut
            fig_bar, ax_bar = plt.subplots(
                figsize=(max(12, len(domain_cols) * 0.8), 6))
            cluster_profiles.T.plot(kind='bar', ax=ax_bar)
            plt.title(
                f'Profil Kluster {domain_names.get(domain, domain)}\n(Nilai Rata-rata)', fontsize=14)
            plt.ylabel('Nilai Rata-rata')
            plt.xlabel('Fitur')
            plt.legend(title='Kluster')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_bar)

            # Interpretasi setiap kluster
            for cluster_idx in range(optimal_k):
                # Identifikasi fitur dengan z-score ekstrem (positif maupun negatif)
                extreme_features_high = z_profiles.loc[cluster_idx].sort_values(
                    ascending=False).head(3)
                extreme_features_low = z_profiles.loc[cluster_idx].sort_values().head(
                    3)

                # Buat interpretasi kluster
                st.markdown(
                    f"#### Kluster {cluster_idx} - {cluster_sizes[cluster_idx]} responden ({cluster_pcts[cluster_idx]}%)")

                # Fitur dengan nilai tinggi (di atas rata-rata)
                high_features_text = []
                for feature, z_score in extreme_features_high.items():
                    if z_score > 0.5:  # Filter hanya yang signifikan
                        high_features_text.append(
                            f"• {feature}: {z_score:.2f} SD di atas rata-rata " +
                            f"(Nilai: {cluster_profiles.loc[cluster_idx, feature]:.2f} vs. Global: {global_mean[feature]:.2f})"
                        )

                # Fitur dengan nilai rendah (di bawah rata-rata)
                low_features_text = []
                for feature, z_score in extreme_features_low.items():
                    if z_score < -0.5:  # Filter hanya yang signifikan
                        low_features_text.append(
                            f"• {feature}: {abs(z_score):.2f} SD di bawah rata-rata " +
                            f"(Nilai: {cluster_profiles.loc[cluster_idx, feature]:.2f} vs. Global: {global_mean[feature]:.2f})"
                        )

                # Tampilkan karakteristik menonjol
                if high_features_text:
                    st.markdown(
                        "**Karakteristik Menonjol (Di Atas Rata-rata):**")
                    for text in high_features_text:
                        st.markdown(text)

                if low_features_text:
                    st.markdown(
                        "**Karakteristik Menonjol (Di Bawah Rata-rata):**")
                    for text in low_features_text:
                        st.markdown(text)

                # Buat narasi singkat berdasarkan pola z-score
                st.markdown("**Interpretasi:**")

                # Hitung tingkat kecenderungan keseluruhan berdasarkan rata-rata z-score
                avg_zscore = z_profiles.loc[cluster_idx].mean()

                if domain == 'fa':  # Financial Attitude
                    if avg_zscore > 0.5:
                        narrative = "Kelompok dengan sikap keuangan positif/baik secara keseluruhan."
                    elif avg_zscore < -0.5:
                        narrative = "Kelompok dengan sikap keuangan kurang baik secara keseluruhan."
                    else:
                        # Cek apakah ada pola khusus dalam aspek tertentu
                        conservative_features = [col for col in domain_cols if "hemat" in col.lower(
                        ) or "berhati-hati" in col.lower()]
                        risk_features = [
                            col for col in domain_cols if "risiko" in col.lower() or "risk" in col.lower()]

                        if conservative_features and z_profiles.loc[cluster_idx, conservative_features].mean() > 0.5:
                            narrative = "Kelompok dengan sikap keuangan konservatif dan berhati-hati."
                        elif risk_features and z_profiles.loc[cluster_idx, risk_features].mean() > 0.5:
                            narrative = "Kelompok dengan sikap keuangan berani mengambil risiko."
                        else:
                            narrative = "Kelompok dengan sikap keuangan moderat atau campuran."

                elif domain == 'fb':  # Financial Behavior
                    if avg_zscore > 0.5:
                        narrative = "Kelompok dengan perilaku keuangan positif/baik secara keseluruhan."
                    elif avg_zscore < -0.5:
                        narrative = "Kelompok dengan perilaku keuangan kurang baik secara keseluruhan."
                    else:
                        # Cek perilaku pengelolaan dan perencanaan
                        planning_features = [
                            col for col in domain_cols if "rencana" in col.lower() or "plan" in col.lower()]
                        saving_features = [
                            col for col in domain_cols if "tabung" in col.lower() or "save" in col.lower()]

                        if planning_features and z_profiles.loc[cluster_idx, planning_features].mean() > 0.5:
                            narrative = "Kelompok dengan perilaku perencanaan keuangan yang baik."
                        elif saving_features and z_profiles.loc[cluster_idx, saving_features].mean() > 0.5:
                            narrative = "Kelompok dengan kebiasaan menabung yang baik."
                        else:
                            narrative = "Kelompok dengan perilaku keuangan moderat atau campuran."

                elif domain == 'fk':  # Financial Knowledge
                    if avg_zscore > 0.5:
                        narrative = "Kelompok dengan pengetahuan keuangan tinggi."
                    elif avg_zscore < -0.5:
                        narrative = "Kelompok dengan pengetahuan keuangan rendah."
                    else:
                        narrative = "Kelompok dengan pengetahuan keuangan moderat."

                elif domain == 'm':  # Materialism
                    if avg_zscore > 0.5:
                        narrative = "Kelompok dengan kecenderungan materialisme tinggi."
                    elif avg_zscore < -0.5:
                        narrative = "Kelompok dengan kecenderungan materialisme rendah."
                    else:
                        narrative = "Kelompok dengan kecenderungan materialisme moderat."

                # Tampilkan narasi
                st.markdown(f"_{narrative}_")

                # Simpan interpretasi dalam dictionary
                domain_interpretations[cluster_idx] = {
                    'size': int(cluster_sizes[cluster_idx]),
                    'percentage': float(cluster_pcts[cluster_idx]),
                    'high_features': {k: float(v) for k, v in extreme_features_high.items() if v > 0.5},
                    'low_features': {k: float(v) for k, v in extreme_features_low.items() if v < -0.5},
                    'narrative': narrative.strip()
                }

            # Simpan semua interpretasi domain ke dalam dictionary utama
            interpretations[domain] = domain_interpretations

    # Tambahan: Visualisasi kombinasi kluster yang umum
    st.subheader("Kombinasi Kluster yang Umum dan Interpretasinya")

    # Get the df_with_clusters from the analyze_domain_clusters function
    # This assumes analyze_domain_clusters has been run before this function
    df_with_clusters = df_original.copy()
    for domain, result in domain_results.items():
        cluster_name = f"{domain}_cluster"
        df_with_clusters[cluster_name] = result['labels']

    cluster_cols = [f"{domain}_cluster" for domain in domain_results.keys()]

    # Hitung pola kluster yang umum
    pattern_counts = df_with_clusters[cluster_cols].value_counts(
    ).reset_index()
    pattern_counts.columns = list(pattern_counts.columns[:-1]) + ['count']
    pattern_counts['percentage'] = pattern_counts['count'] / \
        len(df_with_clusters) * 100

    # Tampilkan 5 pola terbanyak
    top_n = min(5, len(pattern_counts))
    for i, row in pattern_counts.head(top_n).iterrows():
        pattern_str = " - ".join(
            [f"{domain_names.get(d.split('_')[0], d.split('_')[0])} C{int(row[d])}" for d in cluster_cols])
        st.markdown(
            f"#### Kombinasi #{i+1} ({row['percentage']:.1f}% responden)")
        st.markdown(f"**{pattern_str}**")

        st.markdown("**Interpretasi Gabungan:**")
        domain_narratives = []

        for col in cluster_cols:
            domain = col.split('_')[0]
            cluster = int(row[col])
            if domain in interpretations and cluster in interpretations[domain]:
                domain_narratives.append(
                    interpretations[domain][cluster]['narrative'])

        if domain_narratives:
            st.markdown("_" + " ".join(domain_narratives) + "_")
        else:
            st.markdown(
                "_Tidak ada interpretasi tersedia untuk kombinasi ini._")

    return interpretations


def create_sankey_diagram(df_with_clusters, domain_results):
    """
    Create a Sankey diagram showing relationships between clusters across domains

    Parameters:
    - df_with_clusters: DataFrame with cluster assignments for each domain
    - domain_results: Dictionary with clustering results from domain_clustering()

    Returns:
    - Plotly figure object for Streamlit to display
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Please install plotly: pip install plotly")
        return None

    # Prepare data for Sankey diagram
    domain_names = {
        'fa': 'Financial Attitude',
        'fb': 'Financial Behavior',
        'fk': 'Financial Knowledge',
        'm': 'Materialism'
    }

    # Only proceed if we have at least 2 domains
    if len(domain_results) < 2:
        st.warning("Need at least 2 domains to create a Sankey diagram")
        return None

    # Get the list of domains in order
    domains = list(domain_results.keys())

    # Initialize link data
    source = []
    target = []
    value = []
    link_labels = []

    # Create an index mapping for nodes
    node_indices = {}
    node_labels = []
    node_colors = []

    # Define colors for each domain (more distinct colors with better contrast for text)
    domain_colors = {
        'fa': 'rgba(31, 119, 180, 0.7)',   # blue
        'fb': 'rgba(255, 127, 14, 0.7)',   # orange
        'fk': 'rgba(44, 160, 44, 0.7)',    # green
        'm': 'rgba(214, 39, 40, 0.7)'      # red
    }

    # Add nodes for each cluster in each domain
    current_index = 0
    for domain in domains:
        n_clusters = domain_results[domain]['optimal_k']
        domain_color = domain_colors.get(
            domain, 'rgba(0, 0, 0, 0)')  # default gray

        for cluster in range(n_clusters):
            node_key = f"{domain}_{cluster}"
            node_indices[node_key] = current_index
            # Use domain name without HTML tags for better rendering
            domain_display_name = domain_names.get(domain, domain)
            node_labels.append(f"{domain_display_name} Cluster {cluster}")

            node_colors.append(domain_color)
            current_index += 1

    # Create links between domains
    for i in range(len(domains) - 1):
        source_domain = domains[i]
        target_domain = domains[i + 1]

        # Group by source and target clusters to get counts
        flow_counts = df_with_clusters.groupby(
            [f"{source_domain}_cluster", f"{target_domain}_cluster"]
        ).size().reset_index(name='count')

        # Add links for each flow
        for _, row in flow_counts.iterrows():
            src_cluster = int(row[f"{source_domain}_cluster"])
            tgt_cluster = int(row[f"{target_domain}_cluster"])
            count = row['count']
            percentage = 100 * count / len(df_with_clusters)

            source.append(node_indices[f"{source_domain}_{src_cluster}"])
            target.append(node_indices[f"{target_domain}_{tgt_cluster}"])
            value.append(count)
            link_labels.append(f"{count} responden ({percentage:.1f}%)")

    # Create Sankey diagram with improved text rendering
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            label=node_labels,
            color=node_colors,

        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hoverinfo="all",
            hovertemplate="Dari: %{source.label}<br>Ke: %{target.label}<br>Jumlah: %{value} responden<extra></extra>",
        )
    )])

    # Improve overall figure appearance and text rendering
    fig.update_layout(
        title_text="Aliran Responden Antar Kluster Domain",
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Larger font size for better readability
            color="black"  # Dark text for better contrast
        ),
        height=600,
        margin=dict(l=25, r=25, t=60, b=25),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig


def hierarchical_domain_clustering(df, domain_prefixes, scaled_data, random_state=42):
    """
    Performs hierarchical clustering on multiple domains of features, optimizing each domain separately.
    """
    results = {}

    # Process each domain prefix
    for domain_prefix, domain_name in domain_prefixes.items():
        # Extract domain features
        domain_cols = [
            col for col in df.columns if col.startswith(domain_prefix)]
        if len(domain_cols) < 2:
            st.warning(
                f"Domain {domain_name} has less than 2 features. Skipping.")
            continue

        # Get domain data
        domain_data = df[domain_cols].copy()

        # Get domain scaled data (from the pre-scaled full dataset)
        domain_indices = [df.columns.get_loc(col) for col in domain_cols]
        domain_scaled_data = scaled_data[:, domain_indices]

        # Determine optimal number of clusters using WSS
        n_clusters_range = range(2, 8)  # Try 2-7 clusters

        try:
            # Calculate WSS for different numbers of clusters
            wss_values = hc.wss_calculation(
                domain_scaled_data, n_clusters_range)

            # Try to find elbow point for this domain
            try:
                kn = KneeLocator(
                    list(n_clusters_range),
                    wss_values,
                    curve='convex',
                    direction='decreasing'
                )
                # If KneeLocator found an elbow, use it and ensure it's an integer
                domain_optimal_k = int(kn.elbow) if kn.elbow is not None else 2
            except Exception as e:
                # Default to 2 if can't find elbow
                domain_optimal_k = 2
                st.warning(
                    f"Error finding optimal clusters for {domain_name} using hierarchical clustering: {str(e)}")

            # Ensure we have at least 2 clusters
            domain_optimal_k = max(2, domain_optimal_k)

            # Verify we have a valid integer before passing to the function
            if not isinstance(domain_optimal_k, int):
                domain_optimal_k = int(domain_optimal_k)

            # Look at the hierarchical_clustering_with_options function to check its parameter order
            # Directly initialize AgglomerativeClustering to ensure correct parameters
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(
                n_clusters=domain_optimal_k,
                linkage='ward',
                affinity='euclidean'
            )

            labels = model.fit_predict(domain_scaled_data)

            # Calculate validation metrics
            sil_score = silhouette_score(domain_scaled_data, labels)
            db_score = davies_bouldin_score(domain_scaled_data, labels)
            ch_score = calinski_harabasz_score(domain_scaled_data, labels)

            # Perform PCA for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(domain_scaled_data)
            explained_var = pca.explained_variance_ratio_

            # Create result dictionary
            results[domain_prefix] = {
                "domain": domain_prefix,
                "domain_name": domain_name,
                "optimal_k": domain_optimal_k,
                "hierarchical_model": model,
                "labels": labels,
                "scaled_data": domain_scaled_data,
                "pca_data": pca_data,
                "explained_variance": explained_var,
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "domain_columns": domain_cols
            }

        except Exception as e:
            st.error(
                f"Error performing hierarchical clustering on {domain_name} domain: {str(e)}")
            # Continue with next domain instead of failing completely
            continue

    return results
