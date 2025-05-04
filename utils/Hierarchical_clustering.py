from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram


def wss_calculation(K_range, data):
    """
    Calculate WSS (Within Sum of Squares) for different numbers of clusters.

    Parameters:
    K_range: A range object or iterable with the cluster numbers to try
    data: The data to cluster

    Returns:
    List of WSS values for each K in K_range
    """
    WSS = []
    for k in K_range:  # Iterate over the range directly
        cluster = AgglomerativeClustering(
            n_clusters=k, linkage="ward"  # Use k directly
        )
        cluster.fit_predict(data)
        # cluster index
        label = cluster.labels_
        wss = []
        for j in range(k):  # Use k instead of i+1
            # extract each cluster according to its index
            idx = [t for t, e in enumerate(label) if e == j]
            cluster = data[idx,]
            # calculate the WSS:
            cluster_mean = cluster.mean(axis=0)
            distance = np.sum(np.abs(cluster - cluster_mean) ** 2, axis=-1)
            wss.append(sum(distance))
        WSS.append(sum(wss))
    return WSS


def find_elbow(data, plot=True, cluster_range=range(1, 11)):
    """
    Find the elbow point in the WSS plot.

    Parameters:
    data: The data to cluster
    plot: Whether to display the plot
    cluster_range: Range of cluster numbers to try

    Returns:
    int: The optimal number of clusters (elbow point)
    """
    wss_calc = wss_calculation(cluster_range, data)

    # Find the elbow point before plotting
    elbow_point = np.diff(np.diff(wss_calc)).argmax() + 2

    if plot:
        import matplotlib.pyplot as plt
        import streamlit as st

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Optimal number of clusters for Agglomerative Clustering")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Total intra-cluster variation (WSS)")

        # Plot all points
        ax.plot(list(cluster_range), wss_calc, marker="o", linestyle='-', color='blue',
                label='WSS values')

        # Highlight the elbow point with a different color and larger marker
        elbow_index = list(cluster_range).index(elbow_point)
        ax.plot(elbow_point, wss_calc[elbow_index], 'ro', markersize=12,
                label=f'Elbow Point (k={elbow_point})')

        # Add vertical line at the elbow point
        ax.axvline(x=elbow_point, color='r', linestyle='--', alpha=0.3)

        # Add annotation arrow pointing to the elbow
        ax.annotate(f'Optimal k={elbow_point}',
                    xy=(elbow_point, wss_calc[elbow_index]),
                    xytext=(
                        elbow_point+1, wss_calc[elbow_index]+0.1*(max(wss_calc)-min(wss_calc))),
                    arrowprops=dict(facecolor='black',
                                    shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)

        ax.set_xticks(list(cluster_range))
        ax.grid(True)
        ax.legend()

        # Display in Streamlit instead of plt.show()
        st.pyplot(fig)

    # Use st.write instead of print for Streamlit
    import streamlit as st

    return elbow_point


def plot_dendrogram(model, X):
    """
    Plot a dendrogram for the hierarchical clustering model.

    Parameters:
    -----------
    model : AgglomerativeClustering
        The fitted hierarchical clustering model with compute_distances=True
    X : array-like
        The input data used for clustering
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram

    # Create linkage matrix from the children attribute
    def get_linkage_matrix(model):
        # Create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([
            model.children_,
            model.distances_,
            counts
        ]).astype(float)

        return linkage_matrix

    try:
        # Get the linkage matrix
        linkage_matrix = get_linkage_matrix(model)

        # Plot the dendrogram
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use orientation='right' for a more readable layout with many datapoints
        dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=5,  # Only show 5 levels
            orientation='right',
            ax=ax
        )

        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Sample index or cluster')

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Display using Streamlit
        st.pyplot(fig)

    except AttributeError:
        st.warning(
            "Could not plot dendrogram. Make sure you set compute_distances=True when creating the model.")
    except Exception as e:
        st.error(f"Error plotting dendrogram: {str(e)}")


def plot_clusters(data, labels, n_clusters):
    """
    Plot clusters visualization for hierarchical clustering results.

    Parameters:
    -----------
    data : array-like
        The input data used for clustering
    labels : array-like
        The cluster labels
    n_clusters : int
        Number of clusters
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create PCA plot if data is multidimensional
    if data.shape[1] > 2:
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        # Get explained variance
        explained_var = pca.explained_variance_ratio_ * 100

        # Set axis labels with explained variance
        ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1f}%)')
        ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1f}%)')
    else:
        # Use original data if already 2D
        data_2d = data

        # Set basic axis labels
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    # Create scatter plot with clusters
    scatter = ax.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=labels,
        cmap='viridis',
        s=50,
        alpha=0.8,
        edgecolors='w'
    )

    # Add title
    ax.set_title(f'Hierarchical Clustering Results (k={n_clusters})')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add cluster labels
    for i in range(n_clusters):
        # Find the center of each cluster
        cluster_points = data_2d[labels == i]
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            ax.text(
                center[0],
                center[1],
                f'Cluster {i}',
                horizontalalignment='center',
                verticalalignment='center',
                size=12,
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

    # Add colorbar
    legend = ax.legend(*scatter.legend_elements(),
                       loc="upper right", title="Clusters")
    ax.add_artist(legend)

    # Add cluster sizes annotation
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    size_text = "Cluster sizes:\n" + "\n".join([f"Cluster {i}: {size} samples"
                                                for i, size in cluster_sizes.items()])
    ax.text(
        0.02, 0.02,
        size_text,
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7),
        fontsize=9,
        verticalalignment='bottom'
    )

    # Add explained variance if PCA was applied
    if data.shape[1] > 2:
        ax.text(
            0.98, 0.02,
            f'Total explained variance:\n{sum(explained_var):.1f}%',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=9
        )

    # Display in Streamlit
    st.pyplot(fig)


def hierarchical_clustering(data, n_clusters, linkage_method='ward', plot=True, plot_dendrogram=True):
    """
    Perform hierarchical clustering and visualize results with PCA.

    Parameters:
    data: The data to cluster
    n_clusters: The number of clusters to form
    linkage_method: The linkage method to use ('ward', 'complete', 'average', or 'single')
    plot: Whether to display the plots
    plot_dendrogram: Whether to plot the dendrogram (can be resource-intensive for large datasets)

    Returns:
    tuple: (cluster_labels, cluster_model)
    """
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram

    # Perform hierarchical clustering
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        compute_distances=True if plot_dendrogram else False
    )

    labels = model.fit_predict(data)

    if plot:
        # Create a figure with either 1 or 2 subplots depending on dendrogram option
        if plot_dendrogram:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            plot_title = f"Hierarchical Clustering (k={n_clusters}, linkage={linkage_method})"
        else:
            fig, ax2 = plt.subplots(figsize=(10, 6))
            plot_title = f"Hierarchical Clustering with PCA (k={n_clusters}, linkage={linkage_method})"

        # Plot the dendrogram if requested
        if plot_dendrogram and hasattr(model, 'distances_'):
            # Plot the dendrogram
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            linkage_matrix = np.column_stack(
                [model.children_, model.distances_, counts]
            ).astype(float)

            dendrogram(
                linkage_matrix,
                ax=ax1,
                truncate_mode='level',
                p=3,  # Show only last p merged clusters
                color_threshold=0.7 * max(model.distances_),
                leaf_font_size=9
            )

            ax1.set_title('Hierarchical Clustering Dendrogram')
            ax1.set_xlabel('Sample index or (cluster size)')
            ax1.set_ylabel('Distance')

        # Create PCA plot if data is multidimensional
        if data.shape[1] > 2:
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)

            # Get explained variance
            explained_var = pca.explained_variance_ratio_ * 100
        else:
            # Use original data if already 2D
            data_2d = data
            explained_var = [0, 0]  # Placeholder

        # Create scatter plot with clusters
        scatter = ax2.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=0.8,
            edgecolors='w'
        )

        # Add title and labels
        if data.shape[1] > 2:
            ax2.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1f}%)')
            ax2.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1f}%)')
        else:
            ax2.set_xlabel('Feature 1')
            ax2.set_ylabel('Feature 2')

        ax2.set_title('Clusters Visualization')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Add cluster labels
        for i in range(n_clusters):
            # Find the center of each cluster
            cluster_points = data_2d[labels == i]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                ax2.text(
                    center[0],
                    center[1],
                    f'Cluster {i}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=12,
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                )

        # Add colorbar
        legend = ax2.legend(*scatter.legend_elements(),
                            loc="upper right", title="Clusters")
        ax2.add_artist(legend)

        # Add cluster sizes annotation
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        size_text = "Cluster sizes:\n" + "\n".join([f"Cluster {i}: {size} samples"
                                                    for i, size in cluster_sizes.items()])
        ax2.text(
            0.02, 0.02,
            size_text,
            transform=ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=9,
            verticalalignment='bottom'
        )

        # Add explained variance if PCA was applied
        if data.shape[1] > 2:
            ax2.text(
                0.98, 0.02,
                f'Total explained variance:\n{sum(explained_var):.1f}%',
                transform=ax2.transAxes,
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.7),
                fontsize=9
            )

        # Adjust layout and show
        plt.tight_layout()
        fig.suptitle(plot_title, fontsize=16, y=1.05)

        # Display in Streamlit
        st.pyplot(fig)

        # Show silhouette score if we have more than one cluster
        if n_clusters > 1:
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(data, labels)
            st.write(f"Silhouette Score: {silhouette_avg:.3f}")

            # Interpret silhouette score
            if silhouette_avg > 0.5:
                st.success(
                    f"Good cluster separation (score: {silhouette_avg:.3f})")
            elif silhouette_avg > 0.25:
                st.info(
                    f"Reasonable cluster separation (score: {silhouette_avg:.3f})")
            else:
                st.warning(
                    f"Poor cluster separation (score: {silhouette_avg:.3f})")

    return labels, model


def hierarchical_clustering_with_options(data, n_clusters, original_data=None):
    """
    Perform hierarchical clustering with the specified number of clusters.

    Parameters:
    data: The data to cluster
    n_clusters: The number of clusters to form
    original_data: Original DataFrame (before scaling/PCA) to add cluster labels to

    Returns:
    tuple: (cluster_labels, cluster_model)
    """
    import streamlit as st
    from sklearn.cluster import AgglomerativeClustering
    import pandas as pd

    # Create column layout
    col1, col2 = st.columns(2)

    with col1:
        linkage_method = st.selectbox(
            "Linkage method",
            options=["ward", "complete", "average", "single"],
            index=0
        )

    with col2:
        show_dendrogram = st.checkbox("Show dendrogram", value=True)

    # Perform clustering when button is clicked
    if st.button("Run Hierarchical Clustering"):
        with st.spinner("Running hierarchical clustering..."):
            # Perform hierarchical clustering
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                compute_distances=True if show_dendrogram else False
            )

            labels = model.fit_predict(data)

            # Create DataFrame with cluster assignments
            if original_data is not None:
                # Use original data and add cluster column
                clustered_df = original_data.copy()
                clustered_df['Cluster'] = labels
            else:
                # Create DataFrame from the scaled/PCA data
                if hasattr(data, 'columns'):
                    # If data is already a DataFrame
                    clustered_df = data.copy()
                    clustered_df['Cluster'] = labels
                else:
                    # If data is a numpy array
                    clustered_df = pd.DataFrame(
                        data,
                        columns=[
                            f'Feature_{i+1}' for i in range(data.shape[1])]
                    )
                    clustered_df['Cluster'] = labels

            # Store results in session state
            st.session_state.clusters = labels
            st.session_state.hc_model = model
            st.session_state.hc_clusters = n_clusters
            st.session_state.clustered_data = clustered_df  # Save the DataFrame with clusters

            # Show dendrogram if requested
            if show_dendrogram:
                plot_dendrogram(model, data)

            # Plot clusters
            plot_clusters(data, labels, n_clusters)

            return labels, model

    # Return None if button not clicked
    return None, None


def hc_summary(model, data, labels):
    """
    Display a summary of the Hierarchical Clustering results.

    Parameters:
    model: The fitted AgglomerativeClustering model
    data: The input data
    labels: The cluster labels
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    # Basic model information
    st.subheader("Hierarchical Clustering Summary")

    # Get number of clusters
    n_clusters = len(np.unique(labels))

    # Display cluster distribution with a pie chart
    fig, ax = plt.subplots(figsize=(5, 5))

    # Count samples in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100

    # Create pie chart
    ax.pie(
        counts,
        labels=[f"Cluster {i}" for i in range(n_clusters)],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * n_clusters  # Slight explode effect
    )
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    plt.title('Cluster Size Distribution')

    # Display the pie chart
    st.pyplot(fig)

    # Show metrics in columns
    col1, col2, col3 = st.columns(3)

    # Only calculate these metrics if we have multiple clusters
    if n_clusters > 1:
        try:
            # Calculate silhouette score
            silhouette_avg = silhouette_score(data, labels)
            with col1:
                st.metric(
                    "Silhouette Score",
                    f"{silhouette_avg:.3f}",
                    help="Range [-1,1]. Higher is better. >0.5 is good clustering."
                )

            # Calculate Davies-Bouldin Index
            with col2:
                db_score = davies_bouldin_score(data, labels)
                st.metric(
                    "Davies-Bouldin Index",
                    f"{db_score:.3f}",
                    help="Lower is better. Values closer to 0 indicate better clustering."
                )

            # Calculate Calinski-Harabasz Index
            with col3:
                ch_score = calinski_harabasz_score(data, labels)
                st.metric(
                    "Calinski-Harabasz Index",
                    f"{ch_score:.3f}",
                    help="Higher is better. Higher values indicate better clustering."
                )

            # Interpret silhouette score
            if silhouette_avg > 0.5:
                st.success(
                    f"Good cluster separation (silhouette: {silhouette_avg:.3f})")
            elif silhouette_avg > 0.25:
                st.info(
                    f"Reasonable cluster separation (silhouette: {silhouette_avg:.3f})")
            else:
                st.warning(
                    f"Poor cluster separation (silhouette: {silhouette_avg:.3f})")
        except Exception as e:
            st.warning(f"Could not calculate clustering metrics: {str(e)}")

    # Display linkage information
    st.write(f"Linkage method: {model.linkage}")

    # Display cluster details in an expandable section
    with st.expander("Cluster Details", expanded=False):
        # Create a DataFrame with cluster assignments
        cluster_df = pd.DataFrame({
            'Cluster': labels
        })

        # Show counts by cluster
        st.write("### Sample count per cluster")
        st.write(cluster_df['Cluster'].value_counts().sort_index())

        # Calculate stats per cluster if original data columns are available
        if hasattr(st.session_state, 'df'):
            original_df = st.session_state.df
            if len(original_df) == len(labels):
                # Add cluster assignments to original data
                cluster_data = original_df.copy()
                cluster_data['Cluster'] = labels

                # Group by cluster and calculate mean
                st.write("### Mean values per cluster")
                st.dataframe(cluster_data.groupby('Cluster').mean())

    summary = {
        "metrics": {
            "Silhouette Score": silhouette_avg if n_clusters > 1 else None,
            "Davies-Bouldin Index": db_score if n_clusters > 1 else None,
            "Calinski-Harabasz Index": ch_score if n_clusters > 1 else None,

        },
        "n_clusters":  n_clusters,
        "linkage_method": model.linkage,
        # Convert numpy integers to Python integers in the cluster distribution
        "cluster_distribution": {
            int(label): int(count) for label, count in zip(unique_labels, counts)
        }
    }

    # Store summary in session state
    st.session_state.summary = summary

    return model
