import kneed
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import json


def elbow_method(X, max_k=10, plot=True):
    """
    Apply the elbow method to find the optimal number of clusters for KMeans.

    Parameters:
    X (array-like): The input data.
    max_k (int): The maximum number of clusters to consider.
    plot (bool): Whether to display the elbow plot.

    Returns:
    int: The optimal number of clusters.
    """
    import streamlit as st  # Add this import

    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Use the KneeLocator to find the elbow point
    kn = KneeLocator(range(1, max_k + 1), sse,
                     curve='convex', direction='decreasing')

    # Plot the elbow curve if requested
    if plot:
        fig = Kmeans_plot(max_k, sse, kn)  # Remove plt parameter
        st.pyplot(fig)  # Use st.pyplot() to display the figure

    return kn.elbow


def Kmeans_plot(max_k, sse, kn):  # Remove plt parameter
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes
    ax.plot(range(1, max_k + 1), sse, 'bo-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Sum of Squared Distances (SSE)')
    ax.set_title('Elbow Method for Optimal k')

    # Highlight the elbow point if found
    if kn.elbow is not None:
        ax.plot(kn.elbow, sse[kn.elbow-1], 'ro', markersize=12,
                label=f'Elbow Point (k={kn.elbow})')
        ax.axvline(x=kn.elbow, color='r', linestyle='--', alpha=0.3)
        ax.legend()

    ax.grid(True)
    # plt.show()  # Remove this line
    return fig  # Return the figure instead


def kmeans_clustering(X, n_clusters, plot=True, use_pca_for_viz=True, original_data=None, feature_names=None):
    """
    Apply KMeans clustering to the input data.

    Parameters:
    X (array-like): The input data.
    n_clusters (int): The number of clusters.
    plot (bool): Whether to display the clustering plot.
    use_pca_for_viz (bool): Whether to use PCA for visualization of high-dimensional data.

    Returns:
    tuple: The cluster labels and the KMeans model.
    """
    import streamlit as st
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    # Apply KMeans on original data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Create DataFrame from input data
    if original_data is not None:
        # Use original data but add cluster labels
        result_df = original_data.copy()
        result_df['cluster'] = labels
    else:
        # Create DataFrame from X
        result_df = pd.DataFrame(X, columns=feature_names)
        result_df['cluster'] = labels

    # save to streamlit session state

    st.session_state.clustered_data = result_df

    st.session_state.current_algorithm = "KMeans"

    # Display visualization if requested

    if plot:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for visualization
        if X.shape[1] > 2 and use_pca_for_viz:
            # Apply PCA for visualization only
            pca = PCA(n_components=2)
            X_viz = pca.fit_transform(X)

            # Plot in PCA space
            scatter = ax.scatter(
                X_viz[:, 0], X_viz[:, 1], c=labels, cmap='viridis', alpha=0.7)

            # Transform centroids to PCA space for visualization
            centroids_viz = pca.transform(kmeans.cluster_centers_)

            # Add centroids
            ax.scatter(
                centroids_viz[:, 0],
                centroids_viz[:, 1],
                s=300, c='red', marker='X',
                label='Centroids',
                edgecolors='black'
            )

            # Add titles and labels
            ax.set_title(
                f'KMeans Clustering (k={n_clusters}) - PCA Visualization')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')

            # Add explained variance information
            explained_var = pca.explained_variance_ratio_
            ax.text(0.95, 0.05, f'Explained variance:\nPC1: {explained_var[0]:.2%}\nPC2: {explained_var[1]:.2%}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        elif X.shape[1] >= 2:
            # Original plot for 2D data
            scatter = ax.scatter(
                X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)

            # Add centroids
            ax.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                s=300, c='red', marker='X',
                label='Centroids',
                edgecolors='black'
            )

            ax.set_title(f'KMeans Clustering (k={n_clusters})')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

        else:
            # Not enough dimensions to plot
            ax.text(0.5, 0.5, "Data does not have enough dimensions for plotting",
                    horizontalalignment='center', verticalalignment='center')

        # Common visualization elements
        if X.shape[1] >= 2:
            # Add legend for centroids
            ax.legend()

            # Add a colorbar for cluster labels
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster')

            # Add cluster size annotations
            for i in range(n_clusters):
                cluster_size = (labels == i).sum()
                cluster_percent = cluster_size / len(labels) * 100
                ax.annotate(f'Cluster {i}: {cluster_size} points ({cluster_percent:.1f}%)',
                            xy=(0.02, 0.98 - i*0.05), xycoords='axes fraction',
                            fontsize=9, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            ax.grid(True, linestyle='--', alpha=0.7)

        # Display in Streamlit
        st.pyplot(fig)

    # Always return these values
    return labels, kmeans


def kmeans_summary(kmeans, X):
    """
    Display a summary of the KMeans clustering results.

    Parameters:
    kmeans (KMeans): The fitted KMeans model.
    X (array-like): The input data.
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    # Basic model information
    st.subheader("KMeans Clustering Summary")

    # Get the cluster labels
    labels = kmeans.labels_

    # Display cluster distribution with a pie chart
    fig, ax = plt.subplots(figsize=(5, 5))

    # Count samples in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100

    # Create pie chart
    ax.pie(
        counts,
        labels=[f"Cluster {i}" for i in range(kmeans.n_clusters)],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * kmeans.n_clusters  # Slight explode effect
    )
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    plt.title('Cluster Size Distribution')

    # Display the pie chart
    st.pyplot(fig)

    # Show metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # Display inertia (within-cluster sum of squares)
        st.metric(
            "Inertia",
            f"{kmeans.inertia_:.2f}",
            help="Sum of squared distances to closest centroid. Lower is better."
        )

    # Only calculate these metrics if we have multiple clusters
    if kmeans.n_clusters > 1:
        try:
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, labels)
            with col2:
                st.metric(
                    "Silhouette Score",
                    f"{silhouette_avg:.3f}",
                    help="Range [-1,1]. Higher is better. >0.5 is good clustering."
                )

            # Calculate Davies-Bouldin Index
            with col3:
                db_score = davies_bouldin_score(X, labels)
                st.metric(
                    "Davies-Bouldin Index",
                    f"{db_score:.3f}",
                    help="Lower is better. Values closer to 0 indicate better clustering."
                )

            # Additional metric in another row
            ch_score = calinski_harabasz_score(X, labels)
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

    # Display convergence information
    st.write(f"Number of iterations until convergence: {kmeans.n_iter_}")

    # Display the cluster centers in an expandable section
    with st.expander("Cluster Centers", expanded=False):
        # Create a DataFrame for better visualization
        centers_df = pd.DataFrame(
            kmeans.cluster_centers_,
            index=[f"Cluster {i}" for i in range(kmeans.n_clusters)]
        )
        st.dataframe(centers_df)

    # Save the model to session state
    summary = {
        "metrics": {
            "inertia": float(kmeans.inertia_),  # Convert to float
            "silhouette_score": float(silhouette_avg) if kmeans.n_clusters > 1 else None,
            "davies_bouldin_score": float(db_score) if kmeans.n_clusters > 1 else None,
            "calinski_harabasz_score": float(ch_score) if kmeans.n_clusters > 1 else None,
        },
        "n_clusters": int(kmeans.n_clusters),  # Convert to int
        # Convert numpy integers to Python integers in the cluster distribution
        "cluster_distribution": {int(label): int(count) for label, count in zip(unique_labels, counts)},
    }
    st.session_state.summary = summary

    return kmeans
