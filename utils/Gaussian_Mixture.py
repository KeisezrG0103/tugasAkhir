from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def bic_calculation(K=10, data=None):
    if data is None:
        raise ValueError("Data must be provided to calculate BIC.")

    BIC = []
    for i in range(K):
        gmm = GaussianMixture(n_components=i + 1, covariance_type='full')
        gmm.fit(data)
        BIC.append(gmm.bic(data))
    return BIC


def plot_bic(BIC, K=None):
    """
    Plot BIC values for different numbers of components in a Gaussian Mixture Model.

    Parameters:
    BIC: List of BIC values
    K: Number of components (optional, will be inferred from BIC length if not provided)

    Returns:
    matplotlib figure that can be displayed in Streamlit
    """

    # Infer K from BIC length if not provided
    if K is None:
        K = len(BIC)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot BIC values
    ax.plot(range(1, K + 1), BIC, marker='o', linestyle='-', color='blue')

    # Find the minimum BIC value
    min_bic_idx = BIC.index(min(BIC))
    optimal_components = min_bic_idx + 1

    # Highlight the minimum BIC point
    ax.plot(optimal_components, BIC[min_bic_idx], 'ro', markersize=12,
            label=f'Optimal Components: {optimal_components}')

    # Add a vertical line at the optimal point
    ax.axvline(x=optimal_components, color='r', linestyle='--', alpha=0.3)

    # Add annotation
    ax.annotate(f'Minimum BIC: {BIC[min_bic_idx]:.2f}',
                xy=(optimal_components, BIC[min_bic_idx]),
                xytext=(optimal_components + 0.5, BIC[min_bic_idx] * 1.05),
                arrowprops=dict(facecolor='black', shrink=0.05,
                                width=1.5, headwidth=8),
                fontsize=10)

    # Customize the plot
    ax.set_title('BIC for Gaussian Mixture Models')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('BIC (lower is better)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(range(1, K + 1))
    ax.legend()

    # Display in Streamlit
    st.pyplot(fig)

    # Return the optimal number of components
    st.write(
        f"According to BIC, the optimal number of components is: {optimal_components}")

    return optimal_components


def gmm_fit(data, n_components, original_data=None):
    """
    Fit a Gaussian Mixture Model (GMM) to the data.

    Parameters:
    data: Data to fit the GMM
    n_components: Number of components for the GMM

    Returns:
    gmm: Fitted GMM object
    """

    import pandas as pd

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full', random_state=42)
    gmm.fit(data)

    labels = gmm.predict(data)

    # Create DataFrame from input data
    if original_data is not None:
        # Use original data but add cluster labels
        result_df = original_data.copy()
        result_df['cluster'] = labels
    else:
        # Create DataFrame from the input data
        result_df = pd.DataFrame(data)
        result_df['cluster'] = labels

    # save to state
    st.session_state.clustered_data = result_df
    st.session_state.current_algorithm = 'GMM'

    return gmm


def gmm_plot(data, n_components, gmm_model):
    """
    Plot the Gaussian Mixture Model results.

    Parameters:
    data: The data used for fitting (original high-dimensional data)
    n_components: Number of components in the model
    gmm_model: Fitted GaussianMixture model
    """
    import streamlit as st
    from sklearn.decomposition import PCA

    # Get cluster assignments
    labels = gmm_model.predict(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # If data has more than 2 dimensions, use PCA for visualization only
    if data.shape[1] > 2:
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        # Get explained variance
        explained_var = pca.explained_variance_ratio_ * 100

        # Plot the data points colored by cluster
        scatter = ax.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='w'
        )

        # Add labels and title for PCA plot
        ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1f}%)')
        ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1f}%)')
        ax.set_title(f'GMM Clustering (k={n_components}) - PCA Visualization')

        # Note that we're using PCA
        ax.text(
            0.98, 0.98,
            f'Using PCA projection\nTotal variance explained: {sum(explained_var):.1f}%',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    else:
        # For 2D data, we can plot the actual GMM contours
        scatter = ax.scatter(
            data[:, 0],
            data[:, 1],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='w'
        )

        # Create a mesh grid to plot the GMM contours
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Get log probabilities on the mesh grid
        try:
            Z = gmm_model.score_samples(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot the contours
            ax.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7),
                       alpha=0.3, colors='k', linestyles='dashed')

            # Plot filled contours
            ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7),
                        alpha=0.1, cmap='viridis')
        except:
            st.warning(
                "Unable to plot GMM contours due to dimensionality issues")

        # Add labels for 2D plot
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Gaussian Mixture Model (k={n_components})')

    # Add colorbar for cluster labels
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')

    # Add cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_percentages = counts / counts.sum() * 100

    # Format the cluster size information
    size_text = "Cluster distribution:\n"
    for i, (count, percentage) in enumerate(zip(counts, cluster_percentages)):
        size_text += f"Cluster {i}: {count} samples ({percentage:.1f}%)\n"

    # Add text box with cluster information
    ax.text(
        0.02, 0.02,
        size_text,
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7),
        fontsize=9,
        verticalalignment='bottom'
    )

    # Add grid and layout improvements
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

    # Return the labels for further use
    return labels


def GMM_summary(gmm_model, data=None):
    """
    Display summary information about a fitted GMM model.

    Parameters:
    gmm_model: Fitted GaussianMixture model
    data: Original data used to fit the model (needed for silhouette score)
    """
    import streamlit as st
    from sklearn.metrics import silhouette_score

    # Basic model information
    st.subheader("GMM Model Summary")

    # Number of components
    st.write(f"Number of components: {gmm_model.n_components}")

    # Covariance type
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        gmm_model.weights_,
        labels=[f"Component {i}" for i in range(gmm_model.n_components)],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        # Slight explode effect for all slices
        explode=[0.05] * gmm_model.n_components
    )
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    plt.title('GMM Component Weights')

    # Display the pie chart in Streamlit
    st.pyplot(fig)

    # Display component means
    import pandas as pd
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    # Calculate clustering metrics if data is provided
    if data is not None:
        # Get cluster labels
        labels = gmm_model.predict(data)

        # Only calculate metrics if there are multiple clusters and sufficient samples
        if gmm_model.n_components > 1 and data.shape[0] > gmm_model.n_components:
            try:
                # Create columns for metrics display
                # Add this line to define columns
                col1, col2, col3 = st.columns(3)

                # Calculate silhouette score
                silhouette_avg = silhouette_score(data, labels)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette_avg:.3f}",
                              help="Range [-1,1]. Higher is better. >0.5 is good clustering.")

                # Calculate Davies-Bouldin Index
                with col2:
                    db_score = davies_bouldin_score(data, labels)
                    st.metric("Davies-Bouldin Index", f"{db_score:.3f}",
                              help="Lower is better. Values closer to 0 indicate better clustering.")

                # Calculate Calinski-Harabasz Index
                with col3:
                    ch_score = calinski_harabasz_score(data, labels)
                    st.metric("Calinski-Harabasz Index", f"{ch_score:.3f}",
                              help="Higher is better. Higher values indicate better clustering.")

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
    else:
        st.warning("Data not provided - cannot calculate clustering metrics")

    summary = {
        "metrics": {
            "silhouette_score": silhouette_avg if data is not None else None,
            "davies_bouldin_score": db_score if data is not None else None,
            "calinski_harabasz_score": ch_score if data is not None else None
        },
        "n_clusters": int(gmm_model.n_components),
        "weights": gmm_model.weights_,
        "means": gmm_model.means_,
        "covariances": gmm_model.covariances_,
        "converged": gmm_model.converged_,
        "cluster_distribution": {
            f"Cluster {i}": int(count)
            for i, count in enumerate(gmm_model.weights_ * data.shape[0])
        }
    }

    st.write("Saving summary to session state...")

    st.session_state.summary = summary

    return gmm_model
