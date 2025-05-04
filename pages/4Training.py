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

    st.write(st.session_state.summary)


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
        st.write(st.session_state.summary)


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
    st.write(st.session_state.summary)
