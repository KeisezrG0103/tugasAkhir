import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA


def data_loader(path_file):
    """
    Load data from a CSV file and return a DataFrame.
    """
    df = pd.read_csv(path_file)
    return df


def column_to_lowercase(df):
    """
    Convert all column names in the DataFrame to lowercase.
    """
    df.columns = [col.lower() for col in df.columns]
    return df


def reverse_coding(x):
    return 6 - x


def reverse_coding_columns(df, columns):
    for column in columns:
        df[column] = df[column].apply(reverse_coding)
    return df


def PreprocessFinancialKnowledge(value, answer):
    if value == answer:
        return 1
    else:
        return 0


def Preprocess_Data(df):
    # Check if data has already been processed
    financial_columns = ['fk1', 'fk2', 'fk3', 'fk4',
                         'fk5', 'fk6', 'fk7', 'fk8', 'fk9', 'fk10']
    # If any of the financial columns have numeric values, data has been processed
    sample_col = financial_columns[0].lower()
    if sample_col in df.columns and pd.api.types.is_numeric_dtype(df[sample_col]):
        print("Data already processed, skipping string operations")
        return df

    df.columns = df.columns.str.lower()
    fa_columns = ['fa8', 'fa9', 'fa10', 'fa11', 'fa12', 'fa13', 'fa14', 'fa15']
    m_columns = ['m2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']

    # Define the answers in a dictionary for faster lookup
    financial_answers = {
        'fk1': 'lebih dari rp150.000',
        'fk2': 'lebih rendah dari hari ini',
        'fk3': '6%',
        'fk4': 'toko a',
        'fk5': 'rp20.000',
        'fk6': 'saham',
        'fk7': 'saham',
        'fk8': 'mengurangi risiko',
        'fk9': 'benar',
        'fk10': 'salah'
    }

    # Convert column names to lowercase
    column_to_lowercase(df)

    # Drop unnecessary columns
    df = df.drop(columns=['timestamp', 'email'], errors='ignore')

    # Apply reverse coding to relevant columns
    df = reverse_coding_columns(df, fa_columns)
    df = reverse_coding_columns(df, m_columns)

    # Convert financial columns to lowercase
    financialColumns = list(financial_answers.keys())
    # Only apply string operations if columns contain string data
    for col in financialColumns:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.lower()

    # Process all financial knowledge questions in one step
    for column, answer in financial_answers.items():
        df[column] = (df[column] == answer).astype(int)

    # Calculate total financial knowledge
    df['total_fk'] = df[financialColumns].sum(axis=1)

    # Compare with score
    df['comparison'] = df['total_fk'] == df['score']

    print(df[['nama', 'total_fk', 'score', 'comparison']])

    df.drop(columns=['comparison', 'nama', 'score',
            'total_fk', 'umur'], inplace=True)

    return df


def add_vif_analysis(df):
    st.subheader("Variance Inflation Factor (VIF) Analysis")

    st.write("""
    Variance Inflation Factor (VIF) measures how much the variance of a regression coefficient is inflated due to multicollinearity.
    - VIF = 1: No multicollinearity
    - VIF between 1-5: Moderate multicollinearity
    - VIF > 5: High multicollinearity
    - VIF > 10: Very high multicollinearity that may be problematic
    """)

    # Create a copy of the dataframe without non-numeric columns
    df_data_copy = df.copy()

    # Remove non-numeric and unnecessary columns
    df_data_copy = df_data_copy.drop(
        columns=['nama', 'umur', 'total_fk', 'comparison', 'score'])

    # Drop rows with missing values
    df_data_copy = df_data_copy.dropna()

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_data_copy.columns

    # Calculate VIF using statsmodels
    X = df_data_copy.values
    vif_data["VIF"] = [variance_inflation_factor(
        X, i) for i in range(X.shape[1])]

    # Sort by VIF value
    vif_data = vif_data.sort_values("VIF", ascending=False)

    # Display the VIF values
    st.dataframe(vif_data)

    # Plot VIF values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(vif_data["Feature"], vif_data["VIF"])
    ax.axhline(y=5, color='r', linestyle='-',
               label="High Multicollinearity Threshold (5)")
    ax.axhline(y=10, color='darkred', linestyle='-',
               label="Very High Multicollinearity Threshold (10)")
    ax.set_title("Variance Inflation Factor for Each Feature")
    ax.set_xlabel("Features")
    ax.set_ylabel("VIF")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Create summary of multicollinearity
    high_vif = vif_data[vif_data["VIF"] > 5]
    if not high_vif.empty:
        st.write(
            f"Found {len(high_vif)} features with high multicollinearity (VIF > 5):")
        st.dataframe(high_vif)

        st.write("""
        **Recommendations for high VIF values:**
        1. Remove features with the highest VIF values
        2. Combine features using dimensionality reduction (PCA)
        3. Create a new feature from the correlated features
        """)
    else:
        st.write("No features with high multicollinearity detected.")


def pairwise_multicollinearity(df, threshold=0.8):
    """
    Calculate the pairwise correlation between features in the DataFrame
    and return pairs with correlation above the specified threshold.
    """
    corr_matrix = df.corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                high_corr_pairs.append((colname, corr_matrix.columns[j],
                                        corr_matrix.iloc[i, j]))

    return high_corr_pairs


def plot_correlation_matrix(df, threshold=0.8):
    """
    Plot the correlation matrix of the DataFrame and highlight pairs with
    correlation above the specified threshold.
    """

    corr_mat = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.matshow(corr_mat, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_xticks(range(len(corr_mat.columns)))
    ax.set_yticks(range(len(corr_mat.columns)))
    ax.set_xticklabels(corr_mat.columns, rotation=90)
    ax.set_yticklabels(corr_mat.columns)

    plt.title("Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)

    high_corr_pairs = pairwise_multicollinearity(df, threshold)
    if high_corr_pairs:
        st.error(f"Ada fitur yang memiliki korelasi lebih dari {threshold}.\n")
        # for pair in high_corr_pairs:
        #     st.error(f"Fitur {pair[0]} dengan {pair[1]}: {pair[2]}")
        # Display pairs in a table
        high_corr_df = pd.DataFrame(
            high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
        high_corr_df = high_corr_df.sort_values(
            by="Correlation", ascending=False).reset_index(drop=True)
        st.dataframe(high_corr_df)
        st.write(
            f"Fitur dengan korelasi lebih dari {threshold}:\n")

    else:
        st.info(
            f"Tidak ada fitur yang memiliki korelasi lebih dari {threshold}.")


def graph_pairwise_correlation(df, threshold=0.8):
    if any(col in df.columns for col in ['nama', 'score', 'total_fk']):
        df = df.drop(columns=['nama', 'score', 'total_fk'], errors='ignore')
    fig_network, ax_network = plt.subplots(figsize=(10, 8))
    corr_mat = df.corr()
    # Create network graph
    G = nx.Graph()
    for col in corr_mat.columns:
        G.add_node(col)

    # Add edges for correlations above threshold
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i, j]) > threshold:
                G.add_edge(corr_mat.columns[i], corr_mat.columns[j],
                           weight=abs(corr_mat.iloc[i, j]),
                           color='red' if corr_mat.iloc[i, j] < 0 else 'blue')

    # Get edge properties
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=300,
                           node_color='lightblue', alpha=0.2, ax=ax_network)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax_network)
    nx.draw_networkx_edges(G, pos, width=edge_weights,
                           edge_color=edge_colors, alpha=0.7, ax=ax_network)

    plt.title(
        f"Fitur Pairwise Correlation Network dengan Threshold {threshold}")
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig_network)


def perform_pca(dataframe, scaler_type, n_components):
    """
    Perform PCA on the input dataframe.

    Args:
        dataframe: Input dataframe
        scaler_type: Type of scaler to use (StandardScaler, MinMaxScaler, RobustScaler)
        n_components: Number of PCA components

    Returns:
        scaled_data: Original scaled data
        pca_result: PCA transformed data
        explained_variance: Explained variance ratio
        cumulative_variance: Cumulative explained variance
    """
    # Create the appropriate scaler
    if scaler_type == "StandardScaler":
        scaler_obj = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler_obj = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler_obj = RobustScaler()

        # Scale the data first
    scaled_data = scaler_obj.fit_transform(dataframe)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    return scaled_data, pca_result, explained_variance, cumulative_variance


def one_way_anova(df, feature, group_col='Cluster', label_col=None, algorithm_name=None):
    """
    Perform one-way ANOVA test to determine if there are significant differences
    in feature values across clusters.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data with cluster assignments
    feature : str
        The feature to analyze
    group_col : str, default='Cluster'
        The column name containing cluster assignments
    label_col : str, optional
        Column with label information (if available)
    algorithm_name : str, optional
        Name of the clustering algorithm used

    Returns:
    --------
    dict: Dictionary with ANOVA results
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    import streamlit as st
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Ensure group column is properly formatted
    if group_col not in df.columns:
        if 'Cluster' in df.columns:
            group_col = 'Cluster'
        elif 'cluster' in df.columns:
            group_col = 'cluster'
        else:
            raise ValueError(
                f"Group column '{group_col}' not found in DataFrame")

    # Prepare data for ANOVA
    groups = df[group_col].unique()
    n_groups = len(groups)

    # Dictionary to store group data
    group_data = {f"Cluster {g}": df[df[group_col] == g][feature].values
                  for g in groups}

    # Basic statistics per group
    group_stats = {
        f"Cluster {g}": {
            "count": len(df[df[group_col] == g][feature]),
            "mean": df[df[group_col] == g][feature].mean(),
            "std": df[df[group_col] == g][feature].std(),
            "min": df[df[group_col] == g][feature].min(),
            "max": df[df[group_col] == g][feature].max()
        } for g in groups
    }

    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(
        *[group_data[f"Cluster {g}"] for g in groups])

    # Prepare result dictionary
    result = {
        "feature": feature,
        "algorithm": algorithm_name,
        "f_statistic": f_statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "group_stats": group_stats,
        "n_groups": n_groups
    }

    # If ANOVA is significant, perform Tukey's HSD post-hoc test
    if p_value < 0.05:
        # Prepare data for Tukey test
        data = []
        groups_labels = []

        for g in groups:
            values = df[df[group_col] == g][feature].values
            data.extend(values)
            groups_labels.extend([f"Cluster {g}"] * len(values))

        # Perform Tukey's HSD
        tukey_result = pairwise_tukeyhsd(
            endog=data,
            groups=groups_labels,
            alpha=0.05
        )

        # Add Tukey results to the dictionary
        result["tukey"] = {
            "reject": tukey_result.reject.tolist(),
            "pvalues": tukey_result.pvalues.tolist(),
            "meandiffs": tukey_result.meandiffs.tolist(),
            "groups": [(tukey_result.groupsunique[i], tukey_result.groupsunique[j])
                       for i in range(len(tukey_result.groupsunique))
                       for j in range(i+1, len(tukey_result.groupsunique))]
        }

    return result
