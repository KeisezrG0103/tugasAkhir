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


def detect_outliers_with_zscore(df, threshold=3.0):
    """
    Detect outliers in a DataFrame using Z-score method.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the numerical data to analyze for outliers
    threshold : float, default=3.0
        The Z-score threshold to identify outliers (typically 2.5-3.0)

    Returns:
    --------
    outlier_stats : dict
        Dictionary containing outlier statistics for each column
    outlier_df : pandas DataFrame
        DataFrame with boolean flags for outliers in each column
    outlier_count_df : pandas DataFrame
        Summary of outlier counts per column
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    # Create a copy of the dataframe to avoid modifying the original
    df_numeric = df.select_dtypes(include=np.number).copy()

    if df_numeric.empty:
        st.error("No numeric columns found in the dataset.")
        return None, None, None

    # Calculate Z-scores for each numeric column
    z_scores = pd.DataFrame()
    for col in df_numeric.columns:
        z_scores[col] = (df_numeric[col] - df_numeric[col].mean()
                         ) / df_numeric[col].std()

    # Create a DataFrame of boolean values indicating outliers (absolute Z-score > threshold)
    outliers = pd.DataFrame()
    for col in z_scores.columns:
        outliers[col] = abs(z_scores[col]) > threshold

    # Count outliers per column
    outlier_counts = outliers.sum().sort_values(ascending=False)
    outlier_percentages = (outlier_counts / len(df)) * 100

    # Create a summary dataframe of outlier counts and percentages
    outlier_summary = pd.DataFrame({
        'Column': outlier_counts.index,
        'Outlier Count': outlier_counts.values,
        'Percentage': outlier_percentages.values,
        'Total Rows': len(df)
    }).sort_values('Outlier Count', ascending=False)

    # Collect statistics for each column
    outlier_stats = {}
    for col in df_numeric.columns:
        # Get the outlier values
        outlier_values = df_numeric.loc[outliers[col], col]

        # Calculate statistics
        outlier_stats[col] = {
            'count': int(outlier_counts[col]),
            'percentage': float(outlier_percentages[col]),
            'min_value': float(outlier_values.min()) if not outlier_values.empty else None,
            'max_value': float(outlier_values.max()) if not outlier_values.empty else None,
            'mean': float(df_numeric[col].mean()),
            'std': float(df_numeric[col].std()),
            'normal_range': (
                float(df_numeric[col].mean() -
                      threshold * df_numeric[col].std()),
                float(df_numeric[col].mean() +
                      threshold * df_numeric[col].std())
            )
        }

    return outlier_stats, outliers, outlier_summary


def visualize_zscore_outliers(df, outlier_stats, outliers, threshold=3.0):
    """
    Visualize outliers detected using Z-score method.

    Parameters:
    -----------
    df : pandas DataFrame
        Original DataFrame with data
    outlier_stats : dict
        Dictionary containing outlier statistics from detect_outliers_with_zscore
    outliers : pandas DataFrame
        DataFrame with boolean flags for outliers from detect_outliers_with_zscore
    threshold : float, default=3.0
        The Z-score threshold used to identify outliers
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    st.subheader("Z-Score Outlier Analysis")

    st.write(f"""
    Z-score measures how many standard deviations a data point is from the mean.
    - Typical threshold: Z-score > {threshold} (outlier if more than {threshold} standard deviations from mean)
    - Higher threshold = more conservative outlier detection
    - Lower threshold = more aggressive outlier detection
    """)

    # Get numeric columns
    df_numeric = df.select_dtypes(include=np.number).copy()

    # Display summary statistics
    total_outliers = sum(stat['count'] for stat in outlier_stats.values())
    total_possible = len(df) * len(outlier_stats)
    overall_percentage = (total_outliers / total_possible) * \
        100 if total_possible > 0 else 0

    st.metric(
        label="Total Outliers Detected",
        value=f"{total_outliers}",
        delta=f"{overall_percentage:.2f}% of all data points"
    )

    # Create summary dataframe
    summary_data = []
    for col, stats in outlier_stats.items():
        summary_data.append({
            'Column': col,
            'Outlier Count': stats['count'],
            'Percentage': stats['percentage'],
            'Normal Range': f"[{stats['normal_range'][0]:.2f}, {stats['normal_range'][1]:.2f}]"
        })

    summary_df = pd.DataFrame(summary_data).sort_values(
        'Outlier Count', ascending=False)

    # Display summary table
    st.dataframe(summary_df)

    # Plot outlier counts
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary_df['Column'], summary_df['Outlier Count'])

    # Color the bars based on outlier percentage
    for i, bar in enumerate(bars):
        percentage = summary_df.iloc[i]['Percentage']
        if percentage > 10:
            bar.set_color('red')
        elif percentage > 5:
            bar.set_color('orange')
        else:
            bar.set_color('green')

    ax.set_title('Outlier Count by Column')
    ax.set_xlabel('Features')
    ax.set_ylabel('Number of Outliers')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

    # Allow user to select columns for detailed visualization
    if len(outlier_stats) > 0:
        selected_columns = st.multiselect(
            "Select columns for detailed outlier visualization",
            options=list(outlier_stats.keys()),
            default=list(outlier_stats.keys())[:min(3, len(outlier_stats))]
        )

        if selected_columns:
            for col in selected_columns:
                # Create box plot for the selected column
                fig, ax = plt.subplots(figsize=(10, 5))

                # Create box plot with outliers highlighted
                sns.boxplot(x=df_numeric[col], ax=ax)

                # Add range lines
                lower_bound, upper_bound = outlier_stats[col]['normal_range']
                ax.axvline(x=lower_bound, color='r', linestyle='--',
                           label=f'Lower bound ({lower_bound:.2f})')
                ax.axvline(x=upper_bound, color='r', linestyle='--',
                           label=f'Upper bound ({upper_bound:.2f})')

                ax.set_title(f'Box Plot of {col} with Outlier Bounds')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Display histogram with outliers highlighted
                fig, ax = plt.subplots(figsize=(10, 5))

                # Plot regular data points
                sns.histplot(
                    df_numeric.loc[~outliers[col], col],
                    ax=ax,
                    color='blue',
                    label='Normal data'
                )

                # Plot outlier data points
                if outlier_stats[col]['count'] > 0:
                    sns.histplot(
                        df_numeric.loc[outliers[col], col],
                        ax=ax,
                        color='red',
                        label='Outliers'
                    )

                ax.set_title(
                    f'Distribution of {col} with Outliers Highlighted')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Display outlier values for this column
                if outlier_stats[col]['count'] > 0:
                    st.write(f"Outlier values for {col}:")
                    outlier_values = df_numeric.loc[outliers[col], col].sort_values(
                    )
                    st.write(outlier_values)

    # Recommendations based on outlier analysis
    st.subheader("Recommendations for Outlier Handling")

    high_outlier_cols = [
        col for col, stats in outlier_stats.items() if stats['percentage'] > 5]

    if high_outlier_cols:
        st.warning(
            f"Features with high outlier percentages (>5%): {', '.join(high_outlier_cols)}")
        st.write("""
        **Options for handling outliers:**
        1. **Keep them**: If they represent valid but rare data points
        2. **Remove them**: If they represent errors or are not relevant to your analysis
        3. **Transform them**: Apply techniques like log transformation to reduce their impact
        4. **Cap them**: Use winsorization to cap extreme values at a specified percentile
        5. **Treat them separately**: Create separate models for outlier and non-outlier data
        """)
    else:
        st.success("No features have a high percentage of outliers.")


def process_with_outlier_detection(df):
    """
    Process data with Z-score outlier detection and visualization.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process

    Returns:
    --------
    df_processed : pandas DataFrame
        Processed DataFrame with outlier information
    """
    st.header("Outlier Detection with Z-Score")

    # Select Z-score threshold
    threshold = st.slider(
        "Z-score threshold for outlier detection",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Higher values are more conservative (fewer outliers detected)"
    )

    # Detect outliers
    outlier_stats, outliers, outlier_summary = detect_outliers_with_zscore(
        df, threshold)

    if outlier_stats:
        # Visualize outliers
        visualize_zscore_outliers(df, outlier_stats, outliers, threshold)

        # Option to handle outliers
        outlier_handling = st.radio(
            "How would you like to handle outliers?",
            ["Keep all data", "Remove outliers",
                "Cap outliers (Winsorization)"]
        )

        if outlier_handling == "Remove outliers":
            # Calculate which rows to keep (rows with no outliers in any column)
            rows_with_outliers = outliers.any(axis=1)
            cleaned_df = df[~rows_with_outliers]

            removed_count = len(df) - len(cleaned_df)
            st.warning(
                f"Removed {removed_count} rows ({removed_count/len(df)*100:.2f}% of data) containing outliers")

            return cleaned_df

        elif outlier_handling == "Cap outliers (Winsorization)":
            # Cap outliers at the threshold value
            winsorized_df = df.copy()

            for col in outlier_stats.keys():
                lower_bound, upper_bound = outlier_stats[col]['normal_range']
                winsorized_df.loc[df[col] < lower_bound, col] = lower_bound
                winsorized_df.loc[df[col] > upper_bound, col] = upper_bound

            st.success("Capped outliers at Z-score threshold boundaries")
            return winsorized_df

    # Default return the original dataframe
    return df


def test_normality(df):
    """
    Test whether the data follows a normal distribution

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame with numeric columns to test

    Returns:
    --------
    normality_results : pandas DataFrame
        DataFrame with normality test results for each column
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import streamlit as st

    st.subheader("Normality Test")

    st.write("""
    Testing whether data follows a normal distribution using:
    1. **Shapiro-Wilk Test**: Good for small samples (n < 2000)
    2. **D'Agostino-Pearson Test**: Tests both skewness and kurtosis
    3. **Kolmogorov-Smirnov Test**: Non-parametric test for larger samples
    
    For each test, if p-value < 0.05, we reject the null hypothesis that the data is normally distributed.
    """)

    # Filter numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.warning("No numeric columns found for normality testing.")
        return None

    # Initialize results
    results = []

    # Test each column
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()

        # Skip if too few data points
        if len(data) < 3:
            results.append({
                'Column': column,
                'Shapiro_p_value': None,
                'DAgostino_p_value': None,
                'KS_p_value': None,
                'Normally_Distributed': "Insufficient data"
            })
            continue

        # Shapiro-Wilk test (best for n < 2000)
        if len(data) < 2000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            shapiro_p = np.nan  # Too many samples for Shapiro-Wilk

        # D'Agostino-Pearson test
        try:
            dagostino_stat, dagostino_p = stats.normaltest(data)
        except:
            dagostino_p = np.nan

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p = stats.kstest(
                data, 'norm', args=(data.mean(), data.std()))
        except:
            ks_p = np.nan

        # Determine if normally distributed (use most reliable test based on sample size)
        if len(data) < 2000:
            is_normal = shapiro_p > 0.05
        else:
            is_normal = dagostino_p > 0.05 or ks_p > 0.05

        results.append({
            'Column': column,
            'Shapiro_p_value': shapiro_p,
            'DAgostino_p_value': dagostino_p,
            'KS_p_value': ks_p,
            'Normally_Distributed': "Yes" if is_normal else "No"
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    st.dataframe(results_df.style.format({
        'Shapiro_p_value': "{:.4f}",
        'DAgostino_p_value': "{:.4f}",
        'KS_p_value': "{:.4f}"
    }).background_gradient(
        subset=['Shapiro_p_value', 'DAgostino_p_value', 'KS_p_value'],
        cmap='RdYlGn',
        low=0,
        high=0.1
    ))

    # Summary statistics
    normal_count = sum(
        1 for result in results if result['Normally_Distributed'] == "Yes")
    total_count = len(results)

    st.metric(
        label="Features with Normal Distribution",
        value=f"{normal_count}/{total_count}",
        delta=f"{normal_count/total_count*100:.1f}% of features" if total_count > 0 else "N/A"
    )

    # Visualize distributions
    st.subheader("Distribution Visualization")

    # Allow user to select columns to visualize
    selected_columns = st.multiselect(
        "Select columns to visualize distribution",
        options=numeric_df.columns,
        default=list(numeric_df.columns)[:min(3, len(numeric_df.columns))]
    )

    if selected_columns:
        for column in selected_columns:
            data = numeric_df[column].dropna()

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Histogram
            sns.histplot(data, kde=True, ax=ax1)
            ax1.set_title(f'Histogram of {column}')

            # Q-Q plot
            stats.probplot(data, plot=ax2)
            ax2.set_title(f'Q-Q Plot of {column}')

            plt.tight_layout()
            st.pyplot(fig)

            # Show distribution metrics
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skewness", f"{skewness:.4f}",
                          delta="Normal" if abs(skewness) < 0.5 else
                          "Slight Skew" if abs(skewness) < 1 else "Skewed")
            with col2:
                st.metric("Kurtosis", f"{kurtosis:.4f}",
                          delta="Normal" if abs(kurtosis) < 0.5 else
                          "Slight" if abs(kurtosis) < 1 else "Non-normal")

    # Recommendations based on normality test
    st.subheader("Recommendations")

    if normal_count/total_count < 0.5:
        st.warning(
            "Most of your data doesn't follow a normal distribution. Consider:")
        st.write("""
        1. **Data Transformation**: Apply transformations like log, square root, or Box-Cox
        2. **Non-parametric Methods**: Use algorithms that don't assume normality
        3. **Robust Scaling**: Use scaling methods less affected by non-normal distributions
        """)
    else:
        st.success("Most of your data follows a normal distribution.")
        st.write("Your data is suitable for algorithms that assume normality.")

    return results_df


def detect_outliers_with_iqr(df, multiplier=1.5):
    """
    Detect outliers in a DataFrame using the IQR method.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the numerical data to analyze for outliers
    multiplier : float, default=1.5
        The IQR multiplier to identify outliers (1.5 is standard)

    Returns:
    --------
    outlier_df : pandas DataFrame
        DataFrame with boolean flags for outliers in each column
    outlier_summary : pandas DataFrame
        Summary of outlier counts per column
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    # Create a copy of the dataframe with only numeric columns
    df_numeric = df.select_dtypes(include=np.number).copy()

    if df_numeric.empty:
        st.error("Tidak ada kolom numerik yang ditemukan dalam dataset.")
        return None, None

    # Initialize DataFrames for outliers and stats
    outliers = pd.DataFrame(index=df.index)
    stats = {}

    # Calculate IQR and identify outliers for each column
    for col in df_numeric.columns:
        # Calculate quartiles
        q1 = df_numeric[col].quantile(0.25)
        q3 = df_numeric[col].quantile(0.75)
        iqr = q3 - q1

        # Calculate bounds
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        # Identify outliers
        outliers[col] = (df_numeric[col] < lower_bound) | (
            df_numeric[col] > upper_bound)

        # Store statistics
        stats[col] = {
            'Q1': q1,
            'Median': df_numeric[col].median(),
            'Q3': q3,
            'IQR': iqr,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': outliers[col].sum(),
            'Outlier_Percentage': (outliers[col].sum() / len(df)) * 100
        }

    # Create summary dataframe
    summary_data = []
    for col, col_stats in stats.items():
        summary_data.append({
            'Kolom': col,
            'Jumlah Outlier': int(col_stats['Outlier_Count']),
            'Persentase (%)': round(col_stats['Outlier_Percentage'], 2),
            'Batas Bawah': round(col_stats['Lower_Bound'], 2),
            'Batas Atas': round(col_stats['Upper_Bound'], 2),
            'IQR': round(col_stats['IQR'], 2)
        })

    outlier_summary = pd.DataFrame(summary_data).sort_values(
        'Jumlah Outlier', ascending=False)

    return outliers, outlier_summary, stats


def visualize_iqr_outliers(df, outliers, stats):
    """
    Visualize outliers detected using the IQR method.

    Parameters:
    -----------
    df : pandas DataFrame
        Original DataFrame with data
    outliers : pandas DataFrame
        DataFrame with boolean flags for outliers from detect_outliers_with_iqr
    stats : dict
        Dictionary containing outlier statistics from detect_outliers_with_iqr
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    # Get numeric columns
    df_numeric = df.select_dtypes(include=np.number)

    # Display total outliers
    total_outliers = sum(stats[col]['Outlier_Count'] for col in stats)
    total_possible = len(df) * len(stats)
    overall_percentage = (total_outliers / total_possible) * \
        100 if total_possible > 0 else 0

    st.metric(
        label="Total Outlier Terdeteksi",
        value=f"{total_outliers}",
        delta=f"{overall_percentage:.2f}% dari semua data"
    )

    # Plot outlier counts
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract column names and outlier counts
    columns = [col for col in stats.keys()]
    counts = [stats[col]['Outlier_Count'] for col in columns]
    percentages = [stats[col]['Outlier_Percentage'] for col in columns]

    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_columns = [columns[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]

    # Color bars based on percentage
    colors = ['red' if p > 10 else 'orange' if p >
              5 else 'green' for p in sorted_percentages]

    # Create horizontal bar chart
    bars = ax.barh(sorted_columns, sorted_counts, color=colors)

    # Add percentages as labels
    for i, (bar, percentage) in enumerate(zip(bars, sorted_percentages)):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{percentage:.1f}%",
            va='center'
        )

    ax.set_title('Jumlah Outlier per Kolom')
    ax.set_xlabel('Jumlah Outlier')
    ax.set_ylabel('Fitur')
    plt.tight_layout()
    st.pyplot(fig)

    # Allow user to select columns for detailed visualization
    if len(stats) > 0:
        selected_columns = st.multiselect(
            "Pilih kolom untuk visualisasi outlier detail",
            options=list(stats.keys()),
            default=list(stats.keys())[:min(3, len(stats))]
        )

        if selected_columns:
            for col in selected_columns:
                # Create box plot for the selected column
                fig, ax = plt.subplots(figsize=(10, 5))

                # Plot boxplot
                sns.boxplot(x=df_numeric[col], ax=ax)

                # Add bounds as vertical lines
                ax.axvline(x=stats[col]['Lower_Bound'], color='r', linestyle='--',
                           label=f"Batas Bawah ({stats[col]['Lower_Bound']:.2f})")
                ax.axvline(x=stats[col]['Upper_Bound'], color='r', linestyle='--',
                           label=f"Batas Atas ({stats[col]['Upper_Bound']:.2f})")

                ax.set_title(f'Box Plot {col} dengan Batas Outlier')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Show histogram
                fig, ax = plt.subplots(figsize=(10, 5))

                # Get outlier indices for this column
                outlier_indices = outliers[col]

                # Plot regular data points
                sns.histplot(
                    df_numeric.loc[~outlier_indices, col],
                    ax=ax,
                    color='blue',
                    label='Data Normal'
                )

                # Plot outlier data points if any
                if stats[col]['Outlier_Count'] > 0:
                    sns.histplot(
                        df_numeric.loc[outlier_indices, col],
                        ax=ax,
                        color='red',
                        label='Outlier'
                    )

                ax.set_title(f'Distribusi {col} dengan Outlier')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Display outlier values for this column
                if stats[col]['Outlier_Count'] > 0:
                    st.write(f"Nilai outlier untuk {col}:")
                    outlier_values = df_numeric.loc[outlier_indices, col].sort_values(
                    )
                    st.write(outlier_values)


def process_with_iqr_outlier_detection(df):
    """
    Process data with IQR outlier detection and visualization.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to process

    Returns:
    --------
    df_processed : pandas DataFrame
        Processed DataFrame with outlier information
    """
    st.subheader("Deteksi Outlier dengan Metode IQR")

    st.write("""
    Metode IQR (Interquartile Range) adalah cara robust untuk mendeteksi outlier:
    - **Q1**: Kuartil pertama (25% data)
    - **Q3**: Kuartil ketiga (75% data)
    - **IQR**: Jarak antara Q1 dan Q3
    - **Outlier**: Data yang berada di luar range [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    
    Metode ini tidak mengasumsikan distribusi normal, sehingga lebih robust dibandingkan Z-score.
    """)

    # Detect outliers
    outliers, outlier_summary, stats = detect_outliers_with_iqr(df, 1.5)

    if outliers is not None:
        # Display summary table
        st.subheader("Ringkasan Outlier")
        st.dataframe(outlier_summary)

        # Visualize outliers
        visualize_iqr_outliers(df, outliers, stats)

        # Option to handle outliers
        st.subheader("Penanganan Outlier")
        outlier_handling = st.radio(
            "Bagaimana cara menangani outlier?",
            ["Pertahankan semua data", "Hapus baris dengan outlier",
                "Batasi nilai outlier (capping)"]
        )

        if outlier_handling == "Hapus baris dengan outlier":
            # Calculate which rows to keep (rows with no outliers in any column)
            rows_with_outliers = outliers.any(axis=1)
            cleaned_df = df[~rows_with_outliers]

            removed_count = len(df) - len(cleaned_df)
            st.warning(
                f"Menghapus {removed_count} baris ({removed_count/len(df)*100:.2f}% dari data) yang mengandung outlier")

            return cleaned_df

        elif outlier_handling == "Batasi nilai outlier (capping)":
            # Cap outliers at the boundary values
            capped_df = df.copy()

            for col in stats.keys():
                lower_bound = stats[col]['Lower_Bound']
                upper_bound = stats[col]['Upper_Bound']

                # Cap values below lower bound
                capped_df.loc[df[col] < lower_bound, col] = lower_bound

                # Cap values above upper bound
                capped_df.loc[df[col] > upper_bound, col] = upper_bound

            st.success("Nilai outlier dibatasi pada batas IQR")
            return capped_df

    # Default: return the original dataframe
    return df
