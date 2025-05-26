import streamlit as st
import pandas as pd
import utils.data_preprocessing as dp
import utils.data_visualization as dv
import dotenv
import os
# dotenv.load_dotenv()
from pathlib import Path

st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'df' not in st.session_state:
    try:
        # Use direct path instead of trying to get it from environment variables
        data_path = "data/ta_dataset.csv"

        # Check if file exists
        if not Path(data_path).exists():
            st.error(f"File not found: {data_path}")
            st.info("Please make sure the data file exists in the correct location.")
            st.stop()

        # Load data using the path
        df = dp.data_loader(data_path)
        st.success(f"Data loaded successfully from {data_path}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using the direct path as fallback")

        try:
            df = dp.data_loader("data/ta_dataset.csv")
            st.success("Data loaded successfully from fallback path")
        except Exception as e2:
            st.error(f"Failed to load data from fallback path: {str(e2)}")
            st.stop()
else:
    df = st.session_state.df


st.title("Data Preprocessing")

st.markdown(
    """
    1. Drop semua kolom yang tidak diperlukan
    2. Likerd Reverse Coding (Melakukan reverse coding pada kolom yang diperlukan)
    3. Memilih kolom yang diperlukan
    4. Melihat multicollinearity
    5. Melakukan skalasi data
    """
)

st.write("## Data sebelum preprocessing")
original_df = dp.data_loader("data/ta_dataset.csv")
st.dataframe(original_df)
df = dp.Preprocess_Data(df)
st.write("## Data setelah preprocessing")
st.dataframe(df)


st.session_state.df = df

threshold = st.slider(
    "Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
)

multicollinearity = dp.pairwise_multicollinearity(df, threshold)
if len(multicollinearity) > 0:
    st.error(
        f"Membutuhkan penggunaan PCA")
    st.session_state.multicollinearity = multicollinearity
else:
    st.info(f"Tidak ada fitur yang memiliki korelasi lebih dari {threshold}.")
    st.session_state.multicollinearity = []

st.write("## Multicollinearity")
dp.plot_correlation_matrix(df, threshold)
dp.graph_pairwise_correlation(df, threshold)


# Add outlier detection section after multicollinearity
st.write("## Outlier Detection (IQR Method)")

# Define domains
domains = {
    "fa": "Financial Attitude",
    "fb": "Financial Behavior",
    "fk": "Financial Knowledge",
    "m": "Materialism"
}

# Create tabbed interface for outlier analysis
outlier_tabs = st.tabs(["Overview"] + list(domains.values()))

# Function to detect outliers using IQR method


def detect_outliers_iqr(data, multiplier=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    percentage = (outliers / len(data)) * 100

    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_count': outliers,
        'percentage': percentage,
        'is_outlier': (data < lower_bound) | (data > upper_bound)
    }


# Store outlier information
domain_outliers = {}

# Overview tab
with outlier_tabs[0]:
    # Summary of all domains
    st.write("### Ringkasan Outlier per Domain")

    # Create dictionary to store outlier summary
    outlier_summary = []

    # Detect outliers for each domain
    for domain_code, domain_name in domains.items():
        # Get columns for this domain
        domain_cols = [
            col for col in df.columns if col.startswith(domain_code)]

        if not domain_cols:
            continue

        # Compute outliers for each column in the domain
        domain_df = df[domain_cols]
        outlier_results = {}
        total_outliers = 0
        total_possible = len(domain_df) * len(domain_cols)

        for col in domain_cols:
            result = detect_outliers_iqr(df[col], 1.5)
            outlier_results[col] = result
            total_outliers += result['outliers_count']

        # Store results
        domain_outliers[domain_code] = {
            'columns': domain_cols,
            'results': outlier_results,
            'total_outliers': total_outliers,
            'percentage': (total_outliers / total_possible) * 100 if total_possible > 0 else 0
        }

        # Add to summary
        outlier_summary.append({
            'Domain': domain_name,
            'Total Features': len(domain_cols),
            'Total Outliers': total_outliers,
            'Percentage': f"{domain_outliers[domain_code]['percentage']:.2f}%"
        })

    # Display summary table
    st.table(pd.DataFrame(outlier_summary))

    # Overall visualization
    st.write("### Total Outliers per Domain")

    # Create bar chart of outliers by domain
    import plotly.express as px

    fig = px.bar(
        pd.DataFrame(outlier_summary),
        x='Domain',
        y='Total Outliers',
        color='Domain',
        title="Total Outlier Points by Domain"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Store in session state
    st.session_state.domain_outliers = domain_outliers

# Domain-specific tabs
for i, (domain_code, domain_name) in enumerate(domains.items()):
    with outlier_tabs[i+1]:
        if domain_code in domain_outliers:
            st.write(f"### {domain_name} Outlier Analysis")

            # Get the outlier data for this domain
            domain_data = domain_outliers[domain_code]

            # Summary
            st.write(
                f"**Total outliers detected:** {domain_data['total_outliers']}")
            st.write(f"**Percentage:** {domain_data['percentage']:.2f}%")

            # Feature-specific outlier counts
            feature_outliers = {}
            for col, result in domain_data['results'].items():
                feature_outliers[col] = result['outliers_count']

            # Create bar chart of outliers by feature
            feature_df = pd.DataFrame({
                'Feature': list(feature_outliers.keys()),
                'Outlier Count': list(feature_outliers.values())
            }).sort_values(by='Outlier Count', ascending=False)

            fig = px.bar(
                feature_df,
                x='Feature',
                y='Outlier Count',
                title=f"Outlier Count by Feature in {domain_name}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show box plots for features with most outliers
            st.write("### Box Plots for Features")
            top_outlier_features = feature_df.head(min(5, len(feature_df)))[
                'Feature'].tolist()

            # Plot boxplots for top outlier features
            fig = px.box(df, y=top_outlier_features,
                         title=f"Box Plots for Top Outlier Features in {domain_name}")
            st.plotly_chart(fig, use_container_width=True)

            # Display outlier data
            with st.expander("Show Outlier Details"):
                outlier_detail_df = pd.DataFrame({
                    'Feature': [],
                    'Lower Bound': [],
                    'Upper Bound': [],
                    'Outlier Count': [],
                    'Percentage': []
                })

                for col, result in domain_data['results'].items():
                    outlier_detail_df = pd.concat([outlier_detail_df, pd.DataFrame({
                        'Feature': [col],
                        'Lower Bound': [result['lower_bound']],
                        'Upper Bound': [result['upper_bound']],
                        'Outlier Count': [result['outliers_count']],
                        'Percentage': [f"{result['percentage']:.2f}%"]
                    })], ignore_index=True)

                st.dataframe(outlier_detail_df, use_container_width=True)
        else:
            st.info(f"No columns found for {domain_name}")
