import streamlit as st
import os
import numpy as np
import utils.data_preprocessing as dp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if data exists in session state
if 'df' not in st.session_state:
    try:
        st.session_state.df = dp.data_loader('data/ta_dataset.csv')
        st.success("Data loaded successfully")
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
    2. Likert Reverse Coding (Melakukan reverse coding pada kolom yang diperlukan)
    3. Memilih kolom yang diperlukan
    4. Melihat multicollinearity
    5. Melakukan skalasi data
    6. Principal Component Analysis (PCA) per domain
    """
)

st.write("## Data sebelum preprocessing")
original_df = dp.data_loader("data/ta_dataset.csv")
original_df = original_df[(original_df['Umur'] > 17) & (original_df['Umur'] <= 23)]
st.dataframe(original_df)

# Preprocess data
df = dp.Preprocess_Data(df)
st.write("## Data setelah preprocessing")
st.dataframe(df)

st.session_state.df = df

# Multicollinearity Analysis
st.subheader("Analisis Multicollinearity")

threshold = st.slider(
    "Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
)

multicollinearity = dp.pairwise_multicollinearity(df, threshold)
if len(multicollinearity) > 0:
    st.error(f"Membutuhkan penggunaan PCA")
    st.session_state.multicollinearity = multicollinearity
    use_pca_required = True
else:
    st.info(f"Tidak ada fitur yang memiliki korelasi lebih dari {threshold}.")
    st.session_state.multicollinearity = []
    use_pca_required = False

st.write("## Multicollinearity")
dp.plot_correlation_matrix(df, threshold)


# IQR Outlier Analysis
st.subheader("Analisis Outlier (IQR Method)")

st.markdown("""
Analisis outlier menggunakan Interquartile Range (IQR) untuk mengidentifikasi nilai-nilai ekstrem 
yang mungkin mempengaruhi hasil clustering per domain.
""")


iqr_multiplier = 1.5
# Function to detect outliers using IQR per domain
def detect_outliers_iqr_per_domain(data, multiplier=1.5):
    """
    Deteksi outlier menggunakan metode IQR per domain
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data yang akan dianalisis
    multiplier : float
        Pengali IQR untuk menentukan batas outlier
        
    Returns:
    --------
    domain_outlier_summary : pandas.DataFrame
        Summary outlier per domain
    detailed_outlier_info : dict
        Informasi detail outlier per fitur
    """
    # Define domains
    domain_mapping = {
        'fa': 'Financial Attitude',
        'fb': 'Financial Behavior',
        'fk': 'Financial Knowledge',
        'm': 'Materialism'
    }
    
    detailed_outlier_info = {}
    domain_summary = []
    
    for domain_prefix, domain_name in domain_mapping.items():
        # Get domain features
        domain_features = [col for col in data.columns if col.startswith(domain_prefix)]
        
        if len(domain_features) == 0:
            continue
            
        domain_data = data[domain_features]
        total_domain_outliers = 0
        total_domain_values = 0
        
        # Analyze each feature in domain
        for column in domain_features:
            Q1 = domain_data[column].quantile(0.25)
            Q3 = domain_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Deteksi outlier
            column_outliers = (domain_data[column] < lower_bound) | (domain_data[column] > upper_bound)
            outlier_count = column_outliers.sum()
            
            # Store detailed info
            detailed_outlier_info[column] = {
                'domain': domain_name,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outlier_count,
                'outlier_percentage': (outlier_count / len(domain_data)) * 100,
                'outlier_indices': domain_data.index[column_outliers].tolist()
            }
            
            total_domain_outliers += outlier_count
            total_domain_values += len(domain_data)
        
        # Calculate domain percentage
        domain_percentage = (total_domain_outliers / total_domain_values) * 100 if total_domain_values > 0 else 0
        
        # Add to domain summary
        domain_summary.append({
            'No': len(domain_summary) + 1,
            'Domain': domain_name,
            'Total Features': len(domain_features),
            'Total Outliers': total_domain_outliers,
            'Percentage': f"{domain_percentage:.2f}%"
        })
    
    domain_outlier_summary = pd.DataFrame(domain_summary)
    return domain_outlier_summary, detailed_outlier_info

# Perform outlier analysis per domain
with st.spinner("Menganalisis outlier per domain..."):
    domain_summary_df, detailed_info = detect_outliers_iqr_per_domain(df, iqr_multiplier)

# Display domain summary table
st.write("### Ringkasan Outlier per Domain")

# Style the dataframe to match your desired format
def highlight_high_domain_outliers(row):
    percentage = float(row['Percentage'].rstrip('%'))
    if percentage > 5:
        return ['background-color: #B22222; color: white'] * len(row)  # Dark red with white text
    elif percentage > 2:
        return ['background-color: #8B4513; color: white'] * len(row)  # Dark brown with white text
    else:
        return ['background-color: #2E8B57; color: white'] * len(row)  # Dark green with white text

st.dataframe(
    domain_summary_df.style.apply(highlight_high_domain_outliers, axis=1),
    use_container_width=True,
    column_config={
        "No": st.column_config.NumberColumn("No", format="%d"),
        "Domain": st.column_config.TextColumn("Domain"),
        "Total Features": st.column_config.NumberColumn("Total Features", format="%d"),
        "Total Outliers": st.column_config.NumberColumn("Total Outliers", format="%d"),
        "Percentage": st.column_config.TextColumn("Percentage")
    },
    hide_index=True
)

# Overall statistics
total_features = domain_summary_df['Total Features'].sum()
total_outliers = domain_summary_df['Total Outliers'].sum()
overall_percentage = (total_outliers / (total_features * len(df))) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Features", total_features)
with col2:
    st.metric("Total Outliers", total_outliers)
with col3:
    st.metric("Overall Percentage", f"{overall_percentage:.2f}%")

# Domain-wise visualization
st.write("### Visualisasi Outlier per Domain")

# Bar chart of outliers per domain
fig_domain, ax_domain = plt.subplots(figsize=(12, 6))
bars = ax_domain.bar(domain_summary_df['Domain'], domain_summary_df['Total Outliers'])

# Color bars based on percentage
for i, (bar, pct_str) in enumerate(zip(bars, domain_summary_df['Percentage'])):
    pct = float(pct_str.rstrip('%'))
    if pct > 5:
        bar.set_color("#e71717")  # Red
    elif pct > 2:
        bar.set_color("#f6c812")  # Yellow
    else:
        bar.set_color("#0a7620")  # Green

ax_domain.set_xlabel('Domain')
ax_domain.set_ylabel('Number of Outliers')
ax_domain.set_title('Distribusi Outlier per Domain')
ax_domain.grid(True, alpha=0.3)

# Add percentage labels on bars
for i, (bar, total, pct) in enumerate(zip(bars, domain_summary_df['Total Outliers'], domain_summary_df['Percentage'])):
    height = bar.get_height()
    ax_domain.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{total}\n({pct})',
                   ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_domain)

# Detailed analysis per domain (expandable)
with st.expander("Analisis Detail per Domain", expanded=False):
    domain_tabs = st.tabs([row['Domain'] for _, row in domain_summary_df.iterrows()])
    
    domain_mapping = {
        'Financial Attitude': 'fa',
        'Financial Behavior': 'fb', 
        'Financial Knowledge': 'fk',
        'Materialism': 'm'
    }
    
    for i, (_, domain_row) in enumerate(domain_summary_df.iterrows()):
        with domain_tabs[i]:
            domain_name = domain_row['Domain']
            domain_prefix = domain_mapping[domain_name]
            
            # Get features for this domain
            domain_features = [col for col in df.columns if col.startswith(domain_prefix)]
            
            if len(domain_features) > 0:
                # Feature-level outlier details
                feature_details = []
                for feature in domain_features:
                    if feature in detailed_info:
                        info = detailed_info[feature]
                        feature_details.append({
                            'Feature': feature,
                            'Outliers': info['outlier_count'],
                            'Percentage': f"{info['outlier_percentage']:.2f}%",
                            'Q1': f"{info['Q1']:.3f}",
                            'Q3': f"{info['Q3']:.3f}",
                            'IQR': f"{info['IQR']:.3f}",
                            'Lower Bound': f"{info['lower_bound']:.3f}",
                            'Upper Bound': f"{info['upper_bound']:.3f}"
                        })
                
                if feature_details:
                    feature_df = pd.DataFrame(feature_details)
                    feature_df = feature_df.sort_values('Outliers', ascending=False)
                    
                    st.write(f"**Detail Outlier {domain_name}:**")
                    
                    # Ganti bagian highlight_feature_outliers function dengan:
                    def highlight_feature_outliers(row):
                        pct = float(row['Percentage'].rstrip('%'))
                        if pct > 10:
                            return ['background-color: #8B4513; color: white'] * len(row)  # Dark brown with white text
                        elif pct > 5:
                            return ['background-color: #B22222; color: white'] * len(row)  # Dark red with white text
                        else:
                            return [''] * len(row)
                    
                    st.dataframe(
                        feature_df.style.apply(highlight_feature_outliers, axis=1),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Box plots for top outlier features in this domain
                    top_features = feature_df.head(3)['Feature'].tolist()
                    if len(top_features) > 0:
                        st.write(f"**Box Plot - Top {len(top_features)} Features dengan Outlier Terbanyak:**")
                        
                        n_features = len(top_features)
                        fig_box, axes_box = plt.subplots(1, n_features, figsize=(5*n_features, 4))
                        if n_features == 1:
                            axes_box = [axes_box]
                        
                        for j, feature in enumerate(top_features):
                            if j < len(axes_box):
                                df[feature].plot(kind='box', ax=axes_box[j])
                                info = detailed_info[feature]
                                axes_box[j].set_title(f'{feature}\n({info["outlier_count"]} outliers)')
                                axes_box[j].grid(True, alpha=0.3)
                                
                                # Add bounds
                                axes_box[j].axhline(y=info['lower_bound'], 
                                                   color='red', linestyle='--', alpha=0.7)
                                axes_box[j].axhline(y=info['upper_bound'], 
                                                   color='red', linestyle='--', alpha=0.7)
                        
                        plt.tight_layout()
                        st.pyplot(fig_box)
                else:
                    st.info(f"Tidak ada outlier terdeteksi di domain {domain_name}")
            else:
                st.warning(f"Tidak ada fitur ditemukan untuk domain {domain_name}")

# Store outlier information in session state (hanya untuk deteksi)
st.session_state.outlier_info = {
    'domain_summary': domain_summary_df.to_dict(),
    'detailed_analysis': detailed_info,
    'iqr_multiplier': iqr_multiplier,
    'action_taken': 'detection_only',
    'total_outliers': total_outliers,
    'overall_percentage': overall_percentage
}




# Data Scaling Configuration
st.subheader("Konfigurasi Skalasi Data")

scaler_type = "RobustScaler"
# Store scaler choice
st.session_state.scaler_type = scaler_type

# Apply scaling
if scaler_type == "StandardScaler":
    scaler = StandardScaler()
elif scaler_type == "MinMaxScaler":
    scaler = MinMaxScaler()
elif scaler_type == "RobustScaler":
    scaler = RobustScaler()

# Scale the data
scaled_data = scaler.fit_transform(df)
st.session_state.scaler = scaler
st.session_state.scaled_data_original = scaled_data

st.write("### Hasil Skalasi Data")
with st.expander("Hasil Skalasi Data", expanded=False):
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    st.dataframe(scaled_df.head(), use_container_width=True)

# PCA Configuration
st.subheader("Konfigurasi Principal Component Analysis (PCA)")

use_pca = True

if use_pca:
    st.info("PCA akan diterapkan pada setiap domain secara terpisah")
    
    n_components = 2
    
    st.session_state.use_pca = True
    st.session_state.n_components = n_components
    
    # Perform PCA
    with st.spinner("Melakukan PCA per domain..."):
        pca_results = dp.perform_pca(df, scaler_type, n_components)
        
        # Unpack results
        (scaled_data_combined, pca_result_fa, pca_result_fb, pca_result_fk, pca_result_m,
         pca_result_fa_explained, pca_result_fb_explained, pca_result_fk_explained,
         pca_result_m_explained, cumulative_variance, cumulative_variance_fa,
         cumulative_variance_fb, cumulative_variance_fk, cumulative_variance_m,
         explained_variance) = pca_results
    
    # Store PCA results in session state
    st.session_state.pca_result_fa = pca_result_fa
    st.session_state.pca_result_fb = pca_result_fb
    st.session_state.pca_result_fk = pca_result_fk
    st.session_state.pca_result_m = pca_result_m
    st.session_state.pca_explained_variance = {
        'fa': pca_result_fa_explained,
        'fb': pca_result_fb_explained,
        'fk': pca_result_fk_explained,
        'm': pca_result_m_explained
    }
    st.session_state.cumulative_variance = {
        'combined': cumulative_variance,
        'fa': cumulative_variance_fa,
        'fb': cumulative_variance_fb,
        'fk': cumulative_variance_fk,
        'm': cumulative_variance_m
    }
    
    # PCA Visualization and Analysis
    st.subheader("Hasil PCA per Domain")
    
    domain_names = ["Financial Attitude", "Financial Behavior", "Financial Knowledge", "Materialism"]
    domain_keys = ["fa", "fb", "fk", "m"]
    
    # Summary metrics
    cols = st.columns(4)
    for i, (name, key) in enumerate(zip(domain_names, domain_keys)):
        with cols[i]:
            cum_var = st.session_state.cumulative_variance[key][-1]
            st.metric(f"{name}", f"{cum_var:.2%} explained")
    
    # Domain tabs for detailed PCA results
    domain_tabs = st.tabs(domain_names)
    
    pca_data_map = {
        'fa': (pca_result_fa, pca_result_fa_explained, cumulative_variance_fa),
        'fb': (pca_result_fb, pca_result_fb_explained, cumulative_variance_fb),
        'fk': (pca_result_fk, pca_result_fk_explained, cumulative_variance_fk),
        'm': (pca_result_m, pca_result_m_explained, cumulative_variance_m)
    }
    
    for i, (domain_name, domain_key) in enumerate(zip(domain_names, domain_keys)):
        with domain_tabs[i]:
            pca_result, explained_var, cum_var = pca_data_map[domain_key]
            
            # Explained variance chart
            fig = px.bar(
                x=[f"PC{i+1}" for i in range(len(cum_var))],
                y=explained_var,
                labels={"x": "Principal Component", "y": "Explained Variance"},
                title=f"Explained Variance by Component - {domain_name}"
            )
            fig.add_scatter(
                x=[f"PC{i+1}" for i in range(len(cum_var))],
                y=cum_var,
                mode="lines+markers",
                name="Cumulative Variance"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA data preview
            st.write(f"### {domain_name} PCA Preview")
            pca_df = pd.DataFrame(
                pca_result,
                columns=[f"{domain_key.upper()}_PC{i+1}" for i in range(min(n_components, pca_result.shape[1]))]
            )
            st.dataframe(pca_df.head(), use_container_width=True)
            
            # Feature loadings analysis
            with st.expander(f"Analisis Loading Components - {domain_name}"):
                domain_features = [col for col in df.columns if col.startswith(domain_key)]
                
                if len(domain_features) > 0:
                    # Get PCA object for this domain
                    if domain_key == 'fa':
                        domain_data = df.filter(like='fa')
                    elif domain_key == 'fb':
                        domain_data = df.filter(like='fb')
                    elif domain_key == 'fk':
                        domain_data = df.filter(like='fk')
                    elif domain_key == 'm':
                        domain_data = df.filter(like='m')
                    
                    # Scale and fit PCA for loading analysis
                    scaled_domain_data = scaler.fit_transform(domain_data)
                    pca_temp = PCA(n_components=min(n_components, domain_data.shape[1]))
                    pca_temp.fit(scaled_domain_data)
                    
                    # Create loading matrix
                    loadings = pd.DataFrame(
                        pca_temp.components_.T,
                        columns=[f"PC{i+1}" for i in range(pca_temp.n_components_)],
                        index=domain_features
                    )
                    
                    st.write("**Feature Loadings:**")
                    st.dataframe(loadings.style.format("{:.3f}"))
                    
                    # Heatmap of loadings
                    fig_loading = px.imshow(
                        loadings.T,
                        labels=dict(x="Features", y="Principal Components", color="Loading"),
                        title=f"Feature Loadings Heatmap - {domain_name}",
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_loading, use_container_width=True)
    
    # Combined PCA Results
    with st.expander("Hasil PCA Gabungan", expanded=False):
        all_pca_cols = []
        all_pca_data = []
        
        if pca_result_fa.shape[1] > 0:
            fa_cols = [f"FA_PC{i+1}" for i in range(min(n_components, pca_result_fa.shape[1]))]
            all_pca_cols.extend(fa_cols)
            all_pca_data.append(pd.DataFrame(pca_result_fa, columns=fa_cols))
        
        if pca_result_fb.shape[1] > 0:
            fb_cols = [f"FB_PC{i+1}" for i in range(min(n_components, pca_result_fb.shape[1]))]
            all_pca_cols.extend(fb_cols)
            all_pca_data.append(pd.DataFrame(pca_result_fb, columns=fb_cols))
        
        if pca_result_fk.shape[1] > 0:
            fk_cols = [f"FK_PC{i+1}" for i in range(min(n_components, pca_result_fk.shape[1]))]
            all_pca_cols.extend(fk_cols)
            all_pca_data.append(pd.DataFrame(pca_result_fk, columns=fk_cols))
        
        if pca_result_m.shape[1] > 0:
            m_cols = [f"M_PC{i+1}" for i in range(min(n_components, pca_result_m.shape[1]))]
            all_pca_cols.extend(m_cols)
            all_pca_data.append(pd.DataFrame(pca_result_m, columns=m_cols))
        
        if all_pca_data:
            combined_pca = pd.concat(all_pca_data, axis=1)
            st.write("### Combined PCA Results")
            st.dataframe(combined_pca.head(), use_container_width=True)
            
            # Store combined PCA data
            st.session_state.combined_pca_data = combined_pca

else:
    st.info("PCA tidak digunakan")
    st.session_state.use_pca = False
    # Use scaled data without PCA
    st.session_state.pca_result_fa = None
    st.session_state.pca_result_fb = None
    st.session_state.pca_result_fk = None
    st.session_state.pca_result_m = None

# Summary of preprocessing
st.subheader("Ringkasan Preprocessing")

summary_cols = st.columns(3)
with summary_cols[0]:
    st.metric("Original Features", len(original_df.columns))
with summary_cols[1]:
    st.metric("Processed Features", len(df.columns))
with summary_cols[2]:
    if use_pca:
        total_pca_components = (
            (pca_result_fa.shape[1] if pca_result_fa is not None else 0) +
            (pca_result_fb.shape[1] if pca_result_fb is not None else 0) +
            (pca_result_fk.shape[1] if pca_result_fk is not None else 0) +
            (pca_result_m.shape[1] if pca_result_m is not None else 0)
        )
        st.metric("PCA Components", total_pca_components)
    else:
        st.metric("Scaled Features", len(df.columns))

# Status indicators
st.subheader("Status Preprocessing")
status_cols = st.columns(4)

with status_cols[0]:
    if len(multicollinearity) == 0:
        st.success("No Multicollinearity")
    else:
        st.warning(f"{len(multicollinearity)} Multicollinear pairs")

with status_cols[1]:
    st.success(f"Data Scaled ({scaler_type})")

with status_cols[2]:
    if use_pca:
        st.success("PCA Applied")
    else:
        st.info("ℹPCA Not Used")

with status_cols[3]:
    st.success("Ready for Training")

