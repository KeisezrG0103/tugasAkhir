import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="AI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def make_json_serializable(obj):
    """Convert complex types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)  # Convert any other types to strings


# Question mapping for reference
question_mapping = {
    'fa1': 'Memiliki kebiasaan menabung secara teratur merupakan hal yang penting bagi saya.',
    'fa2': 'Menyusun rencana keuangan secara tertulis membantu dalam menentukan prioritas pengeluaran.',
    'fa3': 'Anggaran yang terdokumentasi dengan baik berperan penting dalam keberhasilan pengelolaan keuangan.',
    'fa4': 'Setiap keluarga harus memiliki strategi keuangan yang matang untuk mengantisipasi risiko kehilangan sumber pendapatan utama.',
    'fa5': 'Perencanaan pengeluaran merupakan aspek krusial dalam pengelolaan keuangan pribadi',
    'fa6': 'Menetapkan tujuan keuangan yang jelas sangat penting untuk mencapai kesuksesan finansial',
    'fa7': 'Memvisualisasikan aset dalam jangka waktu 5–10 tahun ke depan membantu dalam pencapaian tujuan keuangan.',
    'fa8': 'Fokus utama dalam perencanaan keuangan adalah kebutuhan saat ini',
    'fa9': 'Perencanaan keuangan untuk masa pensiun tidak diperlukan.',
    'fa10': 'Perencanaan keuangan menghambat pengambilan keputusan investasi.',
    'fa11': 'Rencana tabungan tidak memiliki peran yang signifikan dalam stabilitas keuangan.',
    'fa12': 'Perencanaan keuangan membatasi pemenuhan kebutuhan individu.',
    'fa13': 'Mencatat dan mengelola keuangan secara rinci terlalu memakan waktu.',
    'fa14': 'Menabung bukanlah suatu keharusan.',
    'fa15': 'Selama kebutuhan bulanan terpenuhi, saya tidak perlu mempertimbangkan waktu yang dibutuhkan untuk melunasi utang.',
    'fb1': 'Saya secara rutin mencatat dan mengontrol pengeluaran pribadi',
    'fb2': 'Saya selalu membandingkan harga sebelum melakukan pembelian.',
    'fb3': 'Saya menyisihkan sebagian pendapatan untuk kebutuhan di masa mendatang.',
    'fb4': 'Saya memiliki anggaran keuangan yang terencana untuk pengeluaran saya.',
    'fb5': 'Saya memahami sepenuhnya pembelian yang dilakukan dengan kredit (misalnya, kartu kredit, cicilan, atau fitur "pay later").',
    'fb6': 'Saya selalu membayar seluruh tagihan tepat waktu.',
    'fb7': 'Saya berkomitmen untuk menabung setiap bulan.',
    'fb8': 'Saya mempertimbangkan kondisi keuangan sebelum melakukan pengeluaran dalam jumlah besar.',
    'fb9': 'Saya selalu membayar utang tepat waktu untuk menghindari denda atau bunga tambahan.',
    'fb10': 'Saya menabung secara teratur untuk mencapai target keuangan jangka panjang.',
    'fb11': 'Saya meningkatkan jumlah tabungan apabila pendapatan saya meningkat dalam bulan tertentu.',
    'fb12': ' Saya memiliki tabungan yang setara dengan setidaknya tiga kali pendapatan bulanan saya, yang dapat digunakan kapan saja.',
    'fb13': 'Saya telah secara konsisten menabung dalam 12 bulan terakhir.',
    'fk1': 'Jika Anda memiliki Rp100.000 dalam rekening tabungan dengan bunga 10% per tahun, berapa jumlah yang akan Anda miliki setelah lima tahun?',
    'fk2': 'Jika tingkat bunga tabungan Anda adalah 6% per tahun dan tingkat inflasi 10% per tahun, bagaimana daya beli uang Anda setelah satu tahun?',
    'fk3': 'Anda meminjam Rp1.000.000 dari bank dan diharuskan membayar bunga sebesar Rp60.000 setelah satu tahun. Berapa tingkat bunga pinjaman Anda?',
    'fk4': 'Sebuah telepon genggam dihargai Rp10.000.000. Toko A memberikan diskon Rp1.500.000, sedangkan Toko B memberikan diskon 10%. Manakah pilihan yang lebih menguntungkan?',
    'fk5': 'Lima teman berbagi tagihan makan malam sebesar Rp100.000 secara merata. Berapa jumlah yang harus dibayar oleh masing-masing individu?',
    'fk6': 'Dalam periode 10 tahun, investasi mana yang biasanya memberikan tingkat pengembalian tertinggi?',
    'fk7': 'Investasi mana yang memiliki tingkat fluktuasi nilai paling tinggi?',
    'fk8': 'Apa dampak dari diversifikasi investasi terhadap risiko kehilangan dana?',
    'fk9': 'Sebuah pinjaman 10 tahun memiliki cicilan bulanan lebih besar dibandingkan pinjaman 20 tahun, tetapi total bunga yang dibayarkan lebih kecil. Pernyataan ini:',
    'fk10': 'Investasi dengan tingkat pengembalian tinggi memiliki tingkat risiko yang rendah. Pernyataan ini:',
    'm1': 'Saya lebih memilih gaya hidup sederhana tanpa banyak memiliki barang.',
    'm2': 'Barang yang saya miliki mencerminkan tingkat keberhasilan saya dalam hidup.',
    'm3': 'Saya senang memiliki barang yang dapat mengesankan orang lain.',
    'm4': 'Saya mengagumi individu yang memiliki rumah, pakaian, dan kendaraan mewah.',
    'm5': 'Saya merasakan kepuasan saat membeli barang baru.',
    'm6': 'Saya lebih memilih menjalani gaya hidup yang mewah.',
    'm7': 'Saya merasa hidup saya akan lebih baik jika saya memiliki barang-barang yang belum saya miliki saat ini.',
    'm8': 'Saya akan lebih bahagia jika saya dapat membeli lebih banyak barang.',
    'm9': 'Saya merasa tidak nyaman jika saya tidak mampu membeli barang yang saya inginkan.'
}

# Application header
st.title("AI-Powered Financial Clustering Analysis")
st.markdown("Analisis mendalam tentang pola keuangan di kalangan Generasi Z dan Milenial Indonesia menggunakan Kecerdasan Buatan.")

# --------------------------------------
# 1. API KEY HANDLING
# --------------------------------------
st.sidebar.header("API Configuration")

api_key = None
# Try to get API key from secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ API key loaded from secrets")
except (KeyError, FileNotFoundError):
    # If not in secrets, check session state or request from user
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # Get API key using secure text input
    api_key = st.sidebar.text_input(
        "Enter your Gemini API key:",
        value=st.session_state.api_key,
        type="password",
        help="API key not found in secrets. Please enter manually."
    )

    # Store API key in session state to persist between reruns
    if api_key:
        st.session_state.api_key = api_key
        st.sidebar.success("✅ API key set")

# API URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"


# --------------------------------------
# 2. DATA VALIDATION
# --------------------------------------
st.sidebar.header("Data Status")

required_data = {
    'clustered_data': 'Cluster results', 
    'domain_cluster_results': 'Domain clustering results', 
    'anova_results': 'ANOVA analysis',
    'algorithm_metrics': 'Algorithm metrics'  
}

# Create status indicators
missing_data = []
available_data = []

# Check which data is available
for key, description in required_data.items():
    if key in st.session_state and st.session_state[key] is not None:
        if isinstance(st.session_state[key], pd.DataFrame) and st.session_state[key].empty:
            missing_data.append(description)
        else:
            available_data.append(f"{description}")
    else:
        missing_data.append(description)
        if key not in st.session_state:
            st.session_state[key] = None

# Show what's available and what's missing
if available_data:
    st.sidebar.success(f"✅ Available data: {', '.join(available_data)}")
else:
    st.sidebar.warning("⚠️ No analysis data found")

if missing_data:
    st.sidebar.warning(f"⚠️ Missing data: {', '.join(missing_data)}")

# --------------------------------------
# 3. DATA PREPARATION
# --------------------------------------

# Main content - first show warnings if there are major issues
if not api_key:
    st.error("❌ API key is required for AI analysis.")
    st.info("Silakan masukkan Gemini API key di sidebar untuk melanjutkan.")
    st.stop()

# Stop if cluster data is missing (required for analysis)
if st.session_state.get('clustered_data') is None and st.session_state.get('cluster_results_df') is None:
    st.error(
        "❌ Data cluster diperlukan tetapi tidak ditemukan. Lakukan clustering terlebih dahulu.")
    st.info("Kembali ke halaman Training dan lakukan analisis cluster untuk menghasilkan data yang diperlukan.")
    st.stop()


# Get cluster data - use clustered_data
if st.session_state.get('clustered_data') is not None:
    cluster_df = st.session_state.clustered_data
else:
    st.error("Data cluster tidak ditemukan.")
    st.stop()


# Get algorithm info
algorithm_info = {}
if st.session_state.get('algorithm_metrics') is not None:
    # Get the complete algorithm metrics
    algorithm_info = st.session_state.algorithm_metrics

    # Extract algorithm name - check if it's directly in the metrics or needs to be accessed differently
    if "algorithm" not in algorithm_info and isinstance(algorithm_info, dict):
        # Try to extract from different possible structures
        if st.session_state.get('current_algorithm'):
            algorithm_info["algorithm"] = st.session_state.get(
                'current_algorithm')
        elif "name" in algorithm_info:
            algorithm_info["algorithm"] = algorithm_info["name"]

    # Do the same for scaler and PCA if needed
    if "scaler" not in algorithm_info:
        algorithm_info["scaler"] = st.session_state.get('scaler', "Unknown")

    if "use_pca" not in algorithm_info:
        algorithm_info["use_pca"] = st.session_state.get('use_pca', False)
else:
    # Fallback to individual metrics if algorithm_metrics is not available
    algorithm_info = {
        "algorithm": st.session_state.get('current_algorithm', "Unknown"),
        "scaler": st.session_state.get('scaler', "Unknown"),
        "use_pca": st.session_state.get('use_pca', False)
    }


if 'cluster_statistics' in st.session_state and 'clustered_data' in st.session_state:
    st.success("Menggunakan statistik yang sudah dihitung di halaman Training")

    # Mendapatkan statistik cluster dari session state
    training_stats = st.session_state.cluster_statistics

# --------------------------------------
# 4. BUILD CLUSTER SUMMARY
# --------------------------------------
st.subheader("Data Kluster yang Dianalisis")

# Display cluster summary
cluster_count = len(cluster_df['cluster'].unique(
)) if 'cluster' in cluster_df.columns else "Unknown"
feature_count = len([col for col in cluster_df.columns if col != 'cluster'])
st.write(f"**Jumlah Kluster:** {cluster_count}")
st.write(f"**Jumlah Fitur:** {feature_count}")
st.write(f"**Jumlah Sampel:** {len(cluster_df)}")

# Create tabs for exploring data
tabs = st.tabs(["Data Overview", "Cluster Summary",
               "Significant Features", "JSON Data"])

with tabs[0]:
    st.dataframe(cluster_df, use_container_width=True)

    # Show algorithm info
    st.subheader("Informasi Algoritma")
    algo_cols = st.columns(3)
    with algo_cols[0]:
        st.metric("Algoritma", algorithm_info.get("algorithm", "Unknown"))
    with algo_cols[1]:
        scaler_value = algorithm_info.get("scaler", "Unknown")
        if str(type(scaler_value)).find('sklearn') >= 0:
            scaler_value = "StandardScaler" if "StandardScaler" in str(
                scaler_value) else str(type(scaler_value).__name__)
        st.metric("Scaler", scaler_value)
    with algo_cols[2]:
        st.metric("PCA", "Ya" if algorithm_info.get(
            "use_pca", False) else "Tidak")

# Create a comprehensive cluster summary
    cluster_summary = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": algorithm_info.get("algorithm", "Unknown"),
            "scaler": algorithm_info.get("scaler", "Unknown"),
            "use_pca": algorithm_info.get("use_pca", False),
            "sample_count": len(cluster_df)
        },
        "domains": {
            "fa": "Financial Attitude",
            "fb": "Financial Behavior",
            "fk": "Financial Knowledge",
            "m": "Materialism"
        },
        "question_mapping": question_mapping,
        "clusters": {}
    }

    # Find cluster column - check several possible cluster column names
    cluster_column = None
    possible_cluster_columns = ['cluster', 'Cluster', 'cluster_labels',
                                'fa_cluster', 'fb_cluster', 'fk_cluster', 'm_cluster']

    for col in possible_cluster_columns:
        if col in cluster_df.columns:
            cluster_column = col
            # st.info(f"Using cluster column: {cluster_column}")

    # If no cluster column is found, try to detect it by name pattern
    if not cluster_column:
        for col in cluster_df.columns:
            if 'cluster' in col.lower():
                cluster_column = col
                st.info(f"Detected cluster column: {cluster_column}")
                break

    # If still no cluster column, create a fallback
    if not cluster_column:
        st.warning(
            f"No cluster column found. Available columns: {list(cluster_df.columns)}")
        st.write("Using row index as fallback clustering.")
        # Create 3 artificial clusters
        cluster_df['temp_cluster'] = (cluster_df.index % 3)
        cluster_column = 'temp_cluster'

    # Update cluster count and feature count
    cluster_count = len(cluster_df[cluster_column].unique())
    feature_columns = [col for col in cluster_df.columns
                       if 'cluster' not in col.lower() and col != cluster_column]
    feature_count = len(feature_columns)

    # Update the metadata
    cluster_summary["metadata"]["cluster_count"] = cluster_count
    cluster_summary["metadata"]["feature_count"] = feature_count

    # Display updated summary
    st.write(f"**Jumlah Kluster:** {cluster_count}")
    st.write(f"**Jumlah Fitur:** {feature_count}")
    st.dataframe(cluster_df, use_container_width=True)

    # Process each cluster
    # st.subheader("Analisis Cluster per Domain")

    # Define domains with readable names
    domain_dict = {
        "fa": "Financial Attitude",
        "fb": "Financial Behavior",
        "fk": "Financial Knowledge",
        "m": "Materialism"
    }

    # Detect domain-specific cluster columns
    domain_cluster_columns = {}
    for domain in domain_dict.keys():
        possible_cols = [f'{domain}_cluster',
                         f'cluster_{domain}', f'{domain.upper()}_cluster']
        for col in cluster_df.columns:
            if col in possible_cols or (domain in col.lower() and 'cluster' in col.lower()):
                domain_cluster_columns[domain] = col
                # st.info(
                #     f"Found cluster column for {domain_dict[domain]}: {col}")
                break

    # If no domain-specific columns found, use the general cluster column
    if not domain_cluster_columns:
        for domain in domain_dict.keys():
            domain_cluster_columns[domain] = cluster_column
        st.info(
            f"Using general cluster column for all domains: {cluster_column}")

    # Count clusters per domain and add to summary
    domain_cluster_counts = {}
    for domain, col in domain_cluster_columns.items():
        if col and col in cluster_df.columns:
            unique_clusters = sorted(cluster_df[col].unique())
            domain_cluster_counts[domain] = len(unique_clusters)

            # Add to cluster summary
            if f"{domain}_clusters" not in cluster_summary:
                cluster_summary[f"{domain}_clusters"] = {}

            cluster_summary[f"{domain}_cluster_count"] = len(unique_clusters)
            cluster_summary[f"{domain}_cluster_ids"] = [
                int(x) for x in unique_clusters]

    # Display cluster counts per domain
    st.subheader("Jumlah Cluster per Domain")
    domain_cols = st.columns(len(domain_dict))
    for i, (domain, name) in enumerate(domain_dict.items()):
        with domain_cols[i]:
            count = domain_cluster_counts.get(domain, 0)
            st.metric(
                name, count, help=f"Domain {name} memiliki {count} cluster")

    # Process each domain
    for domain, domain_name in domain_dict.items():
        cluster_col = domain_cluster_columns.get(domain)
        if not cluster_col or cluster_col not in cluster_df.columns:
            st.warning(
                f"Tidak dapat menemukan kolom cluster untuk domain {domain_name}")
            continue

        st.subheader(f"{domain_name} Clusters")

        # Get domain-specific features
        domain_features = [
            col for col in feature_columns if col.startswith(domain)]
        if not domain_features:
            st.info(f"Tidak ada fitur untuk domain {domain_name}")
            continue

        st.write(
            f"Ditemukan {len(domain_features)} fitur untuk domain {domain_name}")

        # Process each cluster within this domain
        for cluster_id in sorted(cluster_df[cluster_col].unique()):
            # Get data for this cluster
            cluster_data = cluster_df[cluster_df[cluster_col] == cluster_id]

            # Create expandable section
            with st.expander(f"{domain_name} - Cluster {cluster_id} ({len(cluster_data)} samples, {len(cluster_data)/len(cluster_df)*100:.1f}%)"):
                # Calculate statistics for domain-specific features
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"#### Statistik Cluster {cluster_id}")
                    st.write(
                        f"**Ukuran:** {len(cluster_data)} sampel ({len(cluster_data)/len(cluster_df)*100:.1f}%)")

                    feature_means = cluster_data[domain_features].mean()

                    training_key = f"domain_mean_{domain}_cluster_{cluster_id}"
                    domain_avg = None

                    possible_keys = [
                        f"domain_mean_{domain}_cluster_{cluster_id}",
                        f"{domain}_cluster_{cluster_id}_mean",
                        f"{domain}_{cluster_id}_mean",
                        f"cluster_{domain}_{cluster_id}_mean"
                    ]

                    # Coba semua kemungkinan key
                    if 'cluster_statistics' in st.session_state:
                        for key in possible_keys:
                            if key in st.session_state.cluster_statistics:
                                domain_avg = st.session_state.cluster_statistics[key]
                                st.success(f"Nilai dari Training: {key}")
                                break

                    if domain_avg is None:
                        for key in possible_keys:
                            if key in st.session_state:
                                domain_avg = st.session_state[key]
                                st.success(f"Nilai dari Session State: {key}")
                                break

                    # Jika masih tidak ditemukan, cek domain_averages jika ada
                    if domain_avg is None and 'domain_averages' in st.session_state:
                        key = f"{domain}_cluster_{cluster_id}"
                        if key in st.session_state.domain_averages:
                            domain_avg = st.session_state.domain_averages[key]
                            st.success(f"Nilai dari domain_averages: {key}")

                    if domain_avg is None:
                        st.warning(
                            f"Nilai tidak ditemukan di Training. Menghitung ulang...")
                        domain_avg = feature_means.mean() if not feature_means.empty else 0

                        if 'domain_averages' not in st.session_state:
                            st.session_state.domain_averages = {}
                        st.session_state.domain_averages[f"{domain}_cluster_{cluster_id}"] = domain_avg

                    st.metric(f"Rata-rata {domain_name}", f"{domain_avg:.2f}")

                    # Create heatmap of feature means
                    if len(feature_means) > 0:
                        fig = px.bar(
                            x=feature_means.index,
                            y=feature_means.values,
                            title=f"Nilai Fitur untuk {domain_name} Cluster {cluster_id}",
                            labels={"x": "Fitur", "y": "Rata-rata"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write(f"#### Fitur-fitur Penting")

                    if not feature_means.empty:
                        # Top features
                        st.write("**Fitur Tertinggi:**")
                        top_n = min(5, len(feature_means))
                        top_features = feature_means.nlargest(top_n).to_dict()

                        # Check if we need to add warnings to any top features
                        has_reverse_fa = any(feature in [
                            'fa8', 'fa9', 'fa10', 'fa11', 'fa12', 'fa13', 'fa14', 'fa15'] for feature in top_features)
                        has_reverse_m = any(feature in [
                                            'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'] for feature in top_features)
                        has_fk = any(feature.startswith('fk')
                                     for feature in top_features)

                        # Display appropriate warnings based on features
                        if has_reverse_fa:
                            st.warning(
                                "⚠️ **FA8-FA15 menggunakan reverse-scoring!**")

                        if has_reverse_m:
                            st.warning(
                                "⚠️ **M2-M9 menggunakan reverse-scoring!** ")

                        if has_fk:
                            st.warning(
                                "⚠️ **FK pada skala 0.0-1.0.** Nilai mendekati 1.0 menunjukkan pengetahuan finansial yang sangat baik.")

                        # Show the features
                        for feature, value in top_features.items():
                            # Add visual indicator for reverse-scored items
                            indicator = ""
                            if feature in ['fa8', 'fa9', 'fa10', 'fa11', 'fa12', 'fa13', 'fa14', 'fa15', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']:
                                indicator = "⚠️ "  # Warning emoji
                            st.write(
                                f"- {indicator}{feature}: {value:.2f} - {question_mapping.get(feature, '')}")

                        # Bottom features
                        st.write("**Fitur Terendah:**")
                        bottom_n = min(5, len(feature_means))
                        bottom_features = feature_means.nsmallest(
                            bottom_n).to_dict()

                        # Check if we need to add warnings to any bottom features
                        has_reverse_fa = any(feature in [
                            'fa8', 'fa9', 'fa10', 'fa11', 'fa12', 'fa13', 'fa14', 'fa15'] for feature in bottom_features)
                        has_reverse_m = any(feature in [
                                            'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'] for feature in bottom_features)
                        has_fk = any(feature.startswith('fk')
                                     for feature in bottom_features)

                        # Display appropriate warnings based on features
                        if has_reverse_fa:
                            st.warning(
                                "⚠️ **FA8-FA15 menggunakan reverse-scoring!**")

                        if has_reverse_m:
                            st.warning(
                                "⚠️ **M2-M9 menggunakan reverse-scoring!**=")

                        if has_fk:
                            st.warning(
                                "⚠️ **FK pada skala 0.0-1.0.** Nilai mendekati 0.0 menunjukkan pengetahuan finansial yang kurang.")

                        # Show the features
                        for feature, value in bottom_features.items():
                            # Add visual indicator for reverse-scored items
                            indicator = ""
                            if feature in ['fa8', 'fa9', 'fa10', 'fa11', 'fa12', 'fa13', 'fa14', 'fa15', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']:
                                indicator = "⚠️ "
                            st.write(
                                f"- {indicator}{feature}: {value:.2f} - {question_mapping.get(feature, '')}")

                # Add to cluster summary
                domain_cluster_key = f"{domain}_cluster_{cluster_id}"
                cluster_summary[f"{domain}_clusters"][domain_cluster_key] = {
                    "size": int(len(cluster_data)),
                    "percentage": f"{len(cluster_data) / len(cluster_df) * 100:.1f}%",
                    "domain_average": float(domain_avg),
                    "top_features": make_json_serializable(top_features),
                    "bottom_features": make_json_serializable(bottom_features),
                    "all_features": make_json_serializable(feature_means.to_dict())
                }

# Create cross-domain pattern analysis
    st.subheader("Analisis Pola Lintas Domain")

    # Find all active domain cluster columns
    active_domains = [domain for domain,
                      col in domain_cluster_columns.items() if col in cluster_df.columns]

    if len(active_domains) >= 2:
        # Create pattern data
        pattern_data = pd.DataFrame()

        # Copy cluster columns from each domain
        for domain, col in domain_cluster_columns.items():
            if col in cluster_df.columns:
                pattern_data[domain] = cluster_df[col]

        # Calculate patterns
        pattern_counts = pattern_data.groupby(
            list(pattern_data.columns)).size().reset_index(name='count')
        pattern_counts['percentage'] = (
            pattern_counts['count'] / len(pattern_data) * 100).round(2)
        pattern_counts = pattern_counts.sort_values('count', ascending=False)

        # Display patterns
        st.write("#### Pola Kombinasi Cluster Teratas")
        st.dataframe(pattern_counts.head(10), use_container_width=True)

        # Store in session state for AI analysis
        st.session_state.cluster_pattern_counts = pattern_counts

        # Create pattern labels
        pattern_counts['pattern_label'] = pattern_counts.apply(
            lambda row: " | ".join(
                [f"{domain.upper()}:{int(row[domain])}" for domain in active_domains]),
            axis=1
        )

        # Visualize top patterns
        top_patterns = pattern_counts.head(min(10, len(pattern_counts)))
        fig = px.bar(
            top_patterns,
            y='pattern_label',
            x='percentage',
            title='10 Pola Kombinasi Cluster Teratas',
            labels={'pattern_label': 'Kombinasi Cluster',
                    'percentage': 'Persentase Responden (%)'},
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create heatmap visualization for domain pairs
        st.write("#### Visualisasi Pola Antar Domain")
        domain_pair = st.selectbox(
            "Pilih pasangan domain untuk visualisasi:",
            options=[f"{d1.upper()} vs {d2.upper()}" for i, d1 in enumerate(
                active_domains) for d2 in active_domains[i+1:]]
        )

        if domain_pair:
            d1, d2 = domain_pair.split(" vs ")[0].lower(
            ), domain_pair.split(" vs ")[1].lower()

            # Create pivot table for heatmap
            pivot_data = pattern_counts.pivot_table(
                index=d1,
                columns=d2,
                values='percentage',
                aggfunc='sum'
            ).fillna(0)

            # Create heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(
                    x=f"{domain_dict[d2]} Cluster",
                    y=f"{domain_dict[d1]} Cluster",
                    color="% Responden"
                ),
                title=f"Distribusi Pola: {domain_dict[d1]} vs {domain_dict[d2]}",
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Minimal diperlukan 2 domain dengan data clustering untuk analisis pola")


    st.subheader("Ringkasan Cluster per Domain")

    cluster_summary["clusters"] = {}

    # Process each domain separately
    for domain, domain_name in domain_dict.items():
        domain_col = domain_cluster_columns.get(domain)
        if not domain_col or domain_col not in cluster_df.columns:
            st.warning(
                f"Tidak dapat menemukan kolom cluster untuk domain {domain_name}")
            continue

        # Get unique clusters for this domain
        unique_clusters = sorted(cluster_df[domain_col].unique())

        st.write(f"### Cluster {domain_name} ({len(unique_clusters)} cluster)")

        # Process each cluster for this domain
        for cluster_id in unique_clusters:
            # Get data for this cluster
            cluster_data = cluster_df[cluster_df[domain_col] == cluster_id]

            # Get domain-specific features
            domain_features = [
                col for col in feature_columns if col.startswith(domain)]

            if not domain_features:
                continue

            # Calculate feature means
            feature_means = cluster_data[domain_features].mean()
            domain_avg = feature_means.mean() if not feature_means.empty else 0

            # Get top and bottom features
            top_features = feature_means.nlargest(
                min(5, len(feature_means))).to_dict() if not feature_means.empty else {}
            bottom_features = feature_means.nsmallest(
                min(5, len(feature_means))).to_dict() if not feature_means.empty else {}

            # Add to cluster summary with domain-specific key
            cluster_key = f"{domain}_cluster_{cluster_id}"
            cluster_summary["clusters"][cluster_key] = {
                "domain": domain,
                "domain_name": domain_name,
                "cluster_id": int(cluster_id),
                "size": int(len(cluster_data)),
                "percentage": f"{len(cluster_data) / len(cluster_df) * 100:.1f}%",
                "domain_average": float(domain_avg),
                "highest_scoring_features": make_json_serializable(top_features),
                "lowest_scoring_features": make_json_serializable(bottom_features),
                "all_features": make_json_serializable(feature_means.to_dict())
            }

            st.info(
                f"{domain_name} Cluster {cluster_id}: {len(cluster_data)} sampel ({len(cluster_data)/len(cluster_df)*100:.1f}%)")

        # Add divider between domains
        st.markdown("---")

    # Summarize the number of clusters per domain
    st.subheader("Statistik Cluster per Domain")
    domain_stats = []
    for domain, domain_name in domain_dict.items():
        domain_col = domain_cluster_columns.get(domain)
        if domain_col and domain_col in cluster_df.columns:
            unique_clusters = sorted(cluster_df[domain_col].unique())
            domain_stats.append({
                "Domain": domain_name,
                "Jumlah Cluster": len(unique_clusters),
                "Cluster IDs": ", ".join(map(str, unique_clusters))
            })

    # Display as a table
    if domain_stats:
        st.table(pd.DataFrame(domain_stats))
    else:
        st.warning("Tidak ada data cluster per domain yang tersedia")
# Add ANOVA results if available
if st.session_state.get('anova_results') is not None:
    anova_results = st.session_state.anova_results

    # Process ANOVA results by domain
    significant_features = {}

    # Check the structure of ANOVA results
    if isinstance(anova_results, dict):
        # Process by domain
        for domain, results in anova_results.items():
            domain_significant = []

            for result in results:
                if isinstance(result, dict) and result.get("significant", False):
                    domain_significant.append({
                        "feature": result.get("feature", "Unknown"),
                        "f_statistic": make_json_serializable(result.get("f_statistic", 0)),
                        "p_value": make_json_serializable(result.get("p_value", 1.0))
                    })

            if domain_significant:
                # Sort by F-statistic
                domain_significant = sorted(
                    domain_significant, key=lambda x: x.get("f_statistic", 0), reverse=True)
                significant_features[domain] = domain_significant

    # Add significant features to summary
    if significant_features:
        cluster_summary["significant_features"] = significant_features
    else:
        cluster_summary["significant_features"] = {
            "note": "No significant features found or ANOVA results in unexpected format"}

# Add silhouette score if available
if st.session_state.get('silhouette_results') is not None:
    try:
        silhouette_score = float(st.session_state.silhouette_results)
        cluster_summary["silhouette_score"] = {
            "score": silhouette_score,
            "interpretation": "Good separation" if silhouette_score > 0.5 else
            "Reasonable structure" if silhouette_score > 0.3 else
            "Weak structure" if silhouette_score > 0 else
            "Potential misclassification"
        }
    except Exception as e:
        cluster_summary["silhouette_score"] = {
            "error": str(e),
            "note": "Could not process silhouette score"
        }

# Display in tabs
with tabs[1]:
    # Show a visual summary of each cluster
    st.subheader("Ringkasan per Kluster")

    # Check if we have any clusters to display
    if not cluster_summary["clusters"]:
        st.warning(
            "Tidak ada kluster yang ditemukan untuk dianalisis. Silakan periksa proses clustering Anda.")
    else:
        # Get number of clusters
        num_clusters = len(cluster_summary["clusters"])

        # Create a flexible layout based on number of clusters
        if num_clusters > 0:
            # Use a reasonable number of columns (max 4 per row for readability)
            cols_per_row = min(num_clusters, 4)
            rows_needed = (num_clusters + cols_per_row -
                           1) // cols_per_row  # Ceiling division

            # Process clusters in batches
            cluster_items = list(cluster_summary["clusters"].items())

            for row in range(rows_needed):
                start_idx = row * cols_per_row
                end_idx = min(start_idx + cols_per_row, num_clusters)
                clusters_in_row = cluster_items[start_idx:end_idx]

                # Create columns for this row
                if clusters_in_row:  # Check that we have clusters to display
                    row_cols = st.columns(len(clusters_in_row))

                    # Fill each column with a cluster metric
                    for i, (cluster_name, cluster_data) in enumerate(clusters_in_row):
                        with row_cols[i]:
                            st.metric(
                                cluster_name,
                                f"{cluster_data['size']} samples",
                                f"{cluster_data['percentage']}"
                            )
        else:
            st.info("Tidak ada kluster tersedia untuk ditampilkan")

with tabs[2]:
    st.subheader("Fitur Signifikan (Hasil ANOVA)")

    if "significant_features" in cluster_summary:
        if isinstance(cluster_summary["significant_features"], dict) and "note" not in cluster_summary["significant_features"]:
            # Create tabs for each domain
            domain_tabs = st.tabs([cluster_summary["domains"].get(
                domain, domain) for domain in cluster_summary["significant_features"].keys()])

            for i, (domain, features) in enumerate(cluster_summary["significant_features"].items()):
                with domain_tabs[i]:
                    # Create DataFrame
                    features_df = pd.DataFrame(features)

                    # Add descriptions
                    features_df["Description"] = features_df["feature"].apply(
                        lambda x: question_mapping.get(x, ""))

                    st.dataframe(features_df, use_container_width=True)

                    # Create bar chart of F-statistics
                    if len(features) > 0:
                        fig = px.bar(
                            features_df.sort_values(
                                "f_statistic", ascending=True).tail(10),
                            y="feature",
                            x="f_statistic",
                            title=f"Top 10 Significant Features in {cluster_summary['domains'].get(domain, domain)}",
                            labels={"feature": "Feature",
                                    "f_statistic": "F-statistic"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(cluster_summary["significant_features"].get(
                "note", "No significant features data available"))
    else:
        st.info("No significant features data available")

with tabs[3]:
    st.subheader("Data JSON untuk AI Analysis")

    # Add cluster pattern counts to the summary if available
    if 'cluster_pattern_counts' in st.session_state and st.session_state.cluster_pattern_counts is not None:
        try:
            # Convert pattern counts to serializable format
            pattern_counts = make_json_serializable(
                st.session_state.cluster_pattern_counts)

            # Add to the cluster summary
            cluster_summary["pattern_counts"] = pattern_counts

            st.success(
                "✅ Successfully added cluster pattern counts to the JSON data")
        except Exception as e:
            st.warning(f"Could not process cluster pattern counts: {e}")
            # Add error info to summary
            cluster_summary["pattern_counts_error"] = str(e)
    else:
        st.info("Note: Cluster pattern counts not available in session state")

    if 'domain_cluster_results' in st.session_state and st.session_state.domain_cluster_results is not None:
        try:
            domain_results = st.session_state.domain_cluster_results

            # Initialize domain metrics structure
            domain_metrics = {
                "silhouette_analysis": {
                    "domain_scores": {},
                    "overall_average": None,
                    "best_domain": None,
                    "best_score": -1
                },
                "optimal_clusters": {},
                "davies_bouldin": {},
                "explained_variance": {}
            }

            # Extract all metrics for each domain
            domain_silhouette_scores = []

            for domain, results in domain_results.items():
                if isinstance(results, dict):
                    # Extract silhouette
                    if 'silhouette' in results:
                        score = float(results['silhouette'])
                        domain_metrics["silhouette_analysis"]["domain_scores"][domain] = {
                            "score": score,
                            "domain_name": cluster_summary["domains"].get(domain, domain),
                            "interpretation": "Good separation" if score > 0.5 else
                            "Reasonable structure" if score > 0.3 else
                            "Weak structure" if score > 0 else
                            "Potential misclassification"
                        }
                        domain_silhouette_scores.append(score)

                        # Track best domain
                        if score > domain_metrics["silhouette_analysis"]["best_score"]:
                            domain_metrics["silhouette_analysis"]["best_score"] = score
                            domain_metrics["silhouette_analysis"]["best_domain"] = domain

                    # Extract optimal k
                    if 'optimal_k' in results:
                        domain_metrics["optimal_clusters"][domain] = int(
                            results['optimal_k'])

                    # Extract Davies-Bouldin index
                    if 'davies_bouldin' in results:
                        domain_metrics["davies_bouldin"][domain] = float(
                            results['davies_bouldin'])

                    # Extract explained variance
                    if 'explained_variance' in results:
                        domain_metrics["explained_variance"][domain] = make_json_serializable(
                            results['explained_variance'])

            # Calculate overall average silhouette score
            if domain_silhouette_scores:
                domain_metrics["silhouette_analysis"]["overall_average"] = sum(
                    domain_silhouette_scores) / len(domain_silhouette_scores)

            # Add to cluster summary
            cluster_summary["domain_metrics"] = domain_metrics

            st.success(
                "✅ Successfully added domain metrics including silhouette scores")
        except Exception as e:
            st.warning(f"Could not fully process domain metrics: {e}")
            # Add error info
            cluster_summary["domain_metrics_error"] = str(e)
    else:
        st.info("Note: domain_cluster_results not available in session state")

    # Convert to JSON string
    cluster_json = json.dumps(
        make_json_serializable(cluster_summary), indent=2)

    # Show a sample
    st.code(cluster_json[:1000] + "...", language="json")

    # Download button
    st.download_button(
        "Download Complete JSON",
        cluster_json,
        "cluster_analysis_data.json",
        "application/json"
    )

    # Add a separate download just for pattern counts if available
    if 'cluster_pattern_counts' in st.session_state and st.session_state.cluster_pattern_counts is not None:
        pattern_counts_json = json.dumps(make_json_serializable(
            st.session_state.cluster_pattern_counts), indent=2)
        st.download_button(
            "Download Pattern Counts JSON Only",
            pattern_counts_json,
            "cluster_pattern_counts.json",
            "application/json"
        )

# --------------------------------------
# 5. AI PROMPT ENGINEERING
# --------------------------------------
st.subheader("Analisis menggunakan Kecerdasan Buatan")


prompt_text = f"""
# DATA DAN KONTEKS ANALISIS

## KONTEKS MASALAH FINANSIAL DI INDONESIA
- Fenomena "Fear of Missing Out" (FOMO) yang mendorong konsumerisme berlebihan
- Tren Pinjaman Online Tidak Bijak di kalangan generasi muda
- Berdasarkan data OJK dan Databoks (2023), pengguna pinjaman online didominasi oleh kelompok usia 19-34 tahun (Gen Z dan Milenial)
- Masalah literasi keuangan yang rendah berdasarkan Survei Nasional Literasi Keuangan 2019

## INSTRUKSI KHUSUS UNTUK ANALISIS:
Analisis ini HARUS berfokus pada kombinasi pola cluster yang ditemukan dalam data "pattern_counts" dan dipadukan dengan karakteristik cluster ("clusters"). Seluruh analisis dan rekomendasi HARUS berdasarkan pola yang teridentifikasi dalam data, bukan berdasarkan asumsi umum.

## DATA CLUSTERING YANG DIANALISIS
{json.dumps(make_json_serializable(cluster_summary), indent=2)}

# PETUNJUK ANALISIS SANGAT PENTING - BACALAH DENGAN SEKSAMA

## PANDUAN ANALISIS PATTERN_COUNTS:
1. Data pattern_counts menunjukkan DISTRIBUSI AKTUAL dari pola cluster di setiap domain (Financial Attitude, Financial Behavior, Financial Knowledge, Materialism)
2. Setiap baris dalam pattern_counts merepresentasikan KOMBINASI KELOMPOK dimana responden berada di setiap domain
3. WAJIB menggunakan pattern_counts sebagai DASAR UTAMA analisis
4. Perhatikan kombinasi cluster yang paling sering muncul dan berikan analisis KHUSUS untuk pola-pola tersebut
5. Buat koneksi langsung antara pola dalam pattern_counts dengan karakteristik yang ditemukan dalam data "clusters"

## PETUNJUK EKSPLISIT UNTUK SETIAP BAGIAN ANALISIS:
1. Setiap pernyataan HARUS merujuk langsung ke kombinasi pola dalam pattern_counts
2. Hindari generalisasi yang tidak didukung pattern_counts
3. Gunakan format: "Berdasarkan pattern_counts, [X]% responden menunjukkan pola [jelaskan pola], yang mengindikasikan [interpretasi]"
4. Saat membahas karakteristik cluster, SELALU sajikan data pola spesifik terlebih dahulu
5. WAJIB menjelaskan keterkaitan antar domain berdasarkan pola cluster yang teridentifikasi

# TUGAS ANALISIS DENGAN PATTERN_COUNTS

Berdasarkan data clustering di atas, silakan berikan analisis terstruktur berikut:
## PENTING - INTERPRETASI SKALA PENILAIAN:

### Skala Materialisme (Label 'm'):
- **Inverse Scoring**: Nilai RENDAH menunjukkan tingkat materialisme yang TINGGI
- **Interpretasi Spesifik**: Semakin rendah skor pada fitur 'm1' hingga 'm9', semakin tinggi kecenderungan materialisme
- **Implikasi Analisis**: Cluster dengan rata-rata rendah pada fitur 'm' harus diinterpretasikan sebagai kelompok dengan tingkat materialisme tinggi
- **Contoh**: Skor m = 2.1 menunjukkan tingkat materialisme lebih tinggi dibandingkan skor m = 4.3

### Skala Pengetahuan Finansial (Label 'fk'):
- **Rentang Nilai**: 0.0 hingga 1.0
- **Interpretasi**: Nilai mendekati 1.0 menunjukkan pengetahuan finansial yang sangat baik
- **Standar Kecukupan**: Nilai > 0.5 dianggap menunjukkan tingkat pengetahuan finansial yang memadai
- **Kriteria Evaluasi**: Nilai < 0.5 mengindikasikan kebutuhan intervensi edukasi finansial
- **Contoh**: Skor fk = 0.75 menunjukkan pengetahuan finansial yang baik, sementara 0.35 menunjukkan pengetahuan yang kurang

### Implikasi untuk Analisis Pola:
Saat menganalisis kombinasi pola dalam pattern_counts, WAJIB memperhatikan arah interpretasi yang berbeda ini, terutama ketika membandingkan domain materialisme dengan domain lainnya.

1. ANALISIS POLA DOMINAN:
   - Identifikasi semua pola kombinasi cluster paling umum dari pattern_counts
   - Jelaskan persentase responden untuk setiap pola
   - Analisis implikasi dari pola-pola ini terhadap literasi keuangan

2. KARAKTERISTIK CLUSTER BERDASARKAN PATTERN_COUNTS:
   - Nama deskriptif untuk setiap cluster berdasarkan pattern_counts dan data cluster
   - Karakteristik utama tiap cluster dengan RUJUKAN SPESIFIK ke domain (FA, FB, FK, M)
   - Bagaimana cluster berinteraksi antar domain (berdasarkan pattern_counts)

3. REKOMENDASI BERBASIS POLA:
   - Rekomendasi pendidikan keuangan untuk setiap POLA KOMBINASI (bukan hanya per cluster)
   - Prioritas intervensi berdasarkan frekuensi pola dalam pattern_counts
   - Intervensi spesifik untuk kombinasi cluster yang problematik

4. ANALISIS STATISTIK:
   - Analisis skor silhouette dan implikasinya (jika tersedia)
   - Diskusikan signifikansi hasil ANOVA dalam konteks pattern_counts
   - Korelasi antara domain berdasarkan pola cluster

5. FENOMENA FINANSIAL INDONESIA:
   - Kaitkan pola spesifik dalam pattern_counts dengan FOMO dan pinjaman online
   - Berikan contoh konkret bagaimana pola tertentu rentan terhadap masalah finansial
   - Sertakan referensi dari jurnal/buku jika memungkinkan

6. RINGKASAN BERBASIS POLA:
   - Temuan utama dari analisis pattern_counts
   - Implikasi untuk literasi keuangan di Indonesia
   - Arah penelitian masa depan berdasarkan pola yang ditemukan

PERHATIAN: Seluruh analisis HARUS menggunakan data pattern_counts sebagai referensi utama dan disertai bukti kuantitatif dari data. Hindari pernyataan umum tanpa merujuk pada pola spesifik dalam data.

Format respons dengan judul, poin-poin yang jelas, dan dalam Bahasa Indonesia.
"""
# Store the prompt for reference
st.session_state.prompt_text = prompt_text

# Advanced options
# with st.expander("Advanced Prompt Options"):
#     custom_prompt = st.text_area(
#         "Customize AI Analysis Prompt",
#         value=prompt_text,
#         height=300,
#         help="Edit the prompt to customize the AI analysis"
#     )

#     if custom_prompt != prompt_text:
#         prompt_text = custom_prompt
#         st.success("Custom prompt will be used")

# --------------------------------------
# 6. AI ANALYSIS GENERATION
# --------------------------------------

# Button to generate analysis
if st.button("Generate AI Analysis", type="primary"):
    with st.spinner("Generating insights with Gemini AI..."):
        try:
            # Request headers
            headers = {'Content-Type': 'application/json'}

            # Request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt_text}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 8192,
                }
            }

            # Make the API call
            response = requests.post(url, headers=headers, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()

                # Extract just the generated text
                if "candidates" in result and result["candidates"]:
                    generated_text = result["candidates"][0]["content"]["parts"][0]["text"]

                    # Store the results
                    st.session_state.ai_analysis = generated_text
                    st.session_state.analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Display the results
                    st.success("✅ AI Analysis completed successfully!")

                    # Create an expander for the complete analysis
                    with st.expander("Complete AI Analysis", expanded=True):
                        st.markdown(generated_text)

                    # Allow downloading the analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Analysis as Text",
                            generated_text,
                            "ai_cluster_analysis.txt",
                            "text/plain"
                        )
                    with col2:
                        # Format as markdown file
                        markdown_content = f"# AI Cluster Analysis\n\n*Generated on {st.session_state.analysis_timestamp}*\n\n{generated_text}"
                        st.download_button(
                            "Download Analysis as Markdown",
                            markdown_content,
                            "ai_cluster_analysis.md",
                            "text/markdown"
                        )
                else:
                    st.error("No content generated. Check API response.")
                    st.json(result)
            else:
                st.error(f"Error: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Exception occurred: {e}")
            st.info("Please check your API key and internet connection.")


elif "ai_analysis" in st.session_state and st.session_state.ai_analysis:
    st.success(
        f"✅ Using previously generated analysis (from {st.session_state.analysis_timestamp})")

    with st.expander("Previous AI Analysis", expanded=True):
        st.markdown(st.session_state.ai_analysis)

    # Allow downloading the previous analysis
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Analysis as Text",
            st.session_state.ai_analysis,
            "ai_cluster_analysis.txt",
            "text/plain"
        )
    with col2:
        # Format as markdown file
        markdown_content = f"# AI Cluster Analysis\n\n*Generated on {st.session_state.analysis_timestamp}*\n\n{st.session_state.ai_analysis}"
        st.download_button(
            "Download Analysis as Markdown",
            markdown_content,
            "ai_cluster_analysis.md",
            "text/markdown"
        )


# with st.expander("Debug Value Consistency"):
#     st.subheader("Verify Value Consistency")

#     if 'domain_averages' in st.session_state:
#         st.write("### Domain Averages from Calculations")
#         for key, value in st.session_state.domain_averages.items():
#             st.write(f"{key}: {value}")

#     if 'cluster_statistics' in st.session_state:
#         st.write("### Values from Training Page")
#         training_values = {}
#         for key in st.session_state.cluster_statistics:
#             if "_mean" in key or "mean_" in key:
#                 training_values[key] = st.session_state.cluster_statistics[key]

#         st.dataframe(pd.DataFrame(
#             training_values.items(), columns=["Key", "Value"]))

#     st.write("### Values Used in AI Analysis JSON")
#     domain_values = {}
#     for domain in ["fa", "fb", "fk", "m"]:
#         if f"{domain}_clusters" in cluster_summary:
#             for key, data in cluster_summary[f"{domain}_clusters"].items():
#                 if "domain_average" in data:
#                     domain_values[key] = data["domain_average"]

#     st.dataframe(pd.DataFrame(domain_values.items(),
#                  columns=["Cluster", "Value Used in AI"]))
