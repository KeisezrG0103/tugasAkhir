import streamlit as st
import json
import requests
import pandas as pd
import numpy as np

# Helper function to make data JSON serializable


def make_json_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
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


api_key = None

# Try to get API key from secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.success("✅ API key loaded from secrets")
except (KeyError, FileNotFoundError):
    # If not in secrets, check session state or request from user
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # Get API key using secure text input
    api_key = st.text_input(
        "Enter your Gemini API key:",
        value=st.session_state.api_key,
        type="password",
        help="API key not found in secrets. Please enter manually."
    )

    # Store API key in session state to persist between reruns
    if api_key:
        st.session_state.api_key = api_key

if not api_key:
    st.error("❌ API key is required to use this feature.")
    st.info("To avoid entering your API key every time, add it to .streamlit/secrets.toml as: GEMINI_API_KEY = 'your-key-here'")
    st.stop()

# API URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

st.header("Gemini-Powered Cluster Insights")
st.subheader("AI Analysis of Your Financial Knowledge Clusters")

# Check for required session state variables
required_data = {
    'cluster_results_df': 'Cluster results dataframe',
    'summary': 'Model summary',
    'current_algorithm': 'Current algorithm',
    'scaler': 'Data scaling method',
    'anova_results': 'ANOVA results'
}

# Create status indicators
missing_data = []
available_data = []

# Check which data is available
for key, description in required_data.items():
    if key in st.session_state and st.session_state[key] is not None:
        available_data.append(f"{description} ({key})")
    else:
        missing_data.append(description)
        if key not in st.session_state:
            st.session_state[key] = None

# Show what's available and what's missing
if available_data:
    st.success(f"✅ Available data: {', '.join(available_data)}")
else:
    st.warning("⚠️ No analysis data found in session state.")

if missing_data:
    st.warning(f"⚠️ Missing data: {', '.join(missing_data)}")
    st.info("Please complete previous steps to generate all required data.")

# Stop if cluster data is missing (required for analysis)
if st.session_state.get('cluster_results_df') is None:
    st.error("❌ Cluster data is required but missing. Please run clustering first.")
    st.stop()

# Display cluster dataframe
cluster_df = st.session_state.cluster_results_df
st.write(
    f"Cluster data loaded with {len([col for col in cluster_df.columns if col != 'cluster'])} features.")
st.dataframe(cluster_df)

# Create a summary of each cluster
cluster_summary = {
    "legends": {
        "fa": "Financial Attitude",
        "fb": "Financial Behavior",
        "fk": "Financial Knowledge",
        "m": "Materialism"
    },
    "question_mapping": question_mapping
}

# Process each cluster
for cluster in sorted(cluster_df['cluster'].unique()):
    # Get cluster means for each feature
    cluster_data = cluster_df[cluster_df['cluster'] == cluster]
    feature_means = cluster_data.mean().drop('cluster')

    # Get top and bottom features
    length = len(feature_means) - 5  # Fix: correct calculation for length
    top_features = feature_means.nlargest(length).to_dict()
    bottom_features = feature_means.nsmallest(5).to_dict()

    # Add cluster data to summary (with serialization)
    cluster_summary[f"Cluster {cluster}"] = {
        "size": int(len(cluster_data)),
        "percentage": f"{len(cluster_data) / len(cluster_df) * 100:.1f}%",
        "top_features_on_avg_response": make_json_serializable(top_features),
        "bottom_features_on_avg_response": make_json_serializable(bottom_features),
        "Anova_Results": {
            "note": "ANOVA results available but not yet processed",
            "raw_data": make_json_serializable(st.session_state.anova_results)
        }
    }

# Add ANOVA results if available
if st.session_state.get('anova_results') is not None:
    anova_df = st.session_state.anova_results

    # Check if anova_df is a DataFrame - if not, convert it
    if not isinstance(anova_df, pd.DataFrame):
        try:
            # If it's a list of dicts or similar structure
            anova_df = pd.DataFrame(anova_df)
            st.info("Converted ANOVA results from list to DataFrame")
        except Exception as conversion_error:
            st.error(
                f"Could not convert ANOVA results to DataFrame: {conversion_error}")
            # Create a simple representation for display
            if isinstance(anova_df, list):
                st.write("ANOVA Results (List format):")
                for i, item in enumerate(anova_df):
                    st.write(f"Item {i}:", item)
            else:
                st.write("ANOVA Results:", anova_df)

            # Create a basic structure for the summary
            cluster_summary["ANOVA_Results"] = {
                "note": "ANOVA results available but in non-standard format",
                "raw_data": make_json_serializable(anova_df)
            }
            # Skip the rest of the ANOVA processing
            st.error(
                "ANOVA results are in an unexpected format. Skipping detailed analysis.")

    else:
        # Display the ANOVA dataframe
        st.subheader("ANOVA Results")
        st.dataframe(anova_df)

        # Get significant features
        try:
            # Check if 'Significant' column exists
            if 'Significant' in anova_df.columns:
                significant_features = anova_df[anova_df['Significant'] == True].sort_values(
                    'p-value')

                # Add to summary if any significant features exist
                if not significant_features.empty:
                    # Check if all required columns exist
                    required_columns = ['Feature', 'F-statistic', 'p-value']
                    missing_columns = [
                        col for col in required_columns if col not in significant_features.columns]

                    if missing_columns:
                        st.warning(
                            f"Missing columns in ANOVA results: {', '.join(missing_columns)}")
                        # Use available columns
                        available_columns = [
                            col for col in required_columns if col in significant_features.columns]
                        serializable_records = make_json_serializable(
                            significant_features[available_columns].to_dict(
                                'records')
                        )
                    else:
                        serializable_records = make_json_serializable(
                            significant_features[required_columns].to_dict(
                                'records')
                        )

                    cluster_summary["ANOVA_Results"] = {
                        "significant_features_count": int(len(significant_features)),
                        "top_significant_features": serializable_records
                    }
                    st.write(
                        f"Found {len(significant_features)} significant features")
                else:
                    st.info(
                        "No statistically significant features found in the ANOVA analysis.")
            else:
                st.warning(
                    "ANOVA results doesn't contain 'Significant' column")
                # Create a simplified representation
                cluster_summary["ANOVA_Results"] = {
                    "column_names": list(anova_df.columns),
                    "data_summary": f"Contains {len(anova_df)} rows of ANOVA data"
                }
        except Exception as e:
            st.error(f"Error processing ANOVA results: {str(e)}")
            # Add debugging info
            st.write("ANOVA DataFrame columns:", list(anova_df.columns)
                     if hasattr(anova_df, 'columns') else "No columns")
            st.write("ANOVA DataFrame type:", type(anova_df))
else:
    st.info("No ANOVA analysis results available. Run the Analysis page first to perform ANOVA tests.")

# Add silhouette results if available
if st.session_state.get('silhoutte_results') is not None:
    try:
        silhouette_score = float(st.session_state.silhoutte_results)
        cluster_summary["Silhouette_Results"] = {
            "overall_score": silhouette_score,
            "interpretation": "Good separation" if silhouette_score > 0.5 else
            "Reasonable structure" if silhouette_score > 0.3 else
            "Weak structure" if silhouette_score > 0 else
            "Potential misclassification"
        }
    except Exception as e:
        st.error(f"Error processing silhouette results: {str(e)}")

# Display cluster summary as JSON with proper error handling
try:
    # Test JSON serialization first
    json_string = json.dumps(cluster_summary, indent=2)
    st.subheader("Cluster Summary")
    # st.json(cluster_summary)
except TypeError as e:
    st.error(f"JSON serialization error: {str(e)}")
    st.write("Attempting to fix serialization issues...")
    serializable_summary = make_json_serializable(cluster_summary)
    st.json(serializable_summary)
    # Update cluster_summary with the serializable version
    cluster_summary = serializable_summary

# Add algorithm info if available
if st.session_state.get('current_algorithm') is not None:
    st.subheader("Algorithm Information")
    st.write(f"**Algorithm:** {st.session_state.current_algorithm}")
    if st.session_state.get('scaler') is not None:
        st.write(f"**Scaling Method:** {st.session_state.scaler}")

# Create prompt with serializable data
prompt_text = f"""
Indonesian problem : consumerism "Fear of Missing Out" , Fenomena Pinjaman Tidak Bijak
Berdasarkan data OJK dan Databoks (2023),pengguna pinjaman online didominasi oleh kelompok usia 19-34 tahun dan menunjukkan bahwa generasi muda merupakan pengguna terbesar layanan pinjaman online.
You are a financial behavior expert analyzing clustering results from a financial literacy survey for Genz and Millenials in Indonesia. The survey includes various financial knowledge and behavior questions, and the data has been clustered into distinct groups based on their responses and please answer using indonesian language.
Based on these cluster statistics:
{json.dumps(make_json_serializable(cluster_summary), indent=2)}

Please provide:
1. A descriptive name for each cluster based on their financial knowledge patterns
2. Key characteristics that distinguish each cluster
3. Tailored financial education recommendations for each cluster
4. Analyze the silhouette score and its implications for the clustering results
5. Discuss the significance of the ANOVA results and their implications for understanding the clusters
6. proof the cluster with the indonesian consumerism phenomenon (With examples by my question and data)
7. Provide a list of references used in the analysis (if u can give based on journal or book)
8. Provide a summary of the analysis in a clear and concise manner
9. make an analysis from my data and cluster summary

Format your response with clear headings and bullet points.
"""

# Store the prompt for reference
st.session_state.prompt_text = prompt_text

# Button to generate analysis
if st.button("Generate AI Analysis"):
    with st.spinner("Generating insights with Gemini AI..."):
        try:
            # Request headers
            headers = {'Content-Type': 'application/json'}

            # Request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt_text}]
                }]
            }

            # Make the API call
            response = requests.post(url, headers=headers, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()

                # Extract just the generated text
                if "candidates" in result and result["candidates"]:
                    generated_text = result["candidates"][0]["content"]["parts"][0]["text"]

                    # Display the results
                    st.subheader("AI Cluster Analysis")
                    st.markdown(generated_text)

                    # Allow downloading the analysis
                    st.download_button(
                        "Download Analysis",
                        generated_text,
                        "gemini_cluster_analysis.txt"
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
