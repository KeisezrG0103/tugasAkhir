import streamlit as st
from utils.data_preprocessing import data_loader
st.set_page_config(
    page_title="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)


if 'PCA' not in st.session_state:
    st.session_state.PCA = None
if 'multicollinearity' not in st.session_state:
    st.session_state.multicollinearity = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
# if 'selected_columns' not in st.session_state:
#     st.session_state.selected_columns = selected_columns
if 'current_algorithm' not in st.session_state:
    st.session_state.current_algorithm = None
if 'pca_components' not in st.session_state:
    st.session_state.pca_components = 2
if 'anova_results' not in st.session_state:
    st.session_state.anova_results = None
if 'tukey_results' not in st.session_state:
    st.session_state.tukey_results = None
if 'silhoutte_results' not in st.session_state:
    st.session_state.silhoutte_results = None


if 'df' not in st.session_state:
    st.session_state.df = data_loader('data/ta_dataset.csv')

df = st.session_state.df


header_container = st.container()

with header_container:
    # Use columns with relative proportions instead of fixed spec
    col1, col2 = st.columns([5, 1])

    with col1:
        st.title("Latar Belakang")

    with col2:
        # Vertically center the button
        st.write("")  # Create some space
        st.link_button(
            label="Form Survei",
            url="https://docs.google.com/forms/d/e/1FAIpQLSe49Y8zX6w9C_k-6dLAChkCGrxSwZkL16lXqfwXkS7VIfjxaw/viewform?usp=sharing",
            use_container_width=True,
            type="primary",
        )

st.markdown(
    """
    <div style="text-align: justify; text-indent: 30px;">
    Selama beberapa dekade, pemerintah seluruh dunia mendorong masyarakatnya untuk meningkatkan keuangan jangka panjang. Krisis keuangan tahun 2008 telah mengubah kebijakan pemerintah yang semula mempromosikan pembiayaan murah yang disalahgunakan untuk meningkatkan belanja konsumen menjadi meningkatkan kesadaran konsumen akan kebutuhan keuangan jangka panjang. Padahal, pada masa ini tabungan adalah hal yang penting untuk kesehatan dan pensiun di masa yang akan datang [1]. 
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Upaya ini tidak dapat berjalan tanpa dukungan dari tingkat literasi finansial yang memadai. Literasi finansial merupakan pemahaman akan pengelolaan keuangan yang baik untuk kehidupan sehari-hari demi meningkatkan kesejahteraan individu maupun masyarakat. Namun sayangnya, tingkat literasi finansial di Indonesia masih tergolong rendah. Berdasarkan Organization for Economic Co-operation and Development (OECD) tahun 2023 menunjukkan Indonesia memiliki indeks skor rata-rata dunia sebesar 60. Indeks rata-rata keuangan yang rendah memberikan indikasi bahwa masyarakat Indonesia membuat keputusan keuangan yang kurang optimal. Hasil survei Otoritas Jasa Keuangan (OJK) menunjukkan bahwa rata-rata literasi finansial masyarakat Indonesia hanya mencapai 49,68 persen, sementara inklusi keuangan mencapai 85,10 persen. Hal ini berarti masyarakat Indonesia mempunyai akses ke layanan keuangan, namun pengetahuan dan keterampilan dalam mengelola keuangan masih sangat kurang [2].
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Dalam konteks ini Generasi Z akan lebih dibahas karena Generasi Z memiliki finansial literasi yang di bawah rata-rata. Menurut data dari OJK tahun 2022, literasi finansial Generasi Z Indonesia sebesar 49,68 persen dan inklusi finansial sebesar 85,10 persen, gap antara inklusi finansial dan literasi finansial ini berarti Generasi Z Indonesia kurang dalam hal pengetahuan finansial [3]. Generasi Z merupakan generasi yang dilahirkan pada periode resesi ekonomi dan hidup pada era digital. Generasi Z selalu diberi perlindungan karena tumbuh pada masa resesi ekonomi, sehingga mereka sering merasakan kecemasan bila keadaan tidak berjalan dengan semestinya. Generasi Z juga dinilai lebih berhati dan merasa cemas bila mengetahui risiko yang mungkin muncul pada suatu situasi tertentu dan cenderung terus mencari informasi mengenai hal tersebut baik melalui internet dan media sosial [4]. 
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Dengan akses digital yang memadai membuat Generasi Z memiliki kecenderungan untuk menggunakan layanan keuangan digital atau Finansial Teknologi. Berdasarkan data dari Otoritas Jasa Keuangan dan DataBok tahun 2023, kelompok usia 19-34 tahun merupakan pengguna terbesar layanan pinjaman online [2]. Fenomena ini sejalan dengan penggunaan Fintek oleh Generasi Z di Indonesia, di mana 80 % dari generasi Z menggunakan aplikasi Fintek dan 60 % dari meraka melakukan transaksi pada aplikasi fintek tersebut. Penggunaan uang tunai pun menurun 40 % dalam 2 tahun terakhir. Selain itu, 45 % generasi Z yang aktif dalam penggunaan Fintek telah terlibat dalam investasi saham dimana hal tersebut menunjukkan ketertarikan dalam pengembangan investasi Generasi Z. Meskipun literasi Fintek terus meningkat, masih terdapat kesenjangan dalam literasi finansial yang mempengaruhi cara menabung Generasi Z [5].
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Melihat kesenjangan pengetahuan literasi finansial dan penggunaan Fintek yang kurang bijak, diperlukan pendekatan menggunakan data untuk mengetahui pola dan perilaku menabung Generasi Z. Dalam hal ini metode unsupervised learning dipilih untuk melakukan pengelompokan pola menabung mereka. Data akan diperoleh melalui survei dengan responden Generasi Z yang menempuh pendidikan perkuliahan di semester 1 sampai 8. Data dari survei ini akan menilai tingkat pengetahuan, perilaku dan sikap finansial serta materialisme dari Generasi Z.
    </div>
    
    
    <div style="text-align: justify; text-indent: 20px; margin-top: 20px;">
    [1] D. Brounen, K. G. Koedijk, and R. A. J. Pownall, “Household financial planning and savings behavior,” <i>Journal of International Money and Finance</i>, vol. 69, pp. 95–107, Dec. 2016, doi: <a href="https://doi.org/10.1016/j.jimonfin.2016.06.011" target="_blank">10.1016/j.jimonfin.2016.06.011</a>.
    </div>
    <br>

    <div style="text-align: justify; text-indent: 20px;">
    [2] R. Rp et al., “PENDIDIKAN LITERASI FINANSIAL.”
    </div>
    <br>

    <div style="text-align: justify; text-indent: 20px;">
    [3] “Navigating the Digital Financial Landscape: The Role of Financial Literacy and Digital Payment Behavior in Shaping Financial Management Among Generation Z Student,” <i>Journal of Logistics, Informatics and Service Science</i>, Jun. 2024, doi: <a href="https://doi.org/10.33168/jliss.2024.0716" target="_blank">10.33168/jliss.2024.0716</a>.
    </div>
    <br>

    <div style="text-align: justify; text-indent: 20px;">
    [4] G. Sakitri, “‘Selamat Datang Gen Z, Sang Penggerak Inovasi!’”
    </div>
    <br>

    <div style="text-align: justify; text-indent: 20px;">
    [5] D. Putra Utama and A. D. Sumarna, “Financial Technology Literacy Impact on Gen-Z in Indonesia,” doi: <a href="https://doi.org/10.38035/dijefa.v4i6" target="_blank">10.38035/dijefa.v4i6</a>.
    </div>
    """,
    unsafe_allow_html=True
)
