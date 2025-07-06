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


st.title("Latar Belakang")
st.markdown(
    """
    <div style="text-align: justify; text-indent: 30px;">
    Selama beberapa dekade, pemerintah di seluruh dunia melakukan dorongan kepada masyarakat untuk meningkatkan keuangan jangka panjang akibat krisis keuangan yang terjadi tahun 2008 [1]. Demi meningkatkan keuangan jangka panjang, diperlukan literasi finansial yang memadahi. Literasi finansial merupakan pemahaman akan pengelolaan keuangan yang baik untuk kehidupan sehari-hari demi meningkatkan kesejahteraan individu maupun masyarakat. Namun, masyarakat di Indonesia memiliki literasi finansial yang rendah.  Hal ini dibuktikan oleh hasil survey Otoritas Jasa Keuangan (OJK) yang menunjukkan bahwa rata-rata literasi finansial masyarakat Indonesia hanya mencapai 49,68%, sementara inklusi keuangan mencapai 85,10%. Hal ini berarti masyarakat Indonesia mempunyai akses ke layanan keuangan, namun pengetahuan dan keterampilan dalam mengelola keuangan masih sangat kurang [2].
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Data dari OJK pada tahun 2023 menunjukkan bahwa sebagian besar permasalahan finansial di Indonesia didominasi oleh rentang usia 19-34 atau generasi Z, dengan tingkat literasi sebesar 49,68% dan inklusi finansial 85,10%. Hal ini menunjukkan bahwa terdapat kesenjangan mengenai pengetahuan keuangan. Generasi Z merupakan generasi yang dilahirkan pada masa resesi ekonomi, hal ini menumbuhkan sifat kehati-hatian dan cemas terhadap risiko serta cenderung mencari suatu informasi melalui internet dan sosial media[3]. Akses terhadap internet tersebut membuat generasi ini menjadi pengguna layanan pinjaman online terbesar [2], dengan 80% dari mereka menggunakan aplikasi fintek dan 60% melakukan transaksi dalam aplikasi tersebut. Penggunaan uang tunai pun menurun 40% dalam 2 tahun terakhir dan 45% dari generasi tersebut melakukan investasi melalui saham. Hal ini menunjukkan ketertarikan generasi tersebut terhadap investasi walaupun terdapat kesenjangan literasi keuangan yang mempengaruhi cara menabung [4].
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Salah satu akibat dari rendahnya literasi finansial adalah konsumerisme. Hal ini mengakibatkan dampak merugikan bagi diri sendiri maupun negara. Salah satu contoh akibat dari konsumerisme adalah krisis keuangan pribadi. Ketika seseorang memiliki utang yang sangat tinggi, sebagian besar pendapatannya akan digunakan untuk membayar utang-utang dan bunga dari utang tersebut. Hal ini dapat mengurangi konsumsi dan investasi, akibatnya, uang yang dihasilkan menjadi tidak dapat digunakan untuk hal yang lebih produktif dan perputaran uang dalam masyarakat menjadi berkurang. Dampak ini juga menyebar hingga skala nasional, di mana menurut studi Kobayashi dan Shirai pada tahun 2017 memaparkan ketika utang menumpuk secara masif, maka perputaran uang akan terfokus untuk pembayaran utang dan bunga bukan pengeluaran untuk barang, jasa, maupun investasi. Akibatnya permintaan barang dan jasa menurun, produksi melambat dan perusahaan, mengurangi tenaga kerja[2].
    </div>
    <br>
    <div style="text-align: justify; text-indent: 30px;">
    Melihat pengetahuan literasi finansial Generasi Z yang rendah dan penggunaan fintek yang kurang bijak akibat kurangnya pengetahuan literasi finansial, diperlukan pendekatan menggunakan data untuk mengetahui perilaku menabung Generasi Z. Model machine learning dengan metode unsupervised learning dipilih karena dapat melihat pola tersembunyi dalam data. Pola tersembunyi ini berupa sikap keuangan, pengetahuan keuangan, perilaku keuangan dan Materialisme. Hal ini akan mengungkapkan perilaku menabung Generasi Z dan melakukan konfirmasi survei OJK. Selain itu, hasil dari model ini juga bermanfaat bagi pemerintah dan pendidik untuk memberikan edukasi lebih dini mengenai pentingnya menabung dan literasi keuangan.   
    </div>
    <br>
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
