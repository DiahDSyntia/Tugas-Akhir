import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns

st.write("""
# Klasifikasi Penyakit Hipertensi 
### Metode SVM (Support Vector Machine)
"""
)

tab_titles = [
    "Homepage",
    "Pre-processing Data",
    "Klasifikasi SVM",
    "Klasifikasi SVM",
    "Uji Coba",]

tabs = st.tabs(tab_titles)
with tabs[0]:
    st.write("""
    Air Terjun Dlundung di Kecamatan Trawas, Kabupaten Mojokerto menjadi salah satu destinasi wisata yang sayang untuk dilewatkan. Panorama alamnya memukau dan udaranya yang sejuk membuat liburan terasa singkat di sini.
    """
    )
    st.write("""
    Dlundung Waterfall berada di area Wana Wisata Dlundung. Tepatnya di Dusun/Desa Ketapanrame, Kecamatan Trawas. Wisata alam ini berjarak sekitar 38 kilometer dengan waktu tempuh sekitar 1 jam 10 menit dari Kota Mojokerto. Jika dari Kantor Kecamatan Trawas, Air Terjun Dlundung hanya sekitar 2,4 kilometer.
    """
    )
    st.write("""
    Air Terjun Dlundung mempunyai panorama alam yang eksotis. Karena lokasinya di antara hutan lereng Gunung Welirang yang lumayan lebat. Banyak pohon besar yang tumbuh di sekitarnya. Tak ayal, udara di wisata alam ini terasa sejuk. Sehingga, cocok untuk melepas penat bersama teman, keluarga atau kekasih tercinta.
    """
    )
    st.write("""
    Ketinggian Air Terjun Dlundung sekitar 14 meter. Air yang berjatuhan dari tebing tak terlalu deras. Sehingga, para pengunjung aman untuk bermain di bawahnya. Selain bermain air, para wisatawan juga bisa berswafoto di bawahnya. Dua spot foto yang tak kalah menarik di atas panggung sisi kanan air terjun dan di depan nama Dlundung Waterfall. Spot selfie juga bisa dijumpai di area parkir Air Terjun Dlundung. Yaitu dengan latar belakang nama Dlundung Waterfall dan lebatnya hutan di atasnya. Akses dari area parkir ke air terjun berupa tangga beton dan jalan paving di tengah rindangnya pepohonan.
    """
    )
