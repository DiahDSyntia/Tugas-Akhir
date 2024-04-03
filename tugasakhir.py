import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import re

import streamlit as st

# Judul navbar
st.sidebar.title('Main Menu')

# Pilihan menu dalam bentuk dropdown
menu_selection = st.sidebar.selectbox('Klik Tombol Di bawah ini', ['Home', 'Pre-Pocesssing Data', 'Klasifikasi SVM','Uji Coba'])

# Membuat konten berdasarkan pilihan menu
if menu_selection == 'Home':
    st.title('Selamat Datang di Website Klasifikasi Hipertensi')
    st.write('Hipertensi adalah kondisi yang terjadi ketika tekanan darah naik di atas kisaran normal, biasanya masyarakat menyebutnya darah tinggi. Penyakit hipertensi berkaitan dengan kenaikan tekanan darah di sistolik maupun diastolik. Faktor faktor yang berperan untuk penyakit ini adalah perubahan gaya hidup, asupan makanan dengan kadar lemak tinggi, dan kurangnya aktivitas fisik seperti olahraga')
    st.write('Faktor Faktor Resiko Hipertensi')
    st.write("""
    1. Jenis Kelamin
    2. Usia
    3. Indeks Massa Tubuh
    4. Sistolik
    5. Diastolik
    6. Nafas
    7. Detak Nadi
    """)
    st.markdown("### Data Hipertensi")
    st.write('Data Hipertensi ini merupakan data dari Puskesmas Modopuro, Mojokerto')
    url = "https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv"
    df = pd.read_csv(url)
    st.write(df)

elif menu_selection == 'Pre-Pocesssing Data':
    st.title('Halaman Pre-pocessing Data')
    st.write('Hipertensi')

    st.markdown("### Data Hipertensi")
    st.write('Data Hipertensi ini merupakan data dari Puskesmas Modopuro, Mojokerto')
    url = "https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv"
    df = pd.read_csv(url)
    st.write(df)

    # Tambahkan tombol untuk memicu proses preprocessing
    if st.button('Proses Data'):
        # Menghapus baris dengan nilai yang hilang (NaN)
        df = df.dropna()
        # Menghapus duplikat data
        df = df.drop_duplicates()

        # Mapping for 'Hipertensi'
        df['Diagnosa'] = df['Diagnosa'].map({'HIPERTENSI 1': 1, 'HIPERTENSI 2': '2', 'TIDAK': 0})

        # Tampilkan hasil preprocessing di bawah tombol
        st.write('Data setelah preprocessing:')
        st.write(df)

elif menu_selection == 'Klasifikasi SVM':
    st.title('Halaman Klasifikasi SVM')
    st.write('This is the contact us page.')

elif menu_selection == 'Uji Coba':
    st.title('Halaman Uji Coba')
    st.write('This is the contact us page.')
