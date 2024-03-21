import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns

import streamlit as st

# Menambahkan CSS khusus untuk navbar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: pink; /* Warna latar belakang sidebar */
    }
    .sidebar .sidebar-content .block-container {
        color: #333; /* Warna teks pada pilihan menu */
    }
    .sidebar .sidebar-content .stButton>button {
        background-color: skyblue; /* Warna latar belakang tombol */
        color: white; /* Warna teks tombol */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul navbar
st.sidebar.title('Main Menu')

# Pilihan menu dalam bentuk dropdown
menu_selection = st.sidebar.selectbox('Klik Tombol Di bawah ini', ['Home', 'Pre-Pocesssing Data', 'Klasifikasi SVM','Uji Coba'])

# Membuat konten berdasarkan pilihan menu
if menu_selection == 'Home':
    st.title('Selamat Data di Halaman Home Page')
    st.write('This is the homepage content.')

elif menu_selection == 'Pre-Pocesssing Data':
    st.title('Halaman Pre-pocessing Data')
    st.write('This is the about us page.')

elif menu_selection == 'Klasifikasi SVM':
    st.title('Halaman Klasifikasi SVM')
    st.write('This is the contact us page.')

elif menu_selection == 'Uji Coba':
    st.title('Halaman Uji Coba')
    st.write('This is the contact us page.')
