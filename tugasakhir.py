import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns

import streamlit as st

# Judul navbar
st.sidebar.title('Main Menu')

# Pilihan menu dalam bentuk dropdown
menu_selection = st.sidebar.selectbox('Klik Tombol Di bawah ini', ['Home', 'Pre-Pocesssing Data', 'Klasifikasi SVM','Uji Coba'])

# Membuat konten berdasarkan pilihan menu
if menu_selection == 'Home':
    st.title('Selamat Datang di Website Klasifikasi Hipertensi')
    st.write('Hipertensi dengan Metode SVM')

elif menu_selection == 'Pre-Pocesssing Data':
    st.title('Halaman Pre-pocessing Data')
    st.write('Hipertensi')

elif menu_selection == 'Klasifikasi SVM':
    st.title('Halaman Klasifikasi SVM')
    st.write('This is the contact us page.')

elif menu_selection == 'Uji Coba':
    st.title('Halaman Uji Coba')
    st.write('This is the contact us page.')
