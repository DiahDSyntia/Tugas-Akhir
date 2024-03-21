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

st.sidebar.write("""
            # Pengertian Analisis Sentimen"""
            )
st.sidebar.write("""
            Analisis sentimen adalah proses menganalisis teks digital untuk menentukan apakah nada emosional pesan tersebut positif, negatif, atau netral.
            Alat analisis sentimen dapat memindai teks ini untuk secara otomatis menentukan sikap penulis terhadap suatu topik.
            """
            )
st.sidebar.write("""
            # Mengapa harus Analisis Sentimen?"""
            )
st.sidebar.write("""
            1. Memberikan wawasan yang objektif
            """
            )
st.sidebar.write("""
            2. Membangun produk dan layanan yang lebih baik
            """
            )
st.sidebar.write("""
            3. Menganalisis dalam skala besar
            """
            )
st.sidebar.write("""
            4. Hasil waktu nyata
            """
            )
