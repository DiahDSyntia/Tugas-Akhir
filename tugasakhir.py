import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns

st.write("""
# Prediksi Breast Cancer
### KNN, Random Forest, Decicion Tree, Gaussian Naive Bayes
"""
)

st.sidebar.write("""
            # Penjelasan Untuk Pengisi Form"""
            )
st.sidebar.write("""
            ####  1. Usia: Diisi dengan angka usia calon pasien yang akan di prediksi
            """)
st.sidebar.write("""
            ####  2. BMI: Diisi dengan jumlah BMI yang ada dalam anda. Angka BMI normal berada pada kisaran 18,5-25.
            """)
st.sidebar.write("""
            ####  3. Glukosa: Diisi dengan jumlah glukosa dalam tubuh anda. Glukosa Normal berkisar < 100mg/dL, jika berpuasa 70-130 mg/dL, < 180 mg/dL (setelah makan), 100-140 mg/dL (sebelum tidur)
            """)
st.sidebar.write("""
            ####  4. Insulin: Diisi dengan jumlah insulin dalam tubuh anda. Insulin normal berkisar di bawah 100 mg/dL.
            """)
st.sidebar.write("""
            ####  5. HOMA: Diisi dengan jumlah HOMA dalam tubuh anda. homeostasis model aseessment (HOMA)
            """)
st.sidebar.write("""
            ####  6. Leptin: Diisi dengan jumlah Leptin dalam tubuh anda. Leptin adalah suatu protein yang berasal dari 167 asam amino,merupakan hormon yang di produksi oleh jaringan adiposa. Biasa ditentukan dalam bentuk (ng/mL)
            """)
st.sidebar.write("""
            ####  7. Adiponectin: Diisi dengan jumlah Adiponectin dalam tubuh anda.
            """)           
st.sidebar.write("""
            ####  8. Resistin: Diisi dengan jumlah resistin dalam tubuh anda. Biasa ditentukan dalam bentuk (ng/mL)
            """)
st.sidebar.write("""
            ####  9. MCP: Diisi dengan jumlah MCP dalam tubuh anda. MCP (Monocyte Chemoattracttant Protein-1). Biasa ditentukan dalam bentuk (pg/dL)
            """)
st.sidebar.write("""
            ####  10. Setelah semuanya terisi silahkan klik prediksi untuk mengetahui hasil dari prediksi tersebut
            """)
