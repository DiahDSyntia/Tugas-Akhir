import streamlit as st
from streamlit_option_menu import option_menu
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
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu", #required
        options=["Homepage", "Pre-Processing Data", "Klasifikasi SVM", "Uji Coba"], #required
    )

if selected == "Homepage":
    st.title(f"Homepage {selected}")
if selected == "Pre-Processing Data":
    st.title(f""Pre-Processing Data" {selected}")
if selected == "Klasifikasi SVM":
    st.title(f""Klasifikasi SVM" {selected}")
if selected == "Uji Coba":
    st.title(f""Uji Coba" {selected}")
