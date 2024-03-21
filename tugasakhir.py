import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns

# Judul navbar
st.sidebar.title('Navigation')

# Menambahkan pilihan menu ke navbar
menu_selection = st.sidebar.radio('Go to', ['Home', 'About', 'Contact'])

# Membuat konten berdasarkan pilihan menu
if menu_selection == 'Home':
    st.title('Welcome to the Home Page')
    st.write('This is the homepage content.')

elif menu_selection == 'About':
    st.title('About Us')
    st.write('This is the about us page.')

elif menu_selection == 'Contact':
    st.title('Contact Us')
    st.write('This is the contact us page.')
