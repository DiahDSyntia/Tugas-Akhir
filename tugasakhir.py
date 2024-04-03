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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    data = pd.read_csv(url)
    st.write(data)

elif menu_selection == 'Pre-Pocesssing Data':
    st.title('Halaman Pre-pocessing Data')
    st.write('Hipertensi')

    st.markdown("### Data Hipertensi")
    st.write('Data Hipertensi ini merupakan data dari Puskesmas Modopuro, Mojokerto')
    url = "https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv"
    data = pd.read_csv(url)
    st.write(data)

    # Tambahkan tombol untuk memicu proses preprocessing
    if st.button('Proses Data'):
        # Menghapus baris dengan nilai yang hilang (NaN)
        data = data.dropna()
        # Menghapus duplikat data
        data = data.drop_duplicates()

        # Mapping for 'Hipertensi'
        data['Diagnosa'] = data['Diagnosa'].map({'HIPERTENSI 1': 1, 'HIPERTENSI 2': '2', 'TIDAK': 0})

       # Melakukan one-hot encoding pada kolom 'Jenis_Kelamin'
       # Melakukan one-hot encoding pada kolom 'Jenis_Kelamin'
        data_encoded = pd.get_dummies(data, columns=['Jenis Kelamin'], prefix='JK')
        # Mengganti nilai yang mewakili keberadaan kategori dengan 1 dan yang tidak dengan 0
        data_encoded.replace({col: {1: '1', 0: '0'} for col in data_encoded.columns}, inplace=True)

        # Melakukan preprocessing pada kolom yang dipilih
        def preprocess_text(text):
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)
            text = re.sub(r'[A-Za-z]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text

        columns_to_clean = ['Usia', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
        for col in columns_to_clean:
            data_encoded[col] = data_encoded[col].apply(preprocess_text)

        # Tampilkan hasil preprocessing di bawah tombol
        st.write('Data setelah preprocessing:')
        st.write(data_encoded)

elif menu_selection == 'Klasifikasi SVM':
    st.title('Halaman Hasil Klasifikasi SVM')
    dataset = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/hasilnormalisasi.csv', sep=';')
    st.write(dataset)
    # Pisahkan fitur dan target
    X = dataset[['Usia', 'IMT', 'Sistole', 'Diastole', 'Nafas','Detak Nadi','JK_L','JK_P']]  # Fitur (input)
    y = dataset['Diagnosa']  # Target (output)

    # Bagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Inisialisasi model SVM
    model = SVC(kernel='linear', C=1)
    # K-Fold Cross Validation
    k_fold = 5
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_fold)
    
    # Menampilkan akurasi K-Fold Cross Validation
    st.write(f'K-Fold Cross Validation Scores: {cv_scores}')
    st.write(f'Mean Accuracy: {cv_scores.mean() * 100:.2f}%')

    # Melatih model pada data latih
    model.fit(X_train, y_train)

    # Menguji model pada data uji
    y_pred = model.predict(X_test)

    # Mengukur akurasi pada data uji
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy on Test Data: {accuracy * 100:.2f}%')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    
    # Tampilkan confusion matrix di halaman Streamlit
    st.write('Confusion Matrix:')
    st.write(conf_matrix)

    # Tampilkan visualisasi confusion matrix menggunakan heatmap
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predict Positive', 'Predict Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig) 

elif menu_selection == 'Uji Coba':
    st.title('Halaman Uji Coba')
    st.write('This is the contact us page.')
