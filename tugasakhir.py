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
from sklearn.preprocessing import OneHotEncoder


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
    def normalize_data(data):
        scaler = MinMaxScaler()
        # Normalisasi data numerik
        numeric_columns = ['Usia', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        return data
    if st.button('Proses Data'):
        # Menghapus baris dengan nilai yang hilang (NaN)
        data = data.dropna()
        # Menghapus duplikat data
        data = data.drop_duplicates()

        # Mapping for 'Hipertensi'
        data['Diagnosa'] = data['Diagnosa'].map({'HIPERTENSI 1': 1, 'HIPERTENSI 2': 2, 'TIDAK': 0})

        # Melakukan one-hot encoding pada kolom 'Jenis Kelamin'
        data = pd.get_dummies(data, columns=['Jenis Kelamin'], prefix='JK')

        def preprocess_text(text):
            # Menghapus karakter non-alphanumeric dan spasi ganda
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)
            # Menghapus karakter alfabet
            text = re.sub(r'[A-Za-z]', '', text)
            # Mengganti spasi ganda dengan spasi tunggal
            text = re.sub(r'\s+', ' ', text)
            # Menghapus spasi di awal dan akhir teks
            text = text.strip()
            return text
        
        columns_to_clean = ['Usia', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
        for col in columns_to_clean:
            data[col] = data[col].apply(preprocess_text)

        # Tampilkan hasil preprocessing di bawah tombol
        st.write('Data setelah preprocessing:')
        st.write(data)

        if st.button('Normalisasi Data'):
            # Normalisasi data
            data = normalize_data(data)

            # Tampilkan hasil normalisasi data
            st.subheader('Data Setelah Normalisasi')
            st.write(data)
            
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
    svm = SVC(kernel='linear', C=1)
    
    # K-Fold Cross Validation
    k_fold = 5
    cv_scores = cross_val_score(svm, X_train, y_train, cv=k_fold)

    # Melatih model pada data latih
    svm.fit(X_train, y_train)
    
    # Mengevaluasi model SVM
    Y_prediction = svm.predict(X_test)
    accuracy_svm = round(accuracy_score(y_test, Y_prediction) * 100, 2)
    acc_svm = round(svm.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test, Y_prediction)
    precision = precision_score(y_test, Y_prediction, average='micro')
    recall = recall_score(y_test, Y_prediction, average='micro')
    f1 = f1_score(y_test, Y_prediction, average='micro')

    # Mengukur akurasi pada data uji
    accuracy = accuracy_score(y_test, Y_prediction)
    st.write(f'Accuracy on Test Data: {accuracy * 100:.2f}%')
    
    # Hitung metrik evaluasi
    precision = precision_score(y_test, Y_prediction, average='micro')
    recall = recall_score(y_test, Y_prediction, average='micro')
    f1 = f1_score(y_test, Y_prediction, average='micro')
    
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, Y_prediction)
    
    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, Y_prediction)
    
    # Tampilkan visualisasi confusion matrix menggunakan heatmap
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predict Positive', 'Predict Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig) 

    # Membuat DataFrame untuk menampilkan metrik evaluasi dalam bentuk tabel
    metrics_data = {'Metric': ['Akurasi','Precision', 'Recall', 'F1 Score'],
                    'Nilai': [accuracy, precision, recall, f1]}
    metrics_df = pd.DataFrame(metrics_data)
    st.write(metrics_df)

elif menu_selection == 'Uji Coba':
    st.title('Halaman Uji Coba')
    col1,col2 = st.columns([2,2])
    with col1:
        usia = st.number_input("Usia",0)
        IMT = st.number_input("IMT",0.00)
        sistole = st.number_input("sistole",0.00)
    with col2:
        diastole = st.number_input("diastole",0.00)
        nafas = st.number_input("nafas",0.00)
        detak_nadi = st.number_input("detak nadi",0.00)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    submit = st.button('Prediksi')

    if submit:
        X_new = np.array([[usia, IMT, sistole, diastole, nafas, detak_nadi, jenis_kelamin]])
        # Prediksi dengan model SVM
        predict = svm.predict(X_new)
    
        # Tulis hasil prediksi
        if predict == 0:
            st.write("""# Anda Tidak Hipertensi""")
        elif predict == 1:
            st.write("""# Anda Hipertensi tingkat 1, Segera Ke Dokter""")
        else:
            st.write("""# Anda Hipertensi tingkat 2, Segera Ke Dokter """)
