import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import re
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import joblib

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics 
from sklearn import preprocessing 
from streamlit_option_menu import option_menu

def preprocess_data(data): 
    def preprocess_text(text):
        # Menghilangkan karakter yang tidak diinginkan, seperti huruf dan tanda baca
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        # Menghilangkan semua huruf (A-Z, a-z)
        text = re.sub(r'[A-Za-z]', '', text)
        # Mengganti spasi ganda dengan spasi tunggal
        text = re.sub(r'\s+', ' ', text)
        # Menghapus spasi di awal dan akhir teks
        text = text.strip()
        return text
    # Replace commas with dots and convert numerical columns to floats
    numerical_columns = ['IMT']
    data[numerical_columns] = data[numerical_columns].replace({',': '.'}, regex=True).astype(float)
    columns_to_clean = ['Usia', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
    for col in columns_to_clean:
        data[col] = data[col].apply(preprocess_text)
    return data

def transform_data(data):
    # Mapping for 'Hipertensi'
    data['Diagnosa'] = data['Diagnosa'].map({'HIPERTENSI 1': 1, 'HIPERTENSI 2': 2, 'TIDAK': 0}) 
    # One-hot encoding for 'Jenis Kelamin'
    one_hot_encoder = OneHotEncoder()
    encoded_gender = one_hot_encoder.fit_transform(data[['Jenis Kelamin']].values.reshape(-1, 1))
    encoded_gender = pd.DataFrame(encoded_gender.toarray(), columns=one_hot_encoder.get_feature_names_out(['Jenis Kelamin']))  
    # Drop the original 'Jenis Kelamin' feature
    data = data.drop('Jenis Kelamin', axis=1)   
    # Concatenate encoded 'Jenis Kelamin' and transformed 'Diagnosa' with original data
    data = pd.concat([data, encoded_gender], axis=1)
    return data
    
def normalize_data(data):
    data.drop(columns=['Jenis Kelamin_P'], inplace=True)
    data.rename(columns={'Jenis Kelamin_L': 'Jenis Kelamin'}, inplace=True)
    scaler = MinMaxScaler()
    columns_to_normalize = ['Usia', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi', 'Jenis Kelamin']
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    # Menghapus baris dengan nilai yang hilang (NaN)
    data = data.dropna()
    # Menghapus duplikat data
    data = data.drop_duplicates()
    return data

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Datasets", "Pre-Processing", "Modelling", "Implementation"],  # required
        icons=["house","folder", "file-bar-graph", "card-list", "calculator"],  # optional
        menu_icon="menu-up",  # optional
        default_index=0,  # optional
        )


if selected == "Home":
    st.title(f'Aplikasi Web Klasifikasi Hipertensi')
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
    #st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
    #df = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv")
    #st.write("Data yang digunakan yaitu data Penyakit Hipertensi dari UPT Puskesmas Modopuro Mojokerto.")
    #st.dataframe(df)

if selected == "Datasets":
    st.title(f"{selected}")
    data_hp = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv")
    st.write("Dataset Hipertensi : ", data_hp) 
    st.write('Jumlah baris dan kolom :', data_hp.shape)
    X=data_hp.iloc[:,0:7].values 
    y=data_hp.iloc[:,7].values
    st.write('Dataset Description :')
    #st.write('1. age: Age of the patient')

if selected == "Pre-Processing":
    st.title(f"{selected}")
    st.markdown('<h3 style="text-align: left;"> Data Asli </h1>', unsafe_allow_html=True)
    st.write("Berikut merupakan data asli yang didapat dari UPT Puskesmas Modopuro Mojokerto.")
    
    df = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv")
    st.write("Dataset Hipertensi : ", df) 
    st.markdown('<h3 style="text-align: left;"> Lakukan Cleaning Data </h1>', unsafe_allow_html=True)
    if st.button("Clean Data"):
        cleaned_data = preprocess_data(df)
        st.write("Cleaning Data Selesai.")
        st.dataframe(cleaned_data)
        st.session_state.cleaned_data = cleaned_data

    st.markdown('<h3 style="text-align: left;"> Lakukan Transformasi Data </h3>', unsafe_allow_html=True)
    if 'cleaned_data' in st.session_state:
        if st.button("Transformasi Data"):
            transformed_data = transform_data(st.session_state.cleaned_data.copy())
            st.write("Transformasi Data Selesai.")
            st.dataframe(transformed_data)
            st.session_state.transformed_data = transformed_data  # Store preprocessed data in session state

    st.markdown('<h3 style="text-align: left;"> Lakukan Normalisasi Data </h1>', unsafe_allow_html=True)
    if 'transformed_data' in st.session_state:  # Check if preprocessed_data exists in session state
        if st.button("Normalisasi Data"):
            normalized_data = normalize_data(st.session_state.transformed_data.copy())
            st.write("Normalisasi Data Selesai.")
            st.dataframe(normalized_data)


if selected == "Modelling":
    st.write("Hasil Akurasi, Presisi, Recall, F1- Score Metode SVM")
    data = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/datanormalisasi2.csv', sep=';')

    # Memisahkan fitur dan target
    X = data[['Usia', 'IMT', 'Sistole', 'Diastole', 'Nafas','Detak Nadi','Jenis Kelamin']]
    y = data['Diagnosa']

    # Bagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model SVM sebagai base estimator
    model = SVC(kernel='linear', C=1)

    # K-Fold Cross Validation
    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_fold)
    
    # Menampilkan akurasi K-Fold Cross Validation
    print(f'K-Fold Cross Validation Scores: {cv_scores}')
    print(f'Mean Accuracy: {cv_scores.mean() * 100:.2f}%')
    
    # Menyimpan nilai akurasi dari setiap lipatan
    accuracies = []
    
    # Melakukan validasi silang dan menyimpan akurasi dari setiap iterasi
    for i, (train_index, test_index) in enumerate(k_fold.split(X_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
        # Melatih model
        model.fit(X_train_fold, y_train_fold)
    
        # Menguji model
        y_pred_fold = model.predict(X_val_fold)
    
        # Mengukur akurasi
        accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
        accuracies.append(accuracy_fold)
    
        print(f'Accuracy di fold {i+1}: {accuracy_fold * 100:.2f}%')
    
    # Menampilkan rata-rata akurasi dari setiap lipatan
    print(f'Mean Accuracy of K-Fold Cross Validation: {np.mean(accuracies) * 100:.2f}%')

    # Melatih model pada data latih
    model.fit(X_train, y_train)

    # Menguji model pada data uji
    y_pred = model.predict(X_test)
    
    # Mengukur akurasi pada data uji
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Tampilkan visualisasi confusion matrix menggunakan heatmap
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig) 
    
    # Generate classification report
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warning
        report = classification_report(y_test, y_pred)
    
        # Display the metrics
        html_code = f"""
        <table style="margin: auto;">
            <tr>
                <td style="text-align: center;"><h5>Accuracy</h5></td>
                <td style="text-align: center;"><h5>Precision</h5></td>
                <td style="text-align: center;"><h5>Recall</h5></td>
                <td style="text-align: center;"><h5>F1- Score</h5></td>
            </tr>
            <tr>
                <td style="text-align: center;">{accuracy * 100:.2f}%</td>
                <td style="text-align: center;">{precision * 100:.2f}%</td>
                <td style="text-align: center;">{recall * 100:.2f}%</td>
                <td style="text-align: center;">{f1 * 100:.2f}%</td>
            </tr>
        </table>
        """
            
        st.markdown(html_code, unsafe_allow_html=True)


if selected == "Implementation":
    data = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/datanormalisasi2.csv', sep=';')
        
    #st.write("Dataset Hipertensi : ", data)
    
    # Memisahkan fitur dan target
    X = data[['Usia', 'IMT', 'Sistole', 'Diastole', 'Nafas','Detak Nadi','Jenis Kelamin']]
    y = data['Diagnosa']

    # Bagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Inisialisasi model SVM sebagai base estimator
    model = SVC(kernel='rbf', C=1)

    # K-Fold Cross Validation
    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_fold)
    
    # Menampilkan akurasi K-Fold Cross Validation
    print(f'K-Fold Cross Validation Scores: {cv_scores}')
    print(f'Mean Accuracy: {cv_scores.mean() * 100:.2f}%')
    
    # Menyimpan nilai akurasi dari setiap lipatan
    accuracies = []
    
    # Melakukan validasi silang dan menyimpan akurasi dari setiap iterasi
    for i, (train_index, test_index) in enumerate(k_fold.split(X_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
        # Melatih model
        model.fit(X_train_fold, y_train_fold)
    
        # Menguji model
        y_pred_fold = model.predict(X_val_fold)
    
        # Mengukur akurasi
        accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
        accuracies.append(accuracy_fold)
    
        print(f'Accuracy di fold {i+1}: {accuracy_fold * 100:.2f}%')
    
    # Menampilkan rata-rata akurasi dari setiap lipatan
    print(f'Mean Accuracy of K-Fold Cross Validation: {np.mean(accuracies) * 100:.2f}%')

    # Melatih model pada data latih
    model.fit(X_train, y_train)

    # Menguji model pada data uji
    y_pred = model.predict(X_test)
    
    # Mengukur akurasi pada data uji
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
        
    st.write("""
    ### Input Data :"""
    )
    Jenis_Kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    # Convert gender to binary
    gender_binary = 1 if Jenis_Kelamin == "Laki-laki" else 0
    Usia = st.number_input("Umur", min_value=0, max_value=150)
    IMT = st.number_input("IMT", min_value=0.0, max_value=100.0)
    Sistole = st.number_input("Sistole", min_value=0, max_value=300)
    Diastole = st.number_input("Diastole", min_value=0, max_value=200)
    Nafas = st.number_input("Nafas", min_value=0, max_value=100)
    Detak_nadi = st.number_input("Detak Nadi", min_value=0, max_value=300)
    submit = st.button("Submit")
    
    if submit:
        # Normalisasi data input
        scaler = MinMaxScaler()
        X_new_normalized = scaler.fit_transform(np.array([[Usia, IMT, Sistole, Diastole, Nafas, Detak_nadi, gender_binary]]))
    
        # Lakukan prediksi dengan model
        prediksi = model.predict(X_new_normalized)
    
        if prediksi == 1 :
            st.write("""# Hasil Prediksi : Hipertensi 1, Silahkan Ke Dokter""")
        elif prediksi == 2:
            st.write("# Hipertensi 2, Silahkan ke dokter")
        else:
            st.write("# Tidak Hipertensi")
