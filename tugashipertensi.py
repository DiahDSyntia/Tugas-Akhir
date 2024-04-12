import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import keras
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(data): 
    def preprocess_text(text):
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = re.sub(r'[A-Za-z]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def preprocess_numerical(text):
        text = re.sub(r'[^0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
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
    # One-hot encoding for 'Jenis Kelamin'
    one_hot_encoder = OneHotEncoder()
    encoded_gender = one_hot_encoder.fit_transform(data[['Jenis Kelamin']].values.reshape(-1, 1))
    encoded_gender = pd.DataFrame(encoded_gender.toarray(), columns=one_hot_encoder.get_feature_names_out(['Jenis Kelamin']))  
    # Menghapus baris dengan nilai yang hilang (NaN)
    data = data.dropna()
    # Menghapus duplikat data
    data = data.drop_duplicates()
    # Mapping for 'Hipertensi'
    data['Diagnosa'] = data['Diagnosa'].map({'HIPERTENSI 1': 1, 'HIPERTENSI 2': 2, 'TIDAK': 0})
    # Drop the original 'Jenis Kelamin' feature
    data = data.drop('Jenis Kelamin', axis=1)    
    # Concatenate encoded 'Jenis Kelamin' and transformed 'Diagnosa' with original data
    data = pd.concat([data, encoded_gender], axis=1)
    return data
    
def normalize_data(data):
    scaler = MinMaxScaler()
    columns_to_normalize = ['Usia', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi']
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    return data

def classify_SVM(data):
    # Pisahkan fitur dan target
    X = data[['Usia', 'IMT', 'Sistole', 'Diastole', 'Nafas', 'Detak Nadi', 'JK_L', 'JK_P']]
    y = data['Diagnosa']

    # Bagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Inisialisasi model SVM
    model = SVC(kernel='linear', C=1, random_state=0)

    # Latih model pada data latih
    model.fit(X_train, y_train)

    # Menguji model pada data uji
    y_pred = model.predict(X_test)

    # Mengukur akurasi pada data uji
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    # Membuat confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix dalam bentuk heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fig = plt.gcf()

    return y_test, y_pred, accuracy, fig

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "PreProcessing Data", "Klasifikasi SVM", "Uji Coba"],
            icons=['house', 'table', 'boxes', 'check2-circle'],
            menu_icon="cast",
            default_index=1,
            orientation='vertical')
    
        upload_file = st.sidebar.file_uploader("Masukkan file csv disini", key=1)
    
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Selamat Data di Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: left;"> Hipertensi </h1>', unsafe_allow_html=True)
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
        st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.write("Data yang digunakan yaitu data Penyakit Hipertensi dari UPT Puskesmas Modopuro Mojokerto.")
            st.dataframe(df)
    
    elif selected == 'PreProcessing Data':
        st.markdown('<h3 style="text-align: left;"> Data Asli </h1>', unsafe_allow_html=True)
        st.write("Berikut merupakan data asli yang didapat dari UPT Puskesmas Modopuro Mojokerto.")
    
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.dataframe(df)
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
    
    elif selected == 'Klasifikasi SVM':
        st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan SVM")
    
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
                normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
                y_true, y_pred, accuracy = classify_SVM(normalized_data)  # Assuming classify_MLP also returns loss
                
                # Generate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
        
                 # Tampilkan visualisasi confusion matrix menggunakan heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Predict Positive', 'Predict Negative'],
                            yticklabels=['Actual Positive', 'Actual Negative'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig) 
        
                # Clear the current plot to avoid displaying it multiple times
                plt.clf()
        
                # Generate classification report
                with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warning
                    report = classification_report(y_true, y_pred, zero_division=0)
        
                # Extract metrics from the classification report
                lines = report.split('\n')
                accuracy = float(lines[5].split()[1]) * 100
                precision = float(lines[2].split()[1]) * 100
                recall = float(lines[3].split()[1]) * 100
        
                # Display the metrics
                html_code = f"""
                <table style="margin: auto;">
                    <tr>
                        <td style="text-align: center;"><h5>Loss</h5></td>
                        <td style="text-align: center;"><h5>Accuracy</h5></td>
                        <td style="text-align: center;"><h5>Precision</h5></td>
                        <td style="text-align: center;"><h5>Recall</h5></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;">{loss:.4f}</td>
                        <td style="text-align: center;">{accuracy:.2f}%</td>
                        <td style="text-align: center;">{precision:.2f}%</td>
                        <td style="text-align: center;">{recall:.2f}%</td>
                    </tr>
                </table>
                """
                
                st.markdown(html_code, unsafe_allow_html=True)
    
    elif selected == 'Uji Coba':
        st.title("Uji Coba")
        st.write("Masukkan nilai untuk pengujian:")
    
        # Input fields
        age = st.number_input("Umur", min_value=0, max_value=150, step=1, value=30)
        bmi = st.number_input("IMT", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
        systole = st.number_input("Sistole", min_value=0, max_value=300, step=1, value=120)
        diastole = st.number_input("Diastole", min_value=0, max_value=200, step=1, value=80)
        breaths = st.number_input("Nafas", min_value=0, max_value=100, step=1, value=16)
        heart_rate = st.number_input("Detak Nadi", min_value=0, max_value=300, step=1, value=70)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    
        # Convert gender to binary
        gender_binary = 1 if gender == "Perempuan" else 0
        
        # Button for testing
        if st.button("Hasil Uji Coba"):
            # Prepare input data for testing
            input_data = pd.DataFrame({
                "Umur": [age],
                "IMT": [bmi],
                "Sistole": [systole],
                "Diastole": [diastole],
                "Nafas": [breaths],
                "Detak Nadi": [heart_rate],
                "Jenis Kelamin": [gender_binary],
                "Diagnosa": [0]  # Placeholder value
            })
    
if __name__ == "__main__":
    main()
