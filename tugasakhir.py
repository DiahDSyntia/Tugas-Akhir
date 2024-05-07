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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier 
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
    st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
    df = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv")
    st.write("Data yang digunakan yaitu data Penyakit Hipertensi dari UPT Puskesmas Modopuro Mojokerto.")
    st.dataframe(df)

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
    data_hf = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv")
    X=data_hf.iloc[:,0:7].values 
    y=data_hf.iloc[:,7].values
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    #Train and Test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    
    #SVM
    SVM = svm.SVC(kernel='linear', C=1) 
    SVM.fit(X_train, y_train)
    Y_prediction = SVM.predict(X_test)
    accuracy_SVM=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_SVM = round(SVM.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='weighted')
    recall =  recall_score(y_test, Y_pred,average='weighted')
    f1 = f1_score(y_test,Y_pred,average='weighted')
    print('Confusion matrix for SVM\n',cm)
    print('accuracy_SVM : %.3f' %accuracy)
    print('precision_SVM : %.3f' %precision)
    print('recall_SVM : %.3f' %recall)
    print('f1-score_SVM : %.3f' %f1)
    st.write("""
    #### Akurasi:""" )
    
    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(9)
    st.write(result_df)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['Decision Tree', 'Random Forest','SVM'],[accuracy_dt, accuracy_rf, accuracy_SVM])
    plt.show()
    st.pyplot(fig)



if selected == "Implementation":
    st.title(f"{selected}")
    st.write("""
            ### Pilih Metode yang anda inginkan :"""
            )
    algoritma=st.selectbox('Pilih', ('Decision Tree','Random Forest','SVM'))

    data_hf = pd.read_csv("https://raw.githubusercontent.com/AmandaCaecilia/datamining/main/heart_failure_clinical_records_dataset.csv")
    X=data_hf.iloc[:,0:12].values 
    y=data_hf.iloc[:,12].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    #Train and Test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, y_train)  
    Y_pred = decision_tree.predict(X_test) 
    accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)
    
    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    Y_prediction = random_forest.predict(X_test)
    accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test,Y_prediction)
    precision =precision_score(y_test, Y_prediction,average='micro')
    recall =  recall_score(y_test, Y_prediction,average='micro')
    f1 = f1_score(y_test,Y_prediction,average='micro')
    print('Confusion matrix for Random Forest\n',cm)
    print('accuracy_random_Forest : %.3f' %accuracy)
    print('precision_random_Forest : %.3f' %precision)
    print('recall_random_Forest : %.3f' %recall)
    print('f1-score_random_Forest : %.3f' %f1)
    
    #SVM
    SVM = svm.SVC(kernel='linear') 
    SVM.fit(X_train, y_train)
    Y_prediction = SVM.predict(X_test)
    accuracy_SVM=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_SVM = round(SVM.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for SVM\n',cm)
    print('accuracy_SVM : %.3f' %accuracy)
    print('precision_SVM : %.3f' %precision)
    print('recall_SVM : %.3f' %recall)
    print('f1-score_SVM : %.3f' %f1)
        
    st.write("""
            ### Input Data :"""
            )
    age = st.number_input("umur =", min_value=40 ,max_value=90)
    anemia = st.number_input("anemia =", min_value=0, max_value=1)
    creatinine_phosphokinase = st.number_input("creatinine_phosphokinase =", min_value=0 , max_value=10000)
    diabetes = st.number_input("diabetes =", min_value=0, max_value=1)
    ejection_fraction = st.number_input("ejection_fraction =", min_value=0, max_value=100)
    high_blood_pressure = st.number_input("high_blood_pressure =", min_value=0 ,max_value=1)
    platelets = st.number_input("platelets =", min_value=100000, max_value=1000000)
    serum_creatinine = st.number_input("serum_creatinine =", min_value=0.0, max_value=10.0)
    serum_sodium = st.number_input("serum_sodium =", min_value=100, max_value=150)
    sex = st.number_input("sex =", min_value=0, max_value=1)
    smoking = st.number_input("smoking =", min_value=0, max_value=1)
    time = st.number_input("time =", min_value=1, max_value=500)
    submit = st.button("Submit")
    if submit :
        if algoritma == 'Decision Tree' :
            X_new = np.array([[age,anemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
            prediksi = decision_tree.predict(X_new)
            if prediksi == 1 :
                st.write(""" ## Hasil Prediksi : resiko meninggal tinggi""")
            else : 
                st.write("""## Hasil Prediksi : resiko meninggal rendah""")
        elif algoritma == 'Random Forest' :
            X_new = np.array([[age,anemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
            prediksi = random_forest.predict(X_new)
            if prediksi == 1 :
                st.write("""## Hasil Prediksi : resiko meninggal tinggi""")
            else : 
                st.write("""## Hasil Prediksi : resiko meninggal rendah""")
        else :
            X_new = np.array([[age,anemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
            prediksi = SVM.predict(X_new)
            if prediksi == 1 :
                st.write("""## Hasil Prediksi : resiko meninggal tinggi""")
            else : 
                st.write("""## Hasil Prediksi : resiko meninggal rendah""")
