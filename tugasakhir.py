import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

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

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Datasets", "Pre-Processing", "Modelling", "Implementation"],  # required
        icons=["house","folder", "file-bar-graph", "card-list", "calculator"],  # optional
        menu_icon="menu-up",  # optional
        default_index=0,  # optional
        )


if selected == "Home":
    st.title(f'Aplikasi Web Data Mining')
    st.write(""" ### Klasifikasi tingkat kematian gagal jantung menggunakan Metode Decision tree, Random forest, dan SVM
    """)
    st.write('Gagal Jantung adalah kondisi ketika otot jantung tidak dapat memompa darah sebagaimana mestinya untuk memenuhi kebutuhan tubuh. Darah merupakan cairan terpenting yang beredar ke seluruh tubuh dengan menyuplai oksigen ke seluruh bagian tubuh. Penyakit kardiovaskular (CVDs) adalah penyebab kematian nomor 1 secara global, merenggut sekitar 17,9 juta nyawa setiap tahun, yang merupakan 31% dari semua kematian di seluruh dunia. Ancaman masalah kardiovaskular yang terus-menerus ini telah meningkat karena pilihan gaya hidup yang buruk seiring dengan sikap acuh tak acuh terhadap kesehatan. Dengan sebagian besar orang berjuang dengan masalah mental, kebiasaan seperti penggunaan tembakau, pola makan yang tidak sehat dan obesitas, ketidakaktifan fisik dan penggunaan alkohol yang berbahaya telah dilakukan oleh populasi massal. Oleh karena itu, orang yang memiliki risiko kardiovaskular tinggi memerlukan deteksi dan manajemen dini di mana model pembelajaran mesin dapat sangat membantu!')


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

    if upload_file is not None:
        df = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/Tugas-Akhir/main/DATABARU3.xlsx%20-%20DATAFIX.csv"
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


if selected == "Modelling":
    st.title(f"{selected}")
    st.write(""" ### Decision Tree, Random Forest, SVM """)
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
    #### Akurasi:""" )
    results = pd.DataFrame({
        'Model': ['Decision Tree','Random Forest','SVM'],
        'Score': [ acc_decision_tree,acc_random_forest, acc_SVM ],
        'Accuracy_score':[accuracy_dt,accuracy_rf,accuracy_SVM]})
    
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
