# LIBRARY / MODULE / PUSTAKA

import streamlit as st
import pandas as pd
import numpy as np
import librosa, os

from itertools import product
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

from warnings import simplefilter

simplefilter(action= "ignore", category= FutureWarning)

# DEFAULT FUNCTIONS

"""Buat jarak di Webpage

Fungsi-fungsi untuk membuat jarak pada webpage menggunakan margin space
dengan ukuran yang bervariatif.
"""

def ms_20():
    st.markdown("<div class= \"ms-20\"></div>", unsafe_allow_html= True)

def ms_40():
    st.markdown("<div class= \"ms-40\"></div>", unsafe_allow_html= True)

def ms_60():
    st.markdown("<div class= \"ms-60\"></div>", unsafe_allow_html= True)

def ms_80():
    st.markdown("<div class= \"ms-80\"></div>", unsafe_allow_html= True)

"""Buat layout di Webpage

Fungsi-fungsi untuk layouting Webpage menggunakan fungsi columns() dari
Streamlit. Argumen yang diterima adalah list untuk ukuran setiap layout
yang diinginkan.

Returns
-------
self : object containers
    Mengembalikan layout container.
"""

def ml_center():
    left, center, right = st.columns([.3, 2.5, .3])
    return center

def ml_split():
    left, right = st.columns([1, 1])
    return left, right

def ml_left():
    left, right = st.columns([1.75, 1])
    return left, right

def ml_right():
    left, right = st.columns([1, 1.75])
    return left, right

"""Cetak text di Webpage

Fungsi-fungsi untuk menampilkan teks dengan berbagai gaya dengan
memanfaatkan fungsi dari streamlit seperti title(), write(),
dan caption().
"""

def show_title(text, underline= False):
    st.title(text)
    if underline:
        st.markdown("---")

def show_text(text, size= 3, underline= False):
    heading = "#" if size == 1 else (
        "##" if size == 2 else (
            "###" if size == 3 else (
                "####" if size == 4 else "#####"
            )
        )
    )

    st.write(f"{heading} {text}")
    if underline:
        st.markdown("---")

def show_caption(text, size= 3, underline= False):
    heading = "#" if size == 1 else (
        "##" if size == 2 else (
            "###" if size == 3 else (
                "####" if size == 4 else "#####"
            )
        )
    )

    st.caption(f"{heading} {text}")
    if underline:
        st.markdown("---")

def show_paragraf(text):
    st.markdown(f"<div class= \"paragraph\">{text}</div>",
                unsafe_allow_html= True)

"""Baca file menggunakan Pandas

Fungsi-fungsi untuk membaca file menggunakan Pandas dengan format file
yang diharapkan berupa file.csv, file.xlsx, dan sejenisnya.
"""

def get_csv(filepath):
    return pd.read_csv(filepath)

def get_excel(filepath):
    return pd.read_excel(filepath)

def mk_dir(dirname):
    """Buat folder pada direktori lokal

    Fungsi ini akan memeriksa path folder, jika tidak ada folder yang
    dimaksud dalam path tersebut maka folder akan dibuat.

    Parameters
    ----------
    dirname : string
        Jalur tempat folder akan dibuat.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# CUSTOM FUNCTIONS
        
def get_musik(directory):
    """Baca file musik

    File musik yang telah dibagi per folder (per genre) akan dibaca
    disini. Penjelajahan file dilakukan dengan os yang nantinya akan
    mendapatkan 3 elemen dari data: filepath, filename, dan genre
    musik.

    Parameters
    ----------
    directory : string
        Jalur utama tempat file musik akan diakses.

    Returns
    -------
    df : object DataFrame or TextFileReader
        File csv (comma-separated values) dikembalikan sebagai
        struktur data dua dimensi dengan sumbu yang diberi label.
    """
    temp_filepath, temp_genre, temp_filename = [], [], []
    for _dir in os.listdir(directory): # main directory
        folderpath = os.path.join(directory, _dir)
        if os.path.isdir(folderpath):
            for filename in os.listdir(folderpath): # genre directory
                filepath = os.path.join(folderpath, filename)

                temp_filepath.append(filepath)
                temp_filename.append(filename)
                temp_genre.append(_dir)
        else:
            temp_filepath.append(folderpath)
            temp_filename.append(_dir)
            temp_genre.append(directory)

    df = pd.DataFrame({
        "filepath": temp_filepath,
        "filename": temp_filename,
        "genre": temp_genre
    })
    return df

@st.cache_data(ttl= 3600, show_spinner= "Fetching data...")
def ekstraksi_fitur_mfcc(df, duration= 30, coef= 13):
    """Ekstraksi Fitur MFCC

    Fitur audio MFCC didasarkan pada persepsi pendengaran
    manusia. Ekstraksi MFCC dilakukan dengan menggunakan Librosa
    yang menyediakan pemrosesan audio.

    Parameters
    ----------
    df : object DataFrame
        Object DataFrame tempat semua file musik (path file) tersimpan.

    duration : int or float
        Durasi musik yang di ekstrak.
        
    coef : int
        Jumlah koefisien MFCC yang ingin dihitung.

    Returns
    -------
    res : object DataFrame
        DataFrame dari data musik dengan fitur dan label yang dicatat.
    """
    mfcc_feature = []
    for _dir in df.iloc[:, 0]:
        y, sr = librosa.load(_dir, duration= duration)
        mfcc = librosa.feature.mfcc(y= y, sr= sr, n_mfcc= coef)
        
        feature = np.mean(mfcc, axis= 1)
        mfcc_feature.append(feature)
    
    res = pd.DataFrame({
        "filename": df.iloc[:, 1],
        **{f"mfcc_{i + 1}": [x[i] for x in mfcc_feature] for i in range(coef)},
        "genre": df.iloc[:, -1]
    })
    return res

@st.cache_data(ttl= 3600, show_spinner= "Train model...")
def tuned_model(features, labels, params, K= 5):
    """Train model tuned

    Pelatihan model menggunakan Random Forest dengan hypertuning parameter
    dan validasi KFold.

    Parameters
    ----------
    features : ndarray or shape (n_samples, n_features)
        Sampel OOB (Out-of-Bag).

    labels : ndarray or shape (n_samples, 1, n_outputs)
        Label sampel OOB.

    K : int
        Jumlah subset Fold.

    params : object Array
        Nilai parameter yang digunakan untuk hypertuning
        parameter model Random Forest.

    Returns
    -------
    score : object DataFrame
        Hasil pelatihan yang menyimpan nilai metrics evaluasi.
    
    params : object DataFrame
        Nilai parameter yang digunakan dalam pelatihan model.
    """
    kfold = KFold(n_splits= K, shuffle= True, random_state= 42)
    
    metrics_eval = {
        "akurasi": 0, "presisi": 0, "recall": 0, "f1-score": 0
    }
    param_values = {}
    temp_ = 0

    for pair in product(*params):
        model = RandomForestClassifier(
            criterion= pair[0], max_depth= pair[1],
            n_estimators= pair[2], max_features= pair[3],
            min_samples_split= pair[4]
        )

        for tr_index, ts_index in kfold.split(features):
            X_train, X_test = features[tr_index], features[ts_index]
            y_train, y_test = labels[tr_index], labels[ts_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score_ = accuracy_score(y_test, y_pred)

            if temp_ < score_:
                temp_ = score_

                metrics_eval["akurasi"] = score_
                metrics_eval["presisi"] = precision_score(y_test, y_pred, average= "macro")
                metrics_eval["recall"] = recall_score(y_test, y_pred, average= "macro")
                metrics_eval["f1-score"] = f1_score(y_test, y_pred, average= "macro")

                param_values["criterion"] = pair[0]
                param_values["max_depth"] = pair[1]
                param_values["n_estimators"] = pair[2]
                param_values["max_features"] = pair[3]
                param_values["min_samples_split"] = pair[4]

    score = pd.DataFrame(metrics_eval, index= [0])
    params = pd.DataFrame(param_values, index= [0])
    return score, params

@st.cache_data(ttl= 3600, show_spinner= "Train model...")
def basic_model(
    features, labels, K= 5, criterion= "gini", max_depth= None,
    n_estimators= 100, max_features= "sqrt", min_samples_split= 2
):
    """Train model basic

    Pelatihan model menggunakan Random Forest dengan beberapa
    persiapan seperti setting parameter dan validasi KFold.

    Parameters
    ----------
    features : ndarray or shape (n_samples, n_features)
        Sampel OOB (Out-of-Bag).

    labels : ndarray or shape (n_samples, 1, n_outputs)
        Label sampel OOB.

    K : int
        Jumlah subset Fold.

    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Fungsi untuk mengatur kualitas dari pembagian. Kriteria
        yang didukung adalah "gini" untuk Gini impurity dan
        "log_loss" dan "entropy" keduanya untuk Shannon
        information gain.

    max_depth : int, default=None
        Kedalaman maksimum pohon. Jika None, maka node
        diperluas hingga semua daun murni (pure) atau hingga
        semua daun berisi kurang dari min_samples_split sampel.

    n_estimators : int, default=100
        Jumlah trees dalam forest.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        Jumlah fitur yang perlu dipertimbangkan saat mencari pemisahan terbaik:

        - Jika int, mempertimbangkan fitur `max_features` di setiap pemisahan.
        - Jika float, maka `max_features` adalah pecahan dan fitur
          `max(1, int(max_features * n_features_in_))` dipertimbangkan di
          setiap pemisahan.
        - Jika "sqrt", maka `max_features=sqrt(n_features)`.
        - Jika "log2", maka `max_features=log2(n_features)`.
        - Jika None, maka `max_features=n_features`.

    min_samples_split : int or float, default=2
        Jumlah minimum sampel yang diperlukan untuk memisahkan node internal:

        - Jika int, maka `min_samples_split` dianggap sebagai nilai minimum.
        - Jika float, maka `min_samples_split` adalah pecahan dan
          `ceil(min_samples_split * n_samples)` adalah minimumnya jumlah
          sampel untuk setiap pemisahan.

    Returns
    -------
    score : object DataFrame
        Hasil pelatihan yang menyimpan nilai metrics evaluasi.
    
    params : object DataFrame
        Nilai parameter yang digunakan dalam pelatihan model.
    """
    kfold = KFold(n_splits= K, shuffle= True, random_state= 42)

    metrics_eval = {
        "akurasi": 0, "presisi": 0, "recall": 0, "f1-score": 0
    }
    param_values = {}

    for tr_index, ts_index in kfold.split(features):
        X_train, X_test = features[tr_index], features[ts_index]
        y_train, y_test = labels[tr_index], labels[ts_index]

        model = RandomForestClassifier(
            criterion= criterion, max_depth= max_depth,
            n_estimators= n_estimators, max_features= max_features,
            min_samples_split= min_samples_split, random_state= 42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_eval["akurasi"] += accuracy_score(y_test, y_pred)
        metrics_eval["presisi"] += precision_score(y_test, y_pred, average= "macro")
        metrics_eval["recall"] += recall_score(y_test, y_pred, average= "macro")
        metrics_eval["f1-score"] += f1_score(y_test, y_pred, average= "macro")

    metrics_eval["akurasi"] /= K
    metrics_eval["presisi"] /= K
    metrics_eval["recall"] /= K
    metrics_eval["f1-score"] /= K
    
    param_values["criterion"] = criterion
    param_values["max_depth"] = max_depth
    param_values["n_estimators"] = n_estimators
    param_values["max_features"] = max_features
    param_values["min_samples_split"] = min_samples_split

    score = pd.DataFrame(metrics_eval, index= [0])
    params = pd.DataFrame(param_values, index= [0])
    return score, params