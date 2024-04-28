# LIBRARY / MODULE / PUSTAKA

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import session_state as ss

from functions import *
from warnings import simplefilter

simplefilter(action= "ignore", category= FutureWarning)

# PAGE CONFIG

st.set_page_config(
    page_title= "App",
    layout= "wide",
    page_icon= "globe",
    initial_sidebar_state= "expanded"
)

## hide menu, header, and footer
st.markdown(
    """<style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .st-emotion-cache-z5fcl4 {padding-top: 1rem;}
    </style>""",
    unsafe_allow_html= True
)

## CSS on style.css
with open("./css/style.css") as file:
    st.markdown(
        "<style>{}</style>".format(file.read()),
        unsafe_allow_html= True
    )

class MyApp():
    """Class dari MyApp dan bersifat dinamis

    Parameters
    ----------
    message : bool, default= False
        Jika False, maka pesan error tidak akan ditampilkan
        dalam Webpage Sistem. Jika True, maka akan menampilkan
        pesan error dalam Sistem Webpage yang dapat dilihat
        dan dianalisis.

    Attributes
    ----------
    message : bool
        Tampilkan pesan error pada Sistem Webpage atau tidak.

    pathdata : str
        Path (jalur) data disimpan dalam lokal direktori.

    menus : list
        Daftar menu yang akan ditampilkan dalam Webpage.

    icons : list
        Daftar icon menu untuk setiap menu yang diampilkan dalam
        WebPage.
    """

    def __init__(self, message= False):
        self.message = message
        self.pathdata = "./data/music"
        self.menus = [
            "Beranda", "Dataset", "Ekstraksi Fitur", "Klasifikasi"
        ]
        self.icons = [
            "house", "music-note-beamed", "soundwave", "bar-chart"
        ]

    def _navigation(self):
        """Navigasi sistem

        Returns
        -------
        selected : string
            Selected menu.
        """
        with st.sidebar:
            selected = option_menu(
                "",
                self.menus,
                icons= self.icons,
                styles= {
                    "container": {
                        "padding": "0 !important",
                        "background-color": "#E6E6EA"
                    },
                    "icon": {"color": "#020122", "font-size": "18px"},
                    "nav-link": {
                        "font-size": "16px", "text-align": "left",
                        "margin": "0px", "color": "#020122"
                    },
                    "nav-link-selected": {"background-color": "#F4F4F8"}
                }
            )
            
            ms_60()
            show_caption("Copyright © 2024 | ~", size= 5)
        return selected
    
    def _exceptionMessage(self, e):
        """Tampilan pesan error

        Parameters
        ----------
        e : exception object
            Objek exception yang tersimpan dalam variabel e.
        """
        ms_20()
        with ml_center():
            st.error("Terjadi masalah")
            if self.message:
                st.exception(e) # tampilkan keterangan error

    def _pageBeranda(self):
        """Halaman beranda

        Halaman ini akan menampilkan judul penelitian dan abstrak dari
        proyek.
        """
        try:
            ms_20()
            show_text(
                "Penerapan Metode Random Forest Dalam Klasifikasi "
                "Genre Musik Menggunakan Ekstraksi Fitur MFCC",
                size= 2,
                underline= True
            )

            ms_40()
            with ml_center():
                show_paragraf(
                    "Musik dapat direpresentasikan sebagai sinyal "
                    "audio yang memiliki fitur di antaranya bandwith, "
                    "frekuensi, spectral roll-off, dan masih banyak "
                    "karakteristik fitur lainnya. Untuk mendapatkan "
                    "informasi fitur yang terkandung dalam musik "
                    "dilakukan proses ekstraksi fitur yaitu untuk "
                    "mendapatkan atribut yang relevan dari data audio "
                    "musik. Data fitur tersebut dapat digunakan untuk "
                    "melakukan tindakan seperti klasifikasi, "
                    "pengenalan, dan analisis musik. Identifikasi "
                    "genre musik dapat membantu pengguna dalam "
                    "menemukan lagu-lagu yang sesuai dengan preferensi "
                    "mereka dan juga berguna dalam klasifikasi dan "
                    "pengindeksan koleksi musik besar. Dalam "
                    "penelitian ini, fokus utama adalah penerapan "
                    "metode Random Forest untuk mengklasifikasikan "
                    "genre musik menggunakan ekstraksi fitur MFCC pada "
                    "data musik berbahasa Indonesia sejumlah 200 data "
                    "dengan 4 genre musik yang berbeda. Model "
                    "penelitian yang akan digunakan adalah ekstraksi "
                    "fitur MFCC dengan 13 koefisien sebagai "
                    "representasi fitur untuk setiap lagu dan metode "
                    "Random Forest sebagai pengklasifikasi."
                )
        
        except Exception as e:
            self._exceptionMessage(e)

    def _pageDataMusik(self):
        """Halaman Dataset

        Bagian ini akan menampilkan DataFrame yang berisi detail data.
        """
        try:
            ms_20()
            show_text("Data Musik", underline= True)
            
            ms_40()
            with ml_center():
                df = get_musik(self.pathdata)
                st.dataframe(df, use_container_width= True, hide_index= True)

                mk_dir("./data/dataframe")
                df.to_csv("./data/dataframe/list-musik.csv", index= False)
        
        except Exception as e:
            self._exceptionMessage(e)

    def _pageEkstraksiFitur(self):
        """Ekstraksi Fitur MFCC

        Halaman ini akan mengekstrak fitur MFCC dari data dengan membaca
        filepath dari DataFrame list-musik. Number input disediakan untuk
        optimasi pada durasi musik dan koefisien MFCC.
        """
        try:
            ms_20()
            show_text("Ekstraksi Fitur")
            show_caption("Mel Frequency Cepstral Coefficients", underline= True)

            left, right = ml_right()
            with left:
                ms_20()
                duration = st.number_input(
                    "Durasi Musik (detik)", min_value= 1, value= 30, step= 1,
                    key= "Number input untuk nilai durasi musik"
                )
                coef = st.number_input(
                    "Koefisien MFCC", min_value= 1, value= 13, step= 1,
                    key= "Number input untuk nilai koefisien"
                )
                
                ms_40()
                btn_extract = st.button(
                    "Submit", key= "Button fit ekstraksi fitur",
                    use_container_width= True
                )
            with right:
                if btn_extract or ss.fit_extract:
                    ss.fit_extract = True

                    show_caption("Fitur MFCC", size= 2)

                    with st.spinner("Extraction features is running..."):
                        df = ekstraksi_fitur_mfcc(
                            get_csv("./data/dataframe/list-musik.csv"),
                            duration= duration, coef= coef
                        )

                    df.to_csv("./data/dataframe/mfcc_features.csv", index= False)
                    st.dataframe(df, use_container_width= True, hide_index= True)
        
        except Exception as e:
            self._exceptionMessage(e)

    def _pageKlasifikasi(self):
        """Klasifikasi Random Forest

        Halaman untuk setting dan training model. Setting ini terdiri dari
        penyetelan parameter baik secara otomatis maupun manual. Hasil
        pelatihan akan ditampilkan dalam nilai metrics evaluasi.
        """
        try:
            ms_20()
            show_text("Klasifikasi", underline= True)

            list_criterion = ["gini", "entropy", "log_loss"]
            list_max_depth = [32, 64, 112, 128]
            list_n_estimators = [50, 100, 125, 150]
            list_max_features = ["sqrt", "log2"]
            list_min_samples_split = [2, 4 ,6]

            left, right = ml_right()
            with left:
                K = st.selectbox(
                    "Jumlah subset Fold", [4, 5, 10], index= 1,
                    key= "Selectbox untuk jumlah subset Fold"
                )

                set_params = st.radio(
                    "Setting Parameters?", ["Set", "Tune", "Default"],
                    horizontal= True, key= "Radio button untuk setting parameters"
                )

                if set_params == "Set":
                    st.markdown("---")
                    criterion = st.selectbox(
                        "criterion", list_criterion,
                        key= "Selectbox parameter criterion"
                    )

                    max_depth = st.selectbox(
                        "max_depth", list_max_depth,
                        key= "Selectbox parameter max_depth"
                    )

                    n_estimators = st.selectbox(
                        "n_estimators", list_n_estimators,
                        key= "Selectbox parameter n_estimators"
                    )

                    max_features = st.selectbox(
                        "max_features", list_max_features,
                        key= "Selectbox parameter max_features"
                    )

                    min_samples_split = st.selectbox(
                        "min_samples_split", list_min_samples_split,
                        key= "Selectbox parameter min_samples_split"
                    )
                
                elif set_params == "Tune":
                    st.markdown("---")
                    criterion = st.multiselect(
                        "criterion", list_criterion,
                        placeholder= "Pilih opsi",
                        key= "Multiselect parameter criterion"
                    )

                    max_depth = st.multiselect(
                        "max_depth", list_max_depth,
                        placeholder= "Pilih opsi",
                        key= "Multiselect parameter max_depth"
                    )

                    n_estimators = st.multiselect(
                        "n_estimators", list_n_estimators,
                        placeholder= "Pilih opsi",
                        key= "Multiselect parameter n_estimators"
                    )

                    max_features = st.multiselect(
                        "max_features", list_max_features,
                        placeholder= "Pilih opsi",
                        key= "Multiselect parameter max_features"
                    )

                    min_samples_split = st.multiselect(
                        "min_samples_split", list_min_samples_split,
                        placeholder= "Pilih opsi",
                        key= "Multiselect parameter min_samples_split"
                    )

                elif set_params == "Default":
                    pass

                ms_40()
                btn_train = st.button(
                    "Submit", use_container_width= True,
                    key= "Button untuk training model"
                )
            with right:
                df = get_csv("./data/dataframe/mfcc_features.csv")
                features = df.iloc[:, 1:14].values
                labels = df.iloc[:, -1].values

                ms_20()
                st.code(
                    f"Jumlah Data Train = {int(len(labels) / K * (K - 1))} data\n"
                    f"Jumlah Data Test = {int(len(labels) / K * 1)} data"
                )

                if btn_train:
                    with st.spinner("Pelatihan model sedang berlangsung..."):
                        if set_params == "Set":
                            score, params = basic_model(
                                features, labels, K= K, criterion= criterion,
                                max_depth= max_depth, n_estimators= n_estimators,
                                max_features= max_features,
                                min_samples_split= min_samples_split
                            )

                            ms_20()
                            show_caption("Jenis parameter yang digunakan: `Set`")
                            text = ""
                            for cols in params.columns:
                                text += f"{cols} : {params[cols][0]}\n"
                            st.code(text)

                            show_caption("Evaluasi Score")
                            for cols in score.columns:
                                st.info(f"{cols}: {score[cols][0] * 100:.2f}%")
                        elif set_params == "Tune":
                            if len(criterion) != 0 and len(max_depth) != 0 and len(n_estimators) != 0 \
                            and len(max_features) != 0 and len(min_samples_split) != 0:
                                metrics_params = {
                                    "criterion": criterion, "max_depth": max_depth,
                                    "n_estimators": n_estimators, "max_features": max_features,
                                    "min_samples_split": min_samples_split
                                }
                                params = [metrics_params[key] for key in metrics_params]
                                score, params = tuned_model(features, labels, params, K= K)

                                ms_20()
                                show_caption("Jenis parameter yang digunakan: `Tune`")
                                text = ""
                                for cols in params.columns:
                                    text += f"{cols} : {params[cols][0]}\n"
                                st.code(text)

                                show_caption("Evaluasi Score")
                                for cols in score.columns:
                                    st.info(f"{cols}: {score[cols][0] * 100:.2f}%")
                            else:
                                ms_40()
                                st.warning("Setiap nilai parameter harus terisi minimal 1!")
                        elif set_params == "Default":
                            score, params = basic_model(features, labels, K= K)
                            
                            ms_20()
                            show_caption("Jenis parameter yang digunakan: `Default`")
                            text = ""
                            for cols in params.columns:
                                text += f"{cols} : {params[cols][0]}\n"
                            st.code(text)

                            show_caption("Evaluasi Score")
                            for cols in score.columns:
                                st.info(f"{cols}: {score[cols][0] * 100:.2f}%")
        
        except Exception as e:
            self._exceptionMessage(e)

    def main(self):
        """Main Program

        Setting session page diatur disini dan konfigurasi setiap halaman
        dipanggil disini.
        """
        with st.container():
            selected = self._navigation() # sidebar navigation

            if "fit_extract" not in ss:
                ss.fit_extract = False
            
            if selected == self.menus[0]:
                self._pageBeranda()
            elif selected == self.menus[1]:
                self._pageDataMusik()
            elif selected == self.menus[2]:
                self._pageEkstraksiFitur()
            elif selected == self.menus[3]:
                self._pageKlasifikasi()

if __name__ == "__main__":
    app = MyApp(message= True)
    app.main()