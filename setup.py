from setuptools import setup, find_packages

setup(
    name= "app",
    version= "1.0.1",
    packages= find_packages(),
    install_requires= [
        "streamlit",
        "streamlit_option_menu",
        "pandas",
        "numpy",
        "librosa",
        "scikit-learn"
    ]
)
