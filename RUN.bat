@echo off

echo Installing dependencies...
pip install .

echo Running the program...
streamlit run src/app.py

pause