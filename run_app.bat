@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run ui\app.py

