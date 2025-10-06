@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
streamlit run GE_Plot_v57.py
pause
