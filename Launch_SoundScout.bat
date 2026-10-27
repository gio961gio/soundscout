@echo off
title SoundScout - Semantic Audio Search Engine
cd /d "%~dp0"
echo ========================================================
echo       SOUNDSCOUT - SEMANTIC AUDIO SEARCH ENGINE
echo ========================================================
echo Launching Dark Studio Web Interface in default browser...
python -m streamlit run app.py
pause
