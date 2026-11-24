@echo off
title HonestQuant ML Trading Research Platform
echo ================================================================
echo    HONESTQUANT: ML TRADING RESEARCH & FAILURE-MODE PLATFORM
echo ================================================================
echo Starting Streamlit Quant Studio UI...
echo.
python -m streamlit run app.py --server.port 8501 --theme.base "dark"
pause
