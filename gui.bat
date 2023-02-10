@echo off

set VENV_DIR=.\venv
set PYTHON=python

call %VENV_DIR%\Scripts\activate.bat

%PYTHON% kohya_gui.py

pause