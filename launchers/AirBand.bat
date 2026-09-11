@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0..\HandDemos"
python air_band.py --source 0 --fullscreen
pause
