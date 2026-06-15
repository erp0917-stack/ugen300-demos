@echo off
chcp 65001 > nul
cd /d "%~dp0"
python face_report.py --camera --lang tw
pause
