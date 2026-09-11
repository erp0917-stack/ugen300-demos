@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0..\WakeUp"
python wakeup.py --reps 5 --source auto --fullscreen
pause
