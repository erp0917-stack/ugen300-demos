@echo off
chcp 65001 >nul
cd /d "%~dp0..\Y1"
python main.py --hef yolov8m.hef --source 0
pause
