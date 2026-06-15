@echo off
chcp 65001 >nul
cd /d "%~dp0..\Y5"
python yoga_coach.py --hef yolov8s_pose.hef --source 0
pause