@echo off
chcp 65001 >nul
cd /d "%~dp0..\Y4"
python hand_raise_counter.py --hef yolov8s_pose.hef --source 0
pause
