@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0..\Classroom"
echo [教室儀表板] 載入模型與鏡頭中,約 10 秒,請稍候…
python classroom.py --source auto --fullscreen
pause
