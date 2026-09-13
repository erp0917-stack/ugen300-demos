@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0..\FaceCheckIn"
echo [人臉報到] 載入模型與鏡頭中,約 10 秒,請稍候…
echo 若超過 20 秒沒有畫面:請先關掉其他正在用 UGen300 的 demo,或重新插拔 UGen300。
python face_checkin.py --source auto --fullscreen
pause
