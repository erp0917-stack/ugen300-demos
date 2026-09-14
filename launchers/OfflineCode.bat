@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0..\OfflineCode" || (echo 找不到 OfflineCode 資料夾,請確認 demo 包沒有被搬動 & pause & exit /b 1)
echo [離線 Code] 載入模型中,約 10 秒,請稍候…
echo 若超過 20 秒沒有畫面:請先關掉其他正在用 UGen300 的 demo,或重新插拔 UGen300。
python offline_code.py 
pause
