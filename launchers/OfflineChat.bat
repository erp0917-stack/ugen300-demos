@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0..\OfflineChat"
python offline_chat.py --model Qwen3-1.7B
pause
