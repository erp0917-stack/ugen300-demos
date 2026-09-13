@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0.."
set "IMG=%ROOT%\scripts\plate_sample.png"
echo ===== UGen300 全部 demo 自檢(每個載模型跑一次,結束後確認裝置仍在)=====
hailortcli scan || (echo [!] 找不到 UGen300,請插上後重試 & pause & exit /b 1)
for %%A in ("WakeUp|wakeup.py" "Calories|calories.py" "OfflineChat|offline_chat.py" "PhotoRestore|photo_restore.py" "HandDemos|gesture_caption.py" "HandDemos|fruit_ninja.py" "HandDemos|air_band.py" "Traffic|traffic_count.py" "HeadCount|head_count.py" "EdgeVsCloud|edge_vs_cloud.py" "FaceCheckIn|face_checkin.py" "Classroom|classroom.py") do (
  for /f "tokens=1,2 delims=|" %%D in (%%A) do (
    echo --- %%D\%%E ---
    pushd "%ROOT%\%%D"
    python %%E --selftest 2>nul | findstr /V "HailoRT"
    popd
    hailortcli scan | findstr /C:"usb/" >nul || (echo [!] 裝置消失於 %%E 之後,請拔插 & pause & exit /b 1)
  )
)
echo --- Traffic\plate_gate.py(用合成車牌圖)---
pushd "%ROOT%\Traffic"
python plate_gate.py --selftest --image "%IMG%" 2>nul | findstr /V "HailoRT"
popd
echo.
echo ===== 全部完成,裝置狀態: =====
hailortcli scan
pause
