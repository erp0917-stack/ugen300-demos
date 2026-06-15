# -*- coding: utf-8 -*-
"""
selfie_timer.py
===============
可重用的「倒數自拍」模組。

主程式只要呼叫一次：
    from selfie_timer import countdown_selfie
    frame, cancelled = countdown_selfie(cap, window_title="穿搭大師")
就會：
  1. 拿到攝影機 cap，每一格畫面正中央疊上大字倒數數字（10 → 1）
  2. 最後 3 秒每秒一聲 winsound.Beep（Windows 內建嗶聲，零素材）
  3. 倒數歸零時，當下那一格就是拍照結果
  4. 拍完畫面閃白 0.5 秒給「已拍」回饋，然後 return
  5. 期間按 q 取消，回傳 (None, True)

對 V1/V3 也適用，介面相同。
"""

import sys
import time
import threading

import cv2
import numpy as np

try:
    import winsound  # Windows 內建
    _HAS_BEEP = sys.platform.startswith("win")
except Exception:
    _HAS_BEEP = False


def _beep_async(freq_hz=1200, ms=180):
    """非同步嗶一聲（不阻塞主迴圈）。"""
    if not _HAS_BEEP:
        return
    def _b():
        try:
            winsound.Beep(int(freq_hz), int(ms))
        except Exception:
            pass
    threading.Thread(target=_b, daemon=True).start()


def _draw_big_number(frame, text, color=(0, 240, 255)):
    """在畫面正中央畫一個大數字（白邊 + 主色），3 公尺外也看得到。"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(6.0, min(h, w) / 90.0)  # 隨畫面解析度自動縮放
    thickness = max(8, int(scale * 1.5))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2

    # 半透明深色背板，提高數字可辨識度
    overlay = frame.copy()
    pad = int(th * 0.6)
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # 白色描邊 + 主色填字
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness + 8, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_hint(frame, text):
    """畫面下方一條中性提示文字（避免擋到人）。"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    x = (w - tw) // 2
    y = h - 24
    cv2.rectangle(frame, (x - 12, y - th - 10), (x + tw + 12, y + 10),
                  (26, 26, 46), -1)
    cv2.putText(frame, text, (x, y), font, 0.7, (0, 212, 216), 2, cv2.LINE_AA)


def _flash_white(window_title, frame, ms=500):
    """拍照後白閃 + 定格回饋，給操作者明確「已拍」訊號。"""
    h, w = frame.shape[:2]
    white = np.full_like(frame, 255)
    # 先一格白
    cv2.imshow(window_title, white)
    cv2.waitKey(80)
    # 再回到 frame 本身定格剩餘時間
    cv2.imshow(window_title, frame)
    cv2.waitKey(max(1, ms - 80))


def countdown_selfie(cap, window_title, seconds=10, beep_last_n=3,
                     flash_ms=500, flip_horizontal=True):
    """
    倒數自拍。

    參數：
      cap            cv2.VideoCapture 物件（外面開好的）
      window_title   要顯示在哪個 OpenCV 視窗（要跟主程式同一個視窗名）
      seconds        倒數秒數（預設 10）
      beep_last_n    最後幾秒每秒嗶一聲（預設 3）；歸零時會嗶較長一聲
      flash_ms       拍完白閃定格的毫秒數
      flip_horizontal  是否把畫面水平鏡像（跟主程式一致就好）

    回傳：
      (frame, cancelled)
        frame      拍到的 BGR 影像；取消時為 None
        cancelled  True 代表使用者按 q 取消
    """
    start = time.time()
    last_int = seconds + 1  # 用來偵測「秒數變了」才嗶聲
    last_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            return None, True
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        last_frame = frame.copy()

        elapsed = time.time() - start
        remaining = seconds - elapsed
        if remaining <= 0:
            break

        # 顯示數字：顯示「還剩幾秒」的上限整數（10→1）
        n = int(np.ceil(remaining))
        # 進入新的整數秒 → 在「最後 N 秒內」嗶聲
        if n != last_int and n <= beep_last_n and n >= 1:
            _beep_async(1200, 160)
        last_int = n

        _draw_big_number(frame, str(n))
        _draw_hint(frame, "Get into frame! Press 'q' to cancel")
        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None, True

    # 歸零：再讀一張當成「快門」那格（更新感更強）
    ok, shutter = cap.read()
    if ok:
        if flip_horizontal:
            shutter = cv2.flip(shutter, 1)
        last_frame = shutter

    # 歸零嗶較長一聲（拍照感）
    _beep_async(1800, 300)

    # 白閃 + 定格回饋
    _flash_white(window_title, last_frame, ms=flash_ms)

    return last_frame, False
