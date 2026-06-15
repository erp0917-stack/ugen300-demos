# -*- coding: utf-8 -*-
"""
gesture_control.py
==================
D4 手勢翻頁的「大腦」：拿到 17 個身體關鍵點後，判斷你做了哪個手勢，
並透過防呆機制決定要不要真的翻頁。

這個檔案完全不依賴 UGen300 / Hailo，是純邏輯，所以可以先寫好、先測試。

三個手勢（已和 Matt 確認的最穩配置）：
  - 舉右手揮一下  → 下一頁
  - 舉左手揮一下  → 上一頁
  - 雙手同時舉高  → 從頭重新播放

作者：Claude（軍師）＋ Matt
"""

import time

# pyautogui 用來「模擬按鍵」控制簡報。
# 這裡用 try 包起來，是為了讓你在還沒裝 pyautogui 時，也能先 import 這個檔案、
# 先測試手勢判斷邏輯（trigger_action 真正按鍵的部分才需要 pyautogui）。
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # 滑鼠移到左上角可強制停止，手勢失控時的保險
    _HAS_PYAUTOGUI = True
except Exception:
    _HAS_PYAUTOGUI = False


# ============================================================
# COCO 17 個關鍵點的標準順序（YOLOv8-pose 就是用這個順序）
# 我們只會用到 鼻子、左右手腕、左右肩膀，但全列出來方便你對照。
# ============================================================
KP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# 手勢代號
GESTURE_NONE = "none"
GESTURE_RIGHT = "right_hand"   # 舉右手
GESTURE_LEFT = "left_hand"     # 舉左手
GESTURE_BOTH = "both_hands"    # 雙手舉高

# 動作代號
ACTION_NEXT = "next"     # 下一頁
ACTION_PREV = "prev"     # 上一頁
ACTION_RESTART = "restart"  # 從頭播放


# ============================================================
# 第一部分：純判斷 —— 給我 17 個點，我告訴你「現在是哪個手勢」
# ============================================================
def detect_raw_gesture(keypoints, conf_threshold=0.5):
    """
    輸入：
      keypoints = 一個 list，長度 17，每個元素是 (x, y, confidence)
                  x, y 是該關鍵點在畫面上的座標（像素）
                  confidence 是這個點的可信度（0~1）
      conf_threshold = 低於這個可信度的點，當作「沒看到」，不採用

    回傳：
      GESTURE_NONE / GESTURE_RIGHT / GESTURE_LEFT / GESTURE_BOTH 其中一個

    判斷原理（重要）：
      影像座標系的 y 軸是「往下變大」，所以「手舉高」= 手腕的 y 比頭還「小」。
      我們用「鼻子」當頭的高度基準。
      手腕 y < 鼻子 y - margin  ===  這隻手舉到比頭還高
    """
    def get(name):
        """取某個關鍵點，若可信度不足回傳 None"""
        x, y, c = keypoints[KP[name]]
        if c < conf_threshold:
            return None
        return (x, y)

    nose = get("nose")
    lw = get("left_wrist")
    rw = get("right_wrist")

    # 連頭都沒偵測到，無法判斷，當作沒手勢
    if nose is None:
        return GESTURE_NONE

    # margin：手腕要高過頭「多少像素」才算數，避免手剛好在頭附近抖動就觸發。
    # 數字越大越「嚴格」（要舉越高）。demo 時可微調，建議 30~60。
    margin = 40
    nose_y = nose[1]

    left_up = (lw is not None) and (lw[1] < nose_y - margin)
    right_up = (rw is not None) and (rw[1] < nose_y - margin)

    # 先判斷「雙手」，再判斷單手（避免雙手舉時誤判成單手）
    if left_up and right_up:
        return GESTURE_BOTH
    if right_up:
        return GESTURE_RIGHT
    if left_up:
        return GESTURE_LEFT
    return GESTURE_NONE


# ============================================================
# 第二部分：防呆狀態機 —— 避免「亂翻頁」的關鍵
# ============================================================
class GestureController:
    """
    這個 class 是「防止亂翻」的守門員。它解決三個現場一定會遇到的問題：

    1. 雜訊閃動：AI 偶爾會誤判一兩格 → 我們要求「連續看到 N 格同一手勢」才算數
    2. 一直舉著連續翻：你舉手沒放下，畫面會一直翻 → 我們要求「手放下再舉」才算新的一次
    3. 抖動連觸：剛翻完手還沒放穩又觸發 → 翻完後設一段「冷卻時間」

    用法：
      ctrl = GestureController()
      每一格畫面：
          action = ctrl.update(raw_gesture)
          if action: trigger_action(action)   # 有動作才翻頁
    """

    def __init__(self, confirm_frames=4, cooldown_sec=1.2):
        # confirm_frames：要連續幾格同一手勢才確認（越大越穩但反應越慢，建議 3~6）
        self.confirm_frames = confirm_frames
        # cooldown_sec：翻頁後幾秒內不再觸發（避免連翻，建議 1.0~1.5）
        self.cooldown_sec = cooldown_sec

        self._last_raw = GESTURE_NONE
        self._stable_count = 0
        self._armed = True          # True = 可以觸發；翻完且手放下前為 False
        self._last_trigger_time = 0.0

    def update(self, raw_gesture):
        """
        輸入這一格的 raw 手勢，回傳「要執行的動作」或 None。
        """
        now = time.time()

        # 累計「連續同一手勢」的格數
        if raw_gesture == self._last_raw:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_raw = raw_gesture

        # 手放下（回到 NONE）→ 重新「上膛」，允許下一次觸發
        if raw_gesture == GESTURE_NONE:
            self._armed = True
            return None

        # 冷卻中，不觸發
        if now - self._last_trigger_time < self.cooldown_sec:
            return None

        # 必須「上膛」狀態（代表上次觸發後手有放下過）
        if not self._armed:
            return None

        # 必須連續確認足夠格數
        if self._stable_count < self.confirm_frames:
            return None

        # ---- 通過所有防呆，正式觸發 ----
        action = None
        if raw_gesture == GESTURE_RIGHT:
            action = ACTION_NEXT
        elif raw_gesture == GESTURE_LEFT:
            action = ACTION_PREV
        elif raw_gesture == GESTURE_BOTH:
            action = ACTION_RESTART

        if action:
            self._armed = False                 # 觸發後鎖住，要等手放下才解鎖
            self._last_trigger_time = now
        return action


# ============================================================
# 第三部分：真正去「按鍵」控制簡報
# ============================================================
def trigger_action(action):
    """
    把動作變成實際的鍵盤按鍵（PowerPoint / Google Slides 播放模式通用）。
      下一頁 = 方向鍵右
      上一頁 = 方向鍵左
      從頭播放 = Home 鍵
    如果你發現左右相反（舉右手卻往回翻），把下面 NEXT 和 PREV 的按鍵對調即可。
    """
    import sys
    if not _HAS_PYAUTOGUI:
        names = {ACTION_NEXT: "-> 下一頁", ACTION_PREV: "<- 上一頁", ACTION_RESTART: "[R] 從頭播放"}
        print(f"[模擬翻頁，未實際按鍵] {names.get(action, action)}", flush=True)
        return

    # 真的送鍵：同時送方向鍵 + PageDown/PageUp/Home，提高 PPT / Google Slides /
    # PDF / 瀏覽器簡報的相容性
    if action == ACTION_NEXT:
        print("[翻頁] 即將送出按鍵：right + pagedown", flush=True)
        pyautogui.press("right")
        pyautogui.press("pagedown")
        print("[翻頁] -> 下一頁（已送出 right + pagedown）", flush=True)
    elif action == ACTION_PREV:
        print("[翻頁] 即將送出按鍵：left + pageup", flush=True)
        pyautogui.press("left")
        pyautogui.press("pageup")
        print("[翻頁] <- 上一頁（已送出 left + pageup）", flush=True)
    elif action == ACTION_RESTART:
        print("[翻頁] 即將送出按鍵：home", flush=True)
        pyautogui.press("home")
        print("[翻頁] [R] 從頭播放（已送出 home）", flush=True)
    sys.stdout.flush()
