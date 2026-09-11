# -*- coding: utf-8 -*-
"""
victory_sound.py
================
「完成 / 過關」歡呼音效——用 Windows 內建 winsound.Beep 連續發音，
零音檔、不裝任何套件（跟 selfie_timer.py 同一招）。

⚠️ 這是「原創」的上揚破關小旋律，不是任天堂瑪利歐那段版權音樂；
   聽起來一樣是遊戲破關的歡呼感，但不踩版權。

用法（在瑜珈教練「完成」那一刻呼叫）：
    from victory_sound import play_victory
    play_victory()              # 預設非阻塞：音效在背景播，程式繼續跑

若想等音效播完再往下（例如播完才關視窗）：
    play_victory(block=True)

非 Windows（沒有 winsound）會自動靜默跳過，不會報錯。
"""

import sys
import time
import threading

try:
    import winsound  # Windows 內建
    _HAS_BEEP = sys.platform.startswith("win")
except Exception:
    _HAS_BEEP = False


# 原創破關小旋律：上揚大三和弦 + 結尾高音華彩，營造「ta-da！過關！」的歡呼感。
# 每個音是 (頻率 Hz, 毫秒)；頻率 0 代表休止（用 sleep）。
# 頻率須在 37~32767 之間（winsound.Beep 限制）。
_VICTORY_NOTES = [
    (784,  100),   # G5
    (988,  100),   # B5
    (1175, 100),   # D6
    (1568, 230),   # G6  ← 衝上去
    (0,     60),   # 短休止
    (1319, 110),   # E6
    (1568, 380),   # G6  ← 收在高音，亮亮的
]

# 備用「小確幸」版本（比較收斂、兩三聲就好，怕太吵時用）
_CHIME_NOTES = [
    (1047, 110),   # C6
    (1319, 110),   # E6
    (1568, 320),   # G6
]


def _play(notes):
    if not _HAS_BEEP:
        return
    for freq, ms in notes:
        try:
            if freq <= 0:
                time.sleep(ms / 1000.0)
            else:
                winsound.Beep(int(freq), int(ms))
        except Exception:
            # 某個音播失敗就跳過，不要讓音效拖垮主程式
            pass


def play_victory(block=False, chime=False):
    """
    播放完成音效。
      block=False（預設）→ 背景播放，不卡住主程式（建議用這個）
      block=True         → 等音效播完才返回
      chime=True         → 改用比較收斂的三聲小版本（怕太吵時）
    """
    notes = _CHIME_NOTES if chime else _VICTORY_NOTES
    if block:
        _play(notes)
    else:
        threading.Thread(target=_play, args=(notes,), daemon=True).start()


# 直接執行這支檔可以先試聽：python victory_sound.py
if __name__ == "__main__":
    print("試聽：完整破關版 …")
    play_victory(block=True)
    time.sleep(0.4)
    print("試聽：收斂小確幸版（chime=True）…")
    play_victory(block=True, chime=True)
    print("done")
