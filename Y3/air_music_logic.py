# -*- coding: utf-8 -*-
"""
air_music_logic.py  ——  空氣樂器的「大腦」（純邏輯，不依賴硬體）
================================================================
拿到手腕的位置後，判斷手揮到畫面的哪個「音區」，該不該觸發音效。
這個檔案不依賴 UGen300，可獨立測試。

設計：把畫面橫向切成 N 個直條（音區），每個音區一個音。
手腕揮進某個音區、且是「新揮入」（不是停在那），就觸發那個音。
"""

import time

# COCO 關鍵點索引（與 hailo_pose 一致）
LEFT_WRIST = 9
RIGHT_WRIST = 10


class AirInstrument:
    """
    用法：
        inst = AirInstrument(num_zones=5, width=640)
        每一格畫面：
            notes = inst.update(keypoints, conf_threshold=0.5)
            for n in notes: play_sound(n)   # n 是音區編號 0..N-1
    """

    def __init__(self, num_zones=5, width=640, cooldown_sec=0.15):
        self.num_zones = num_zones
        self.width = width
        self.cooldown_sec = cooldown_sec
        # 記錄每隻手上一次所在的音區，用來判斷「新揮入」
        self._last_zone = {"L": None, "R": None}
        self._last_fire = {}  # zone -> last time，防止同區狂響

    def _zone_of(self, x):
        """x 座標 → 音區編號"""
        if x is None:
            return None
        z = int(x / self.width * self.num_zones)
        return max(0, min(self.num_zones - 1, z))

    def update(self, keypoints, conf_threshold=0.5):
        """回傳這一格要觸發的音區清單（可能 0、1 或 2 個音，雙手）"""
        if keypoints is None:
            self._last_zone = {"L": None, "R": None}
            return []

        fired = []
        now = time.time()

        for hand, idx in (("L", LEFT_WRIST), ("R", RIGHT_WRIST)):
            x, y, c = keypoints[idx]
            zone = self._zone_of(x) if c >= conf_threshold else None

            # 「新揮入」一個音區（跟上一格不同區）才觸發
            if zone is not None and zone != self._last_zone[hand]:
                last = self._last_fire.get(zone, 0)
                if now - last >= self.cooldown_sec:
                    fired.append(zone)
                    self._last_fire[zone] = now
            self._last_zone[hand] = zone

        return fired
