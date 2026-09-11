"""
gestures.py —— 由 21 點手部關鍵點判斷靜態手勢(純幾何規則,不需訓練、不碰裝置)。

MediaPipe 索引:0 手腕;拇指 1-4;食指 5-8;中指 9-12;無名指 13-16;小指 17-20(x4 為指尖)。
手指「伸直」= 指尖到手腕的距離 > 第二關節(PIP)到手腕的距離 × 1.15(與手的方向無關)。
拇指「伸直」= 拇指尖到小指根(17)的距離 > 拇指根(2)到小指根的距離 × 1.25。
"""
import math

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(pts):
    """回傳 dict:thumb/index/middle/ring/pinky → True(伸直)/False(彎曲)。"""
    w = pts[0]
    st = {}
    for f in ("index", "middle", "ring", "pinky"):
        st[f] = _d(pts[TIP[f]], w) > _d(pts[PIP[f]], w) * 1.15
    st["thumb"] = _d(pts[4], pts[17]) > _d(pts[2], pts[17]) * 1.25
    return st


def hand_size(pts):
    return max(_d(pts[0], pts[9]), 1e-6)  # 手腕到中指根


def classify(pts):
    """回傳 (label, emoji_or_text)。pts: 21×(x,y)。"""
    st = finger_states(pts)
    ext = [st[f] for f in ("index", "middle", "ring", "pinky")]
    n = sum(ext); size = hand_size(pts)
    thumb_index_close = _d(pts[4], pts[8]) < size * 0.4

    if thumb_index_close and st["middle"] and st["ring"] and st["pinky"]:
        return "OK", "OK"
    if thumb_index_close and n == 0:
        return "比心", "比心"
    if st["thumb"] and n == 0:
        # 拇指朝上 = 讚;朝下 = 倒讚(y 向下為正)
        return ("讚", "讚") if pts[4][1] < pts[2][1] else ("倒讚", "倒讚")
    if st["index"] and st["pinky"] and not st["middle"] and not st["ring"]:
        return "搖滾", "搖滾"
    if n == 0:
        return "拳頭", "0 / 拳頭"
    if ext == [True, False, False, False]:
        return "1", "1 / 指"
    if ext == [True, True, False, False]:
        return "2", "2 / 剪刀 / 耶"
    if ext == [True, True, True, False]:
        return "3", "3"
    if n == 4 and not st["thumb"]:
        return "4", "4"
    if n == 4 and st["thumb"]:
        return "5", "5 / 手掌 / 停"
    if ext == [False, False, False, True]:
        return "小指", "小指 / 打勾勾"
    return "?", "辨識中…"


class GestureSmoother:
    """連續 N 幀同一結果才輸出,避免閃爍。"""
    def __init__(self, n=4):
        self.n = n; self.last = None; self.count = 0; self.stable = None

    def update(self, label):
        if label == self.last:
            self.count += 1
        else:
            self.last, self.count = label, 1
        if self.count >= self.n:
            self.stable = label
        return self.stable
