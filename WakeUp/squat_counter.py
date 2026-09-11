"""
squat_counter.py —— 深蹲計數狀態機(純邏輯,不碰裝置,可離線測試)

輸入:每一幀的 17 點關鍵點 list[(x, y, conf)],COCO 順序。
規則:
  - 取左右膝角(髖–膝–踝),用「看得到且較可信」的那一邊;兩邊都有就取平均。
  - 站立:膝角 > UP_DEG;蹲下:膝角 < DOWN_DEG。
  - 一次完整的「站 → 蹲 → 站」計 1 下;中間需連續 CONFIRM 幀才切換狀態,避免抖動。
  - 看不到腿(關鍵點信心不足)時狀態不變、回傳 visible=False。
"""
import math

L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def _pt(kps, idx, conf):
    x, y, c = kps[idx]
    return (x, y) if c >= conf else None


def _angle(a, b, c):
    """b 為頂點的夾角(度)。任一點缺就回 None。"""
    if not (a and b and c):
        return None
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    na = math.hypot(bax, bay); nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return None
    cos = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nc)))
    return math.degrees(math.acos(cos))


def knee_angle(kps, conf=0.4):
    """回傳膝角(度)或 None。兩邊都可用取平均,只有一邊就用那一邊。"""
    lk = _angle(_pt(kps, L_HIP, conf), _pt(kps, L_KNEE, conf), _pt(kps, L_ANK, conf))
    rk = _angle(_pt(kps, R_HIP, conf), _pt(kps, R_KNEE, conf), _pt(kps, R_ANK, conf))
    vals = [v for v in (lk, rk) if v is not None]
    return sum(vals) / len(vals) if vals else None


class SquatCounter:
    STAND, DOWN = "stand", "down"

    def __init__(self, target=5, up_deg=160.0, down_deg=105.0, confirm_frames=3, conf=0.4):
        self.target = target
        self.up_deg, self.down_deg = up_deg, down_deg
        self.confirm = confirm_frames
        self.conf = conf
        self.reset()

    def reset(self):
        self.count = 0
        self.state = self.STAND
        self._pending = None      # 候選狀態
        self._pending_n = 0       # 候選連續幀數
        self.angle = None

    @property
    def done(self):
        return self.count >= self.target

    def update(self, kps):
        """餵一幀。回傳 dict(count, state, angle, visible, done, just_counted)。"""
        just = False
        ang = knee_angle(kps, self.conf) if kps else None
        self.angle = ang
        if ang is None:
            self._pending = None; self._pending_n = 0
            return self._status(visible=False, just_counted=False)

        if ang < self.down_deg:
            cand = self.DOWN
        elif ang > self.up_deg:
            cand = self.STAND
        else:
            cand = None  # 中間過渡區,不改變狀態

        if cand is None or cand == self.state:
            self._pending = None; self._pending_n = 0
        else:
            if cand == self._pending:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = cand, 1
            if self._pending_n >= self.confirm:
                if self.state == self.DOWN and cand == self.STAND and not self.done:
                    self.count += 1; just = True
                self.state = cand
                self._pending = None; self._pending_n = 0
        return self._status(visible=True, just_counted=just)

    def _status(self, visible, just_counted):
        return dict(count=self.count, state=self.state, angle=self.angle,
                    visible=visible, done=self.done, just_counted=just_counted)
