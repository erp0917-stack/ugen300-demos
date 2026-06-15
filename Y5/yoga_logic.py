# -*- coding: utf-8 -*-
"""
yoga_logic.py  ——  瑜伽姿勢判斷 + 維持計時 + 即時教學回饋（純邏輯，不依賴硬體）
================================================================================
每個姿勢的檢查函式回傳 (ok, hint)：
  ok   = 是否擺對
  hint = 還沒對時，下一步該修什麼（英文短句，給畫面即時回饋用）
另外每個姿勢有 steps（英文分步說明，畫面列出）。

⚠️ 顯示文字一律英文（OpenCV 畫不出中文會亂碼）。中文只在註解與終端機。
"""

import time
import math
import random

NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def _get(kps, idx, conf=0.45):
    x, y, c = kps[idx]
    return (x, y) if c >= conf else None


def _angle(a, b, c):
    if not (a and b and c):
        return None
    bax = a[0]-b[0]; bay = a[1]-b[1]
    bcx = c[0]-b[0]; bcy = c[1]-b[1]
    dot = bax*bcx + bay*bcy
    na = math.hypot(bax, bay); nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/(na*nc)))))


def _torso(kps, conf=0.45):
    ls = _get(kps, L_SH, conf); rs = _get(kps, R_SH, conf)
    lh = _get(kps, L_HIP, conf); rh = _get(kps, R_HIP, conf)
    if not (ls and rs and lh and rh):
        return None
    return abs((lh[1]+rh[1])/2 - (ls[1]+rs[1])/2)


def _legs_visible(kps, conf=0.4):
    """至少一邊的膝+踝看得到（腿型姿勢才判斷得了）。"""
    left = _get(kps, L_KNEE, conf) and _get(kps, L_ANK, conf)
    right = _get(kps, R_KNEE, conf) and _get(kps, R_ANK, conf)
    return bool(left or right)


# ── 各姿勢檢查：回傳 (ok, hint) ────────────────────────────────────────
def check_tree(kps, conf=0.45):
    nose = _get(kps, NOSE, conf); lw = _get(kps, L_WR, conf); rw = _get(kps, R_WR, conf)
    if not nose:
        return False, "Show your head to camera"
    if not (lw and rw):
        return False, "Raise both hands so camera sees them"
    if lw[1] < nose[1] and rw[1] < nose[1]:
        return True, ""
    return False, "Raise BOTH hands above your head"


def check_warrior(kps, conf=0.45):
    ls = _get(kps, L_SH, conf); rs = _get(kps, R_SH, conf)
    lw = _get(kps, L_WR, conf); rw = _get(kps, R_WR, conf)
    if not (ls and rs):
        return False, "Show your shoulders to camera"
    if not (lw and rw):
        return False, "Raise both arms to the sides"
    sh_y = (ls[1]+rs[1])/2; sh_w = abs(ls[0]-rs[0])
    tol = max(40, sh_w*0.4)
    level = abs(lw[1]-sh_y) < tol and abs(rw[1]-sh_y) < tol
    wide = abs(lw[0]-rw[0]) > sh_w*1.3
    if not wide:
        return False, "Spread arms out WIDE"
    if not level:
        return False, "Keep arms LEVEL with shoulders"
    return True, ""


def check_warrior1(kps, conf=0.45):
    nose = _get(kps, NOSE, conf); lw = _get(kps, L_WR, conf); rw = _get(kps, R_WR, conf)
    if not (lw and rw and nose):
        return False, "Raise both hands overhead"
    if not (lw[1] < nose[1] and rw[1] < nose[1]):
        return False, "Raise BOTH hands above your head"
    if not _legs_visible(kps):
        return False, "Step back so camera sees your legs"
    lk = _angle(_get(kps, L_HIP, conf), _get(kps, L_KNEE, conf), _get(kps, L_ANK, conf))
    rk = _angle(_get(kps, R_HIP, conf), _get(kps, R_KNEE, conf), _get(kps, R_ANK, conf))
    if any(a is not None and a < 150 for a in (lk, rk)):
        return True, ""
    return False, "Bend your front knee (lunge)"


def check_triangle(kps, conf=0.45):
    ls = _get(kps, L_SH, conf); rs = _get(kps, R_SH, conf)
    lw = _get(kps, L_WR, conf); rw = _get(kps, R_WR, conf)
    if not (ls and rs and lw and rw):
        return False, "Show shoulders and both hands"
    sh_w = abs(ls[0]-rs[0])
    if sh_w < 1:
        return False, "Face the camera"
    if abs(lw[0]-rw[0]) <= sh_w*1.2:
        return False, "Spread arms apart"
    if abs(lw[1]-rw[1]) <= sh_w*0.7:
        return False, "Tilt: one hand UP, one hand DOWN"
    return True, ""


def check_squat(kps, conf=0.45):
    ls = _get(kps, L_SH, conf); rs = _get(kps, R_SH, conf)
    lw = _get(kps, L_WR, conf); rw = _get(kps, R_WR, conf)
    if not _legs_visible(kps):
        return False, "Step back so camera sees your legs"
    lk = _angle(_get(kps, L_HIP, conf), _get(kps, L_KNEE, conf), _get(kps, L_ANK, conf))
    rk = _angle(_get(kps, R_HIP, conf), _get(kps, R_KNEE, conf), _get(kps, R_ANK, conf))
    knees = [a for a in (lk, rk) if a is not None]
    if not knees or min(knees) >= 120:
        return False, "Bend knees and SQUAT down"
    if not (ls and rs and lw and rw):
        return False, "Bring hands to your chest"
    sh_w = abs(ls[0]-rs[0])
    if math.hypot(lw[0]-rw[0], lw[1]-rw[1]) < max(60, sh_w*0.5):
        return True, ""
    return False, "Put PALMS TOGETHER at chest"


def check_tree1(kps, conf=0.45):
    nose = _get(kps, NOSE, conf); lw = _get(kps, L_WR, conf); rw = _get(kps, R_WR, conf)
    if not (lw and rw and nose):
        return False, "Raise both hands overhead"
    if not (lw[1] < nose[1] and rw[1] < nose[1]):
        return False, "Raise BOTH hands above your head"
    if not _legs_visible(kps):
        return False, "Step back so camera sees your legs"
    la = _get(kps, L_ANK, conf); ra = _get(kps, R_ANK, conf)
    t = _torso(kps, conf)
    if not (la and ra and t):
        return False, "Show both feet to camera"
    if abs(la[1]-ra[1]) > t*0.4:
        return True, ""
    return False, "LIFT one foot off the ground"


# 顯示名稱、檢查函式、英文分步說明
POSES = {
    "tree":     ("Tree", check_tree,
                 ["1. Stand tall", "2. Raise both arms", "3. Hands ABOVE your head"]),
    "warrior":  ("Warrior II", check_warrior,
                 ["1. Feet apart", "2. Arms out to the sides", "3. Level with shoulders"]),
    "warrior1": ("Warrior I", check_warrior1,
                 ["1. Step one foot forward", "2. Bend front knee", "3. Raise both arms up"]),
    "triangle": ("Triangle", check_triangle,
                 ["1. Feet wide apart", "2. Lean to one side", "3. One arm up, one down"]),
    "squat":    ("Squat", check_squat,
                 ["1. Feet apart", "2. Bend knees, squat down", "3. Palms together at chest"]),
    "tree1":    ("Tree One-Leg", check_tree1,
                 ["1. Stand on one leg", "2. Lift the other foot", "3. Raise both arms up"]),
}


def is_full_body(kps, conf=0.35):
    """（保留備用）寬鬆全身判斷：頭、軀幹、手、腳大致都在。"""
    if not kps:
        return False
    head = _get(kps, NOSE, conf) is not None
    sh = (_get(kps, L_SH, conf) is not None) or (_get(kps, R_SH, conf) is not None)
    hip = (_get(kps, L_HIP, conf) is not None) or (_get(kps, R_HIP, conf) is not None)
    wr = (_get(kps, L_WR, conf) is not None) or (_get(kps, R_WR, conf) is not None)
    an = (_get(kps, L_ANK, conf) is not None) or (_get(kps, R_ANK, conf) is not None)
    return head and sh and hip and wr and an


class YogaCoach:
    """單一姿勢：判斷 + 維持計時 + 即時 hint。state 含 hint / steps。"""
    def __init__(self, pose_name="tree", hold_target=8.0, confirm_frames=3):
        self.pose_name = pose_name
        self.label, self._check, self.steps = POSES[pose_name]
        self.hold_target = hold_target
        self.confirm_frames = confirm_frames
        self._stable = 0
        self._accum = 0.0
        self._last_tick = None
        self._done = False
        self._hint = ""

    def update(self, keypoints, active=True):
        now = time.time()
        if not active:
            self._stable = 0; self._last_tick = None
            return self._state()

        if keypoints:
            ok, hint = self._check(keypoints)
        else:
            ok, hint = False, "Step into camera view"
        self._hint = hint

        if ok:
            self._stable += 1
        else:
            self._stable = 0

        confirmed = self._stable >= self.confirm_frames
        if confirmed:
            if self._last_tick is not None:
                self._accum += now - self._last_tick
            self._last_tick = now
            if self._accum >= self.hold_target:
                self._done = True
        else:
            self._last_tick = None

        return self._state(confirmed)

    def _state(self, confirmed=False):
        return {
            "label": self.label,
            "in_pose": confirmed,
            "held": min(self._accum, self.hold_target),
            "target": self.hold_target,
            "done": self._done,
            "hint": "" if (confirmed or self._done) else self._hint,
            "steps": self.steps,
        }

    def reset(self):
        self._stable = 0; self._accum = 0.0; self._last_tick = None
        self._done = False; self._hint = ""


class RandomCoach:
    """隨機出題：洗牌牌堆(一輪不重複)，六個全做完→完成狀態(不再循環)。按 restart 重玩。"""
    def __init__(self, hold_target=8.0, confirm_frames=3, switch_delay=2.0, pool=None):
        random.seed()
        self.hold_target = hold_target
        self.confirm_frames = confirm_frames
        self.switch_delay = switch_delay
        self.pool = pool if pool else list(POSES.keys())
        self.total = len(self.pool)        # 一輪要做幾個（=姿勢總數）
        self._start_round()

    def _start_round(self):
        self.completed = 0
        self._deck = list(self.pool)
        random.shuffle(self._deck)
        self._phase = "active"
        self._celebrate_start = None
        self._celebrate_frames = 0
        self.finished = False
        self._coach = self._new_coach()

    def restart(self):
        """重玩一輪（重新洗牌）。"""
        self._start_round()

    def _draw_from_deck(self):
        # 一輪內不重複；牌堆空了不重洗（由 finished 控制結束）
        return self._deck.pop() if self._deck else None

    def _new_coach(self):
        name = self._draw_from_deck()
        if name is None:
            return None
        return YogaCoach(name, hold_target=self.hold_target,
                         confirm_frames=self.confirm_frames)

    def update(self, keypoints, active=True):
        # 已完成一輪：固定回報完成狀態，等使用者 restart 或離開
        if self.finished:
            return {"label": "All Done", "in_pose": False, "held": 0.0,
                    "target": self.hold_target, "done": False, "hint": "", "steps": [],
                    "completed": self.completed, "celebrating": False,
                    "celebrate_remain": 0.0, "pose_name": None, "finished": True}

        if self._phase == "celebrate":
            self._celebrate_frames += 1
            elapsed = time.time() - self._celebrate_start
            if elapsed >= self.switch_delay or self._celebrate_frames >= 90:
                self.completed += 1
                nxt = self._new_coach()
                if nxt is None:
                    # 一輪六個全做完 → 進完成狀態
                    self.finished = True
                    return self.update(keypoints, active)
                self._coach = nxt
                self._phase = "active"
                self._celebrate_start = None
                self._celebrate_frames = 0
                s = self._coach._state()
                s.update({"completed": self.completed, "celebrating": False,
                          "celebrate_remain": 0.0, "pose_name": self._coach.pose_name,
                          "finished": False})
                return s
            state = self._coach.update(keypoints, active=False)
            state["done"] = True
            state["celebrate_remain"] = max(0.0, self.switch_delay - elapsed)
            state["completed"] = self.completed
            state["celebrating"] = True
            state["pose_name"] = self._coach.pose_name
            state["finished"] = False
            return state

        state = self._coach.update(keypoints, active=active)
        if active and state["done"]:
            self._phase = "celebrate"
            self._celebrate_start = time.time()
            self._celebrate_frames = 0
            state["done"] = True
        state["completed"] = self.completed
        state["celebrating"] = (self._phase == "celebrate")
        state["celebrate_remain"] = 0.0
        state["pose_name"] = self._coach.pose_name
        state["finished"] = False
        return state

    def skip(self):
        """手動跳下一題（也會推進輪次；發完就完成）。"""
        if self.finished:
            return
        self.completed += 1
        nxt = self._new_coach()
        if nxt is None:
            self.finished = True
            return
        self._coach = nxt
        self._phase = "active"
        self._celebrate_start = None
        self._celebrate_frames = 0
