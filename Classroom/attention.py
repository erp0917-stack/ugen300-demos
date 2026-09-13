# -*- coding: utf-8 -*-
"""
attention.py —— 教室儀表板的純邏輯層(不碰模型、不碰鏡頭,可離線測試)。

輸入:每個人的 17 個 COCO 關鍵點 (x, y, conf) 與畫面上的手機框;
輸出:每個人的狀態(舉手 / 低頭 / 滑手機 / 趴睡 / 轉頭 / 專心)與其所在九宮格。

規則(影像座標 y 向下):
  舉手   手腕高於鼻子(或鼻子看不到時高於肩線)                      持續 RAISE_HOLD 秒
  低頭   兩眼平均高度低於兩耳平均高度,或鼻子貼近肩線               持續 DOWN_HOLD 秒
  趴睡   鼻子低於肩線,或肩膀看得到但整個頭看不到                   持續 SLEEP_HOLD 秒
  轉頭   只看得到單邊眼睛(另一眼信心極低)                           持續 TURN_HOLD 秒
  滑手機 手機框中心落在此人框內(下 2/3 區)                           持續 PHONE_HOLD 秒
不專心的優先序:滑手機 > 趴睡 > 低頭 > 轉頭;舉手可與其他狀態並存。
"""
import time

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO = 0, 1, 2, 3, 4, 5, 6
L_WRI, R_WRI, L_HIP, R_HIP = 9, 10, 11, 12
KP_CONF = 0.3
RAISE_HOLD, DOWN_HOLD, PHONE_HOLD, SLEEP_HOLD, TURN_HOLD = 0.5, 2.0, 3.0, 3.0, 5.0
RULES = ("head_down", "phone", "sleeping", "turned")   # 可由使用者開關的四種不專心
RULE_NAMES = {"head_down": "低頭", "phone": "滑手機", "sleeping": "趴睡", "turned": "轉頭"}
PRIORITY = ("phone", "sleeping", "head_down", "turned")


def _ok(kps, i, thr=KP_CONF):
    return kps[i][2] >= thr


def bbox_from_kps(kps, pad=0.25):
    """由可信關鍵點推出人框(含邊距);沒有可信點回傳 None。"""
    pts = [(x, y) for x, y, c in kps if c >= KP_CONF]
    if len(pts) < 2: return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys); w, h = max(w, 20), max(h, 20)
    return (int(min(xs) - w * pad), int(min(ys) - h * pad * 1.6), int(max(xs) + w * pad), int(max(ys) + h * pad))


def shoulder_line(kps):
    if _ok(kps, L_SHO) and _ok(kps, R_SHO):
        return (kps[L_SHO][1] + kps[R_SHO][1]) / 2, abs(kps[L_SHO][0] - kps[R_SHO][0])
    return None, None


def instant_flags(kps):
    """單幀的瞬時判斷,回傳 dict(raise, head_down, sleeping, turned) 布林。"""
    f = dict(raise_=False, head_down=False, sleeping=False, turned=False)
    sho_y, sho_w = shoulder_line(kps)
    # 舉手
    ref_y = kps[NOSE][1] if _ok(kps, NOSE) else sho_y
    if ref_y is not None:
        for w in (L_WRI, R_WRI):
            if _ok(kps, w) and kps[w][1] < ref_y: f["raise_"] = True
    # 低頭 / 趴睡
    if sho_y is not None and sho_w and sho_w > 0:
        head_seen = _ok(kps, NOSE) or _ok(kps, L_EYE) or _ok(kps, R_EYE)
        if not head_seen:
            f["sleeping"] = True
        else:
            if _ok(kps, NOSE):
                if kps[NOSE][1] > sho_y: f["sleeping"] = True
                elif kps[NOSE][1] > sho_y - 0.30 * sho_w: f["head_down"] = True
            eyes = [kps[i][1] for i in (L_EYE, R_EYE) if _ok(kps, i)]
            ears = [kps[i][1] for i in (L_EAR, R_EAR) if _ok(kps, i)]
            if eyes and ears and sum(eyes) / len(eyes) > sum(ears) / len(ears) + 0.06 * sho_w:
                f["head_down"] = True
    # 轉頭:一眼清楚、另一眼幾乎看不到
    le, re = kps[L_EYE][2], kps[R_EYE][2]
    if (le >= 0.5 and re < 0.15) or (re >= 0.5 and le < 0.15): f["turned"] = True
    return f


def phone_in_box(phone_boxes, box):
    """任何手機中心落在人框內(且在人框上緣往下 1/6 以後,排除舉高拍照的情況以外都算)。"""
    if box is None: return False
    x1, y1, x2, y2 = box; h = y2 - y1
    for px1, py1, px2, py2 in phone_boxes:
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        if x1 <= cx <= x2 and y1 + h / 6 <= cy <= y2: return True
    return False


def grid_cell(box, W, H):
    """九宮格:回傳 (row, col),row 0=後排(畫面上方)…2=前排,col 0=左…2=右(以鏡頭視角)。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(2, max(0, int(cy * 3 / H))), min(2, max(0, int(cx * 3 / W)))


GRID_ROW = ("後", "中", "前"); GRID_COL = ("左", "中", "右")


class PersonState:
    """每個追蹤 ID 一份:記各種狀態的起始時間,超過持續門檻才成立。"""
    __slots__ = ("since", "state", "raised", "box", "cell")
    def __init__(self):
        self.since = {}; self.state = "ok"; self.raised = False; self.box = None; self.cell = (1, 1)

    def _hold(self, key, active, hold, now):
        if not active: self.since.pop(key, None); return False
        self.since.setdefault(key, now)
        return now - self.since[key] >= hold

    def update(self, kps, phone_boxes, enabled, now=None, box=None):
        now = time.time() if now is None else now
        self.box = box or bbox_from_kps(kps)
        f = instant_flags(kps)
        self.raised = self._hold("raise", f["raise_"], RAISE_HOLD, now)
        held = dict(
            phone=self._hold("phone", phone_in_box(phone_boxes, self.box), PHONE_HOLD, now),
            sleeping=self._hold("sleeping", f["sleeping"], SLEEP_HOLD, now),
            head_down=self._hold("head_down", f["head_down"] and not f["sleeping"], DOWN_HOLD, now),
            turned=self._hold("turned", f["turned"], TURN_HOLD, now),
        )
        self.state = next((k for k in PRIORITY if enabled.get(k, True) and held[k]), "ok")
        return self.state


def summarize(states, W, H):
    """states: {tid: PersonState} → 面板數字。"""
    out = dict(raised=0, inattentive=0, by=dict.fromkeys(RULES, 0), grid=[[0] * 3 for _ in range(3)])
    for s in states.values():
        if s.raised:
            out["raised"] += 1
            if s.box: r, c = grid_cell(s.box, W, H); out["grid"][r][c] += 1
        if s.state != "ok": out["inattentive"] += 1; out["by"][s.state] += 1
    return out
