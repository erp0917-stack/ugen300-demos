# -*- coding: utf-8 -*-
"""
attention.py —— 教室儀表板的純邏輯層(不碰模型、不碰鏡頭,可離線測試)。

輸入:每個人的 17 個 COCO 關鍵點 (x, y, conf) 與「歸給此人的手機框」;
輸出:每個人的狀態(舉手 / 低頭 / 手機 / 趴下 / 轉頭 / 專心)與其所在九宮格。

規則(影像座標 y 向下;門檻以肩寬為尺度,並針對「筆電鏡頭俯視、近距離」校正過):
  舉手   手腕高於眼睛線 0.6 肩寬以上且手肘高於肩線(撥頭髮、抓頭不算);
         手腕出框看不到時,手肘要高於眼線才算                                   持續 RAISE_HOLD 秒
  低頭   兩眼平均高度低於兩耳 0.12 肩寬,或鼻子距肩線不到 0.18 肩寬            持續 DOWN_HOLD 秒
  趴下   鼻子低於肩線(背對鏡頭、頭出框都不算)                                 持續 SLEEP_HOLD 秒
  轉頭   一眼清楚、另一眼信心 < 0.3                                             持續 TURN_HOLD 秒
  手機   手機框中心在此人框內、肩線以下,且離任一可見手腕 1.2 肩寬內(桌上的不算) 持續 PHONE_HOLD 秒
不專心的優先序:手機 > 趴下 > 低頭 > 轉頭;舉手可與其他狀態並存。
"""
import time

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO = 0, 1, 2, 3, 4, 5, 6
L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP = 7, 8, 9, 10, 11, 12
KP_CONF = 0.3
RAISE_HOLD, DOWN_HOLD, PHONE_HOLD, SLEEP_HOLD, TURN_HOLD = 1.2, 2.0, 3.0, 3.0, 2.5
RAISE_ABOVE_EYES, PHONE_NEAR_WRIST = 0.6, 1.2   # 以肩寬為單位
DOWN_EYE_EAR, DOWN_NOSE_SHO, TURN_FAR_EYE = 0.12, 0.18, 0.30
MIN_SHO_W_FACE = 60      # 肩寬小於這個像素的人(太遠)不判轉頭/低頭:眼睛信心本來就低,會誤判
RULES = ("head_down", "phone", "sleeping", "turned")   # 可由使用者開關的四種不專心
RULE_NAMES = {"head_down": "低頭", "phone": "手機", "sleeping": "趴下", "turned": "轉頭"}
PRIORITY = ("phone", "sleeping", "head_down", "turned")


def _ok(kps, i, thr=KP_CONF):
    return kps[i][2] >= thr


def bbox_from_kps(kps, pad=0.25, idx=None):
    """由可信關鍵點推出框(含邊距);idx 指定只用哪些點(例如頭肩 0–6 做追蹤框)。沒有可信點回傳 None。"""
    pts = [(x, y) for j, (x, y, c) in enumerate(kps) if c >= KP_CONF and (idx is None or j in idx)]
    if len(pts) < 2: return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys); w, h = max(w, 20), max(h, 20)
    return (int(min(xs) - w * pad), int(min(ys) - h * pad * 1.6), int(max(xs) + w * pad), int(max(ys) + h * pad))


HEAD_SHOULDER = (0, 1, 2, 3, 4, 5, 6)


def shoulder_line(kps):
    if _ok(kps, L_SHO) and _ok(kps, R_SHO):
        return (kps[L_SHO][1] + kps[R_SHO][1]) / 2, abs(kps[L_SHO][0] - kps[R_SHO][0])
    return None, None


def instant_flags(kps, frame_h=None):
    """單幀的瞬時判斷,回傳 dict(raise_, head_down, sleeping, turned) 布林。frame_h 用於趴下的「肩膀在畫面下半」條件。"""
    f = dict(raise_=False, head_down=False, sleeping=False, turned=False)
    sho_y, sho_w = shoulder_line(kps)
    eyes_y = [kps[i][1] for i in (L_EYE, R_EYE) if _ok(kps, i)]
    eye_line = sum(eyes_y) / len(eyes_y) if eyes_y else (kps[NOSE][1] if _ok(kps, NOSE) else None)
    # 舉手:手腕高於眼線 0.6 肩寬 + 手肘高於肩線;手腕出框時要手肘高於眼線(整隻手臂舉直)
    if sho_y is not None and sho_w and sho_w >= MIN_SHO_W_FACE * 0.5 and eye_line is not None:   # 側身肩寬趨近 0 時門檻失效,不判
        for wri, elb in ((L_WRI, L_ELB), (R_WRI, R_ELB)):
            elbow_up = _ok(kps, elb) and kps[elb][1] < sho_y
            if _ok(kps, wri):
                if elbow_up and kps[wri][1] < eye_line - RAISE_ABOVE_EYES * sho_w: f["raise_"] = True
            elif _ok(kps, elb) and kps[elb][1] < eye_line:
                f["raise_"] = True
    # 低頭 / 趴下
    if sho_y is not None and sho_w and sho_w > 0:
        head_seen = _ok(kps, NOSE) or bool(eyes_y)
        if head_seen and sho_w >= MIN_SHO_W_FACE:   # 頭看不到(背對、出框)一律不判:分不出是趴下還是轉身
            if _ok(kps, NOSE):
                if kps[NOSE][1] > sho_y: f["sleeping"] = True
                elif kps[NOSE][1] > sho_y - DOWN_NOSE_SHO * sho_w: f["head_down"] = True
            ears = [kps[i][1] for i in (L_EAR, R_EAR) if _ok(kps, i)]
            if eyes_y and ears and sum(eyes_y) / len(eyes_y) > sum(ears) / len(ears) + DOWN_EYE_EAR * sho_w:
                f["head_down"] = True
    # 轉頭:一眼清楚、另一眼幾乎看不到
    le, re = kps[L_EYE][2], kps[R_EYE][2]
    if sho_w is not None and sho_w >= MIN_SHO_W_FACE and ((le >= 0.5 and re < TURN_FAR_EYE) or (re >= 0.5 and le < TURN_FAR_EYE)):
        f["turned"] = True
    return f


def wrists_of(kps):
    """可見的手腕座標清單。"""
    return [(kps[i][0], kps[i][1]) for i in (L_WRI, R_WRI) if _ok(kps, i)]


def assign_phones(phone_boxes, persons):
    """把每支手機歸給「框包含手機中心、手機在肩線以下、離手腕夠近、中心最近」的那一個人。
    persons: [(pid, box, sho_y, sho_w, wrists)];wrists 為可見手腕座標清單(看得到手腕時手機必須靠近手腕,桌上的不算)。
    回傳 {pid: [phone_box, ...]}。"""
    out = {}
    for pb in phone_boxes:
        cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        best, best_d = None, None
        for pid, box, sho_y, sho_w, wrists in persons:
            if box is None: continue
            x1, y1, x2, y2 = box
            if not (x1 <= cx <= x2 and y1 <= cy <= y2): continue
            if sho_y is not None and cy < sho_y: continue        # 舉到肩線以上(拍投影片)不算
            if wrists:
                scale = sho_w if sho_w else (x2 - x1) * 0.5     # 肩膀看不到就用人框寬的一半當尺度
                if min(((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5 for wx, wy in wrists) > PHONE_NEAR_WRIST * scale: continue
            d = ((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2
            if best is None or d < best_d: best, best_d = pid, d
        if best is not None: out.setdefault(best, []).append(pb)
    return out


def grid_cell(box, W, H):
    """九宮格:回傳 (row, col),row 0=後排(畫面上方)…2=前排,col 0=左…2=右(以鏡頭視角)。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(2, max(0, int(cy * 3 / H))), min(2, max(0, int(cx * 3 / W)))


GRID_ROW = ("後", "中", "前"); GRID_COL = ("左", "中", "右")


class PersonState:
    """每個追蹤 ID 一份:記各種狀態的起始時間,超過持續門檻才成立。"""
    __slots__ = ("since", "state", "raised", "box", "sho_y", "sho_w")
    def __init__(self):
        self.since = {}; self.state = "ok"; self.raised = False; self.box = None; self.sho_y = None; self.sho_w = None

    def _hold(self, key, active, hold, now):
        if not active: self.since.pop(key, None); return False
        self.since.setdefault(key, now)
        return now - self.since[key] >= hold

    def shift(self, dt):
        """畫面凍結了 dt 秒:把所有計時起點往後推,凍結期間不算持續。"""
        for k in self.since: self.since[k] += dt

    def update(self, kps, my_phones, enabled, now=None, box=None, frame_h=None):
        now = time.time() if now is None else now
        self.box = box or bbox_from_kps(kps); self.sho_y, self.sho_w = shoulder_line(kps)
        f = instant_flags(kps, frame_h)
        self.raised = self._hold("raise", f["raise_"], RAISE_HOLD, now)
        held = dict(
            phone=self._hold("phone", bool(my_phones), PHONE_HOLD, now),
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
