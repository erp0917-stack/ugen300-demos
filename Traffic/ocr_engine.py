"""
ocr_engine.py —— PaddleOCR v5 mobile(偵測 + 辨識)在 UGen300 上的封裝。

偵測 paddle_ocr_v5_mobile_detection.hef:輸入 544×960×3,輸出 544×960×1 機率圖(DB 演算法)
辨識 paddle_ocr_v5_mobile_recognition.hef:輸入 48×320×3,輸出 1×40×18385(CTC:0=blank, 1..18383=字典, 18384=空白)
字典 ppocrv5_dict.txt(18383 行)。

純邏輯(DB 後處理、CTC 解碼、文字框排序)不碰裝置,可離線測試。
"""
import os
import re

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DET_HEF = os.path.join(_HERE, "paddle_ocr_v5_mobile_detection.hef")
REC_HEF = os.path.join(_HERE, "paddle_ocr_v5_mobile_recognition.hef")
DICT_PATH = os.path.join(_HERE, "ppocrv5_dict.txt")
DET_H, DET_W = 544, 960
REC_H, REC_W = 48, 320


def load_dict(path=DICT_PATH):
    with open(path, encoding="utf-8") as f:
        chars = [line.rstrip("\n") for line in f]
    return ["<blank>"] + chars + [" "]   # 索引 0 = blank,最後 = 空白


# ---------- 純邏輯 ----------
def db_boxes(prob, thresh=0.3, box_thresh=0.5, unclip=1.6, min_size=4, max_boxes=50):
    """DB 機率圖 → 文字框 list[(x1,y1,x2,y2,score)],座標為機率圖尺度。"""
    binary = (prob > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours[:max_boxes * 4]:
        if len(c) < 3:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < min_size:
            continue
        score = float(prob[y:y + h, x:x + w].mean())
        if score < box_thresh:
            continue
        # unclip:依面積/周長比外擴
        area, peri = w * h, 2 * (w + h)
        d = area * unclip / max(peri, 1) * 0.5
        x1, y1 = max(0, int(x - d)), max(0, int(y - d))
        x2, y2 = min(prob.shape[1] - 1, int(x + w + d)), min(prob.shape[0] - 1, int(y + h + d))
        out.append((x1, y1, x2, y2, score))
    out.sort(key=lambda b: -b[4])
    return out[:max_boxes]


def ctc_decode(logits, charset):
    """logits (T, C) → (text, mean_conf)。貪婪解碼、去重複、去 blank。"""
    idx = logits.argmax(axis=1); probs = logits.max(axis=1)
    chars, confs, prev = [], [], -1
    for i, p in zip(idx, probs):
        if i != prev and i != 0:
            chars.append(charset[int(i)] if int(i) < len(charset) else ""); confs.append(float(p))
        prev = i
    return "".join(chars), (float(np.mean(confs)) if confs else 0.0)


def sort_boxes_reading_order(boxes):
    """由上到下、由左到右。"""
    return sorted(boxes, key=lambda b: (round(b[1] / 20), b[0]))


PLATE_RE = re.compile(r"^(?:[A-Z]{2,3}-?\d{3,4}|\d{3,4}-?[A-Z]{2,3}|[A-Z]{3}-?\d{4}|\d{4}-?[A-Z]{2})$")


def normalize_plate(text):
    """把辨識出的字串整理成台灣車牌格式;不像車牌回 None。"""
    t = re.sub(r"[^A-Za-z0-9\-]", "", text).upper().replace("O", "0") if text else ""
    t = t.replace("--", "-")
    # 常見誤認:字母區的 0→O、數字區的 O→0 已上面處理;I→1
    if not t:
        return None
    m = PLATE_RE.match(t) or PLATE_RE.match(t.replace("-", ""))
    if not m:
        return None
    s = m.group(0).replace("-", "")
    # 標準化成 XXX-1234 / 1234-XX
    a = re.match(r"^([A-Z]{2,3})(\d{3,4})$", s)
    if a: return f"{a.group(1)}-{a.group(2)}"
    b = re.match(r"^(\d{3,4})([A-Z]{2,3})$", s)
    if b: return f"{b.group(1)}-{b.group(2)}"
    return s


# ---------- 需要裝置 ----------
class _Model:
    def __init__(self, hef, float_out=False):
        import hailo_vdevice
        from hailo_platform import FormatType
        vd = hailo_vdevice.get()
        self.m = vd.create_infer_model(hef); self.m.set_batch_size(1)
        self.oname = self.m.output_names[0]
        if float_out:
            self.m.output(self.oname).set_format_type(FormatType.FLOAT32)
        self.c = self.m.configure()
        self.ishape = tuple(self.m.input().shape); self.oshape = tuple(self.m.output(self.oname).shape)
        self.dtype = np.float32 if float_out else np.uint8

    def __call__(self, rgb):
        b = self.c.create_bindings(); b.input().set_buffer(np.ascontiguousarray(rgb))
        out = np.zeros(self.oshape, self.dtype); b.output(self.oname).set_buffer(out)
        self.c.run([b], 10000)
        return out


class OCREngine:
    def __init__(self):
        self.det = _Model(DET_HEF); self.rec = _Model(REC_HEF, float_out=True)
        self.charset = load_dict()

    def detect(self, bgr):
        """回傳原圖座標的文字框 [(x1,y1,x2,y2,score)]。"""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (DET_W, DET_H)), cv2.COLOR_BGR2RGB)
        prob = self.det(rgb).reshape(DET_H, DET_W).astype(np.float32) / 255.0
        boxes = db_boxes(prob)
        sx, sy = w / DET_W, h / DET_H
        return [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), s) for x1, y1, x2, y2, s in boxes]

    def recognize(self, crop_bgr):
        """單一文字框 → (text, conf)。等比縮放到高 48、寬不足補白。"""
        h, w = crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return "", 0.0
        nw = min(REC_W, max(8, int(w * REC_H / h)))
        rs = cv2.resize(crop_bgr, (nw, REC_H))
        canvas = np.full((REC_H, REC_W, 3), 255, np.uint8); canvas[:, :nw] = rs
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        logits = self.rec(rgb).reshape(-1, len(self.charset) if self.rec.oshape[-1] == len(self.charset) else self.rec.oshape[-1])
        return ctc_decode(logits, self.charset)

    def read_all(self, bgr, min_conf=0.5):
        res = []
        for x1, y1, x2, y2, s in sort_boxes_reading_order(self.detect(bgr)):
            text, conf = self.recognize(bgr[y1:y2, x1:x2])
            if text and conf >= min_conf:
                res.append(dict(box=(x1, y1, x2, y2), text=text, conf=conf))
        return res
