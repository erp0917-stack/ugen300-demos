# -*- coding: utf-8 -*-
"""
face_engine.py —— 人臉偵測(SCRFD)+ 人臉特徵(ArcFace MobileFaceNet)+ 本機人臉庫。

    det = FaceDetector("scrfd_10g.hef")            # faces = det.detect(bgr) → [dict(box, score, kps(5,2))]
    emb = FaceEmbedder("arcface_mobilefacenet.hef") # v = emb.embed(bgr, kps) → 512-d 單位向量(對齊失敗回傳 None)
    db  = FaceDB("faces.json")                      # db.add(name, vec, thumb) / db.match(vec) → (name, sim)

SCRFD 解碼:三個 stride(8/16/32)各有 score(H,W,2)、bbox(H,W,8)、kps(H,W,20),每格 2 個 anchor,
距離單位為 stride。ArcFace 用 5 點相似變換對齊到 112x112 標準模板。
兩個模型在 Hailo-8L(UGen200)的 Model Zoo 清單中同樣存在。
"""
import base64
import json
import os

import cv2
import numpy as np

from hailo_model import HailoModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
EMB_DIM = 512
# ArcFace 112x112 標準五點模板(左眼、右眼、鼻、左嘴角、右嘴角)—— 與 insightface face_align.arcface_src 逐值相同
_ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                              [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _nms(boxes, scores, thr=0.4):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1); order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thr]
    return keep


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih; ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class FaceDetector:
    def __init__(self, hef="scrfd_10g.hef", conf=0.5, nms=0.4):
        self.m = HailoModel(hef if os.path.isabs(hef) else os.path.join(_HERE, hef)); self.conf = conf; self.nms = nms
        # 依空間大小與通道數把輸出層對到 (stride, 種類)
        self.layers = {}
        for n, shp in self.m.output_shapes.items():
            h, c = shp[0], shp[-1]; s = self.m.input_h // h
            kind = {2: "score", 8: "bbox", 20: "kps"}[c]
            self.layers[(s, kind)] = n
        self._anchors = {}
        for s in _STRIDES:
            h, w = self.m.input_h // s, self.m.input_w // s
            ys, xs = np.mgrid[0:h, 0:w]
            ac = np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float32) * s
            self._anchors[s] = np.repeat(ac, _NUM_ANCHORS, axis=0)
        self._scores_are_logits = None   # 第一次推論時決定(HEF 有沒有把 sigmoid 編進去),之後固定

    def _letterbox(self, bgr):
        h, w = bgr.shape[:2]; r = min(self.m.input_w / w, self.m.input_h / h)
        nw, nh = max(1, int(w * r)), max(1, int(h * r))
        canvas = np.zeros((self.m.input_h, self.m.input_w, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), r

    def detect(self, bgr, max_faces=20):
        inp, r = self._letterbox(bgr); out = self.m.infer(inp)
        if self._scores_are_logits is None:
            allsc = np.concatenate([out[self.layers[(s, "score")]].reshape(-1) for s in _STRIDES])
            self._scores_are_logits = bool(allsc.min() < 0 or allsc.max() > 1)
        boxes, scores, kpss = [], [], []
        for s in _STRIDES:
            sc = out[self.layers[(s, "score")]].reshape(-1)
            if self._scores_are_logits: sc = 1 / (1 + np.exp(-sc))
            bb = out[self.layers[(s, "bbox")]].reshape(-1, 4) * s
            kp = out[self.layers[(s, "kps")]].reshape(-1, 5, 2) * s
            idx = np.where(sc >= self.conf)[0]
            if idx.size == 0: continue
            ac = self._anchors[s][idx]
            b = np.stack([ac[:, 0] - bb[idx, 0], ac[:, 1] - bb[idx, 1], ac[:, 0] + bb[idx, 2], ac[:, 1] + bb[idx, 3]], 1)
            k = kp[idx] + ac[:, None, :]
            boxes.append(b); scores.append(sc[idx]); kpss.append(k)
        if not boxes: return []
        boxes = np.concatenate(boxes); scores = np.concatenate(scores); kpss = np.concatenate(kpss)
        keep = _nms(boxes, scores, self.nms)[:max_faces]
        faces = []
        for i in keep:
            x1, y1, x2, y2 = (boxes[i] / r).astype(int)
            faces.append(dict(box=(int(x1), int(y1), int(x2), int(y2)), score=float(scores[i]), kps=kpss[i] / r))
        faces.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        return faces


class FaceEmbedder:
    def __init__(self, hef="arcface_mobilefacenet.hef"):
        self.m = HailoModel(hef if os.path.isabs(hef) else os.path.join(_HERE, hef))

    @staticmethod
    def align(bgr, kps):
        """五點相似變換對齊到 112x112;關鍵點退化(估不出變換)時回傳 None。"""
        # RANSAC 門檻設很大 = 五點全當 inlier 做最小平方(等價 insightface 的 Umeyama);LMEDS 會隨機丟點,對齊會跳
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), _ARCFACE_TEMPLATE, method=cv2.RANSAC, ransacReprojThreshold=1e4, refineIters=10)
        if M is None: return None
        return cv2.warpAffine(bgr, M, (112, 112), borderValue=0)

    def embed(self, bgr, kps):
        face = self.align(bgr, kps)
        if face is None: return None
        v = self.m.infer(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        v = next(iter(v.values())).reshape(-1).astype(np.float32)
        if not np.isfinite(v).all(): return None
        return v / (np.linalg.norm(v) + 1e-6)


def square_crop(img, box, pad=0.25):
    """以人臉框中心取正方形(含邊距),給縮圖用,不會壓扁。"""
    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + pad)
    a, b = int(max(0, cx - half)), int(max(0, cy - half)); c, d = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    crop = img[b:d, a:c]
    if not crop.size: return img[:64, :64]
    h, w = crop.shape[:2]; m = min(h, w); oy, ox = (h - m) // 2, (w - m) // 2   # 貼邊時夾成長方形,再切回正方形
    return crop[oy:oy + m, ox:ox + m]


def _b64_thumb(bgr):
    ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (64, 64)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _thumb_from_b64(s):
    if not s: return None
    try: return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)
    except Exception: return None


class FaceDB:
    """人臉庫:每人一筆 {name, vec(512), thumb(base64 jpg)};只存向量與 64px 縮圖。
    讀寫失敗不丟例外:save() 回傳 False 並把原因放在 last_error(給畫面顯示)。"""
    def __init__(self, path):
        self.path = path; self.people = []; self.last_error = None
        if os.path.exists(path):
            try:
                data = json.load(open(path, encoding="utf-8"))
                if not isinstance(data, list): raise ValueError("頂層不是陣列")
                loaded, bad = [], 0
                for p in data:
                    try:
                        vec = np.asarray(p["vec"], np.float32)
                        if vec.shape != (EMB_DIM,) or not np.isfinite(vec).all(): raise ValueError(f"向量長度 {vec.shape} 不是 {EMB_DIM} 或含 NaN")
                        loaded.append(dict(name=str(p["name"]), vec=vec, thumb=_thumb_from_b64(p.get("thumb", ""))))
                    except Exception as e:  # noqa: BLE001  單筆壞資料只略過,其他人保留
                        bad += 1; print(f"[FaceDB] 略過一筆壞資料:{type(e).__name__} {e}")
                self.people = loaded
                if bad: self.last_error = f"faces.json 有 {bad} 筆壞資料已略過"
            except Exception as e:  # noqa: BLE001  整檔壞掉:不載入,並先備份原檔避免之後 save() 覆蓋
                self.last_error = f"faces.json 讀取失敗:{type(e).__name__}"; print("[FaceDB]", self.last_error, e)
                try:
                    import shutil; shutil.copy(path, path + ".bak")
                except Exception: pass

    def save(self):
        try:
            tmp = self.path + ".tmp"
            json.dump([dict(name=p["name"], vec=[round(float(x), 5) for x in p["vec"]], thumb=_b64_thumb(p["thumb"]) if p["thumb"] is not None else "")
                       for p in self.people], open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, self.path); self.last_error = None; return True
        except Exception as e:  # noqa: BLE001
            self.last_error = f"faces.json 無法寫入:{type(e).__name__}(檔案被其他程式開著?)"; print("[FaceDB]", self.last_error, e); return False

    def add(self, name, vec, thumb=None):
        """新增或覆蓋同名;回傳 (是否為覆蓋, 是否存檔成功)。向量含 NaN 直接拒絕。"""
        vec = np.asarray(vec, np.float32)
        if vec.shape != (EMB_DIM,) or not np.isfinite(vec).all():
            self.last_error = "特徵向量異常,未建檔"; return False, False
        existed = any(p["name"] == name for p in self.people)
        self.people = [p for p in self.people if p["name"] != name]
        self.people.append(dict(name=name, vec=np.asarray(vec, np.float32), thumb=thumb))
        return existed, self.save()

    def remove_last(self):
        """刪最後一筆;存檔失敗就放回去並回傳 None(畫面顯示 last_error)。"""
        if not self.people: self.last_error = None; return None
        p = self.people.pop()
        if not self.save(): self.people.append(p); return None
        return p["name"]

    def match(self, vec, thr=0.45, margin=0.08):
        """回傳 (name, sim);沒有超過門檻、或第一名與第二名太接近(分不清是誰)時回傳 (None, best_sim)。"""
        if not self.people or vec is None: return None, 0.0
        sims = np.array([float(np.dot(p["vec"], vec)) for p in self.people])
        order = np.argsort(sims)[::-1]; best = sims[order[0]]
        if not np.isfinite(best) or best < thr: return None, float(best) if np.isfinite(best) else 0.0
        if len(order) > 1 and best - sims[order[1]] < margin and sims[order[1]] >= thr: return None, float(best)
        return self.people[order[0]]["name"], float(best)

    def names(self): return [p["name"] for p in self.people]
