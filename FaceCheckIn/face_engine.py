# -*- coding: utf-8 -*-
"""
face_engine.py —— 人臉偵測(SCRFD)+ 人臉特徵(ArcFace MobileFaceNet)+ 本機人臉庫。

    det = FaceDetector("scrfd_10g.hef")            # faces = det.detect(bgr) → [dict(box, score, kps(5,2))]
    emb = FaceEmbedder("arcface_mobilefacenet.hef") # v = emb.embed(bgr, kps) → 512-d 單位向量
    db  = FaceDB("faces.json")                      # db.add(name, vec, thumb) / db.match(vec) → (name, sim)

SCRFD 解碼:三個 stride(8/16/32)各有 score(H,W,2)、bbox(H,W,8)、kps(H,W,20),每格 2 個 anchor,
距離單位為 stride。ArcFace 用 5 點相似變換對齊到 112x112 標準模板。
"""
import base64
import json
import os

import cv2
import numpy as np

from hailo_model import HailoModel

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
# ArcFace 112x112 標準五點模板(左眼、右眼、鼻、左嘴角、右嘴角)
_ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [41.5493, 92.3655],
                              [70.7299, 92.2041], [56.1396, 92.2848]], dtype=np.float32)


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


class FaceDetector:
    def __init__(self, hef="scrfd_10g.hef", conf=0.5, nms=0.4):
        self.m = HailoModel(hef); self.conf = conf; self.nms = nms
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

    def _letterbox(self, bgr):
        h, w = bgr.shape[:2]; r = min(self.m.input_w / w, self.m.input_h / h)
        nw, nh = int(w * r), int(h * r)
        canvas = np.zeros((self.m.input_h, self.m.input_w, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), r

    def detect(self, bgr, max_faces=20):
        inp, r = self._letterbox(bgr); out = self.m.infer(inp)
        boxes, scores, kpss = [], [], []
        for s in _STRIDES:
            sc = out[self.layers[(s, "score")]].reshape(-1)
            if sc.min() < 0 or sc.max() > 1: sc = 1 / (1 + np.exp(-sc))
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
        self.m = HailoModel(hef)

    @staticmethod
    def align(bgr, kps):
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), _ARCFACE_TEMPLATE, method=cv2.LMEDS)
        return cv2.warpAffine(bgr, M, (112, 112), borderValue=0)

    def embed(self, bgr, kps):
        face = self.align(bgr, kps)
        v = self.m.infer(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        v = next(iter(v.values())).reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-6)


def _b64_thumb(bgr):
    ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (64, 64)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _thumb_from_b64(s):
    if not s: return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)


class FaceDB:
    """人臉庫:每人一筆 {name, vec(512), thumb(base64 jpg)};只存向量與 64px 縮圖。"""
    def __init__(self, path):
        self.path = path; self.people = []
        if os.path.exists(path):
            try:
                for p in json.load(open(path, encoding="utf-8")):
                    self.people.append(dict(name=p["name"], vec=np.asarray(p["vec"], np.float32), thumb=_thumb_from_b64(p.get("thumb", ""))))
            except (OSError, ValueError, KeyError) as e:
                print(f"[FaceDB] 讀取失敗,改用空白庫:{e}")

    def save(self):
        json.dump([dict(name=p["name"], vec=[round(float(x), 5) for x in p["vec"]], thumb=_b64_thumb(p["thumb"]) if p["thumb"] is not None else "")
                   for p in self.people], open(self.path, "w", encoding="utf-8"), ensure_ascii=False)

    def add(self, name, vec, thumb=None):
        self.people = [p for p in self.people if p["name"] != name]
        self.people.append(dict(name=name, vec=np.asarray(vec, np.float32), thumb=thumb)); self.save()

    def remove_last(self):
        if self.people: p = self.people.pop(); self.save(); return p["name"]
        return None

    def match(self, vec, thr=0.45):
        """回傳 (name, sim);沒有超過門檻回傳 (None, best_sim)。"""
        if not self.people: return None, 0.0
        sims = [float(np.dot(p["vec"], vec)) for p in self.people]
        i = int(np.argmax(sims))
        return (self.people[i]["name"] if sims[i] >= thr else None), sims[i]

    def names(self): return [p["name"] for p in self.people]
