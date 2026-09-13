"""
tracker.py —— 簡單多目標追蹤 + 越線計數(純邏輯,可離線測試)。

CentroidTracker:用 IoU + 中心距離把相鄰幀的框配對,給穩定 ID;消失 N 幀就刪。
LineCounter:每個 ID 的中心點跨過計數線(y=line_y)時,依方向計一次(上行/下行),每 ID 只算一次。
"""
import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih; ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    __slots__ = ("id", "box", "label", "missed", "history")
    def __init__(self, tid, box, label):
        self.id, self.box, self.label, self.missed = tid, box, label, 0
        self.history = [self.center()]

    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class CentroidTracker:
    def __init__(self, max_missed=10, iou_thresh=0.2, dist_thresh=80):
        self.tracks = {}; self.next_id = 1
        self.max_missed, self.iou_thresh, self.dist_thresh = max_missed, iou_thresh, dist_thresh

    def update(self, detections):
        """detections: [(label, score, (x1,y1,x2,y2))] → 回傳目前存活的 tracks(list)。"""
        unmatched = list(range(len(detections))); used = set()
        # 依 IoU 由高到低配對
        pairs = []
        for tid, tr in self.tracks.items():
            for j, (lab, sc, box) in enumerate(detections):
                s = iou(tr.box, box)
                if s < self.iou_thresh:
                    cx, cy = tr.center(); dx = (box[0] + box[2]) / 2 - cx; dy = (box[1] + box[3]) / 2 - cy
                    if (dx * dx + dy * dy) ** 0.5 > self.dist_thresh:
                        continue
                    s = 0.01
                pairs.append((s, tid, j))
        pairs.sort(reverse=True)
        matched_t = set()
        for s, tid, j in pairs:
            if tid in matched_t or j in used:
                continue
            tr = self.tracks[tid]; tr.box = detections[j][2]; tr.label = detections[j][0]; tr.missed = 0
            tr.history.append(tr.center()); tr.history = tr.history[-30:]
            matched_t.add(tid); used.add(j)
        created = set()
        for j in range(len(detections)):
            if j not in used:
                lab, sc, box = detections[j]
                self.tracks[self.next_id] = Track(self.next_id, box, lab); created.add(self.next_id); self.next_id += 1
        for tid in list(self.tracks):
            if tid not in matched_t and tid not in created:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
        return list(self.tracks.values())


class LineCounter:
    def __init__(self, line_y):
        self.line_y = line_y; self.counted = {}; self.counts = {}   # counts[label][dir]
        self.events = []

    def update(self, tracks):
        new = []
        for tr in tracks:
            if tr.id in self.counted or len(tr.history) < 2:
                continue
            (_, y0), (_, y1) = tr.history[-2], tr.history[-1]
            if (y0 < self.line_y <= y1) or (y1 < self.line_y <= y0):
                d = "down" if y1 > y0 else "up"
                self.counted[tr.id] = d
                self.counts.setdefault(tr.label, {"up": 0, "down": 0})[d] += 1
                new.append((tr.id, tr.label, d)); self.events.append((tr.label, d))
        return new

    def total(self):
        return sum(v["up"] + v["down"] for v in self.counts.values())
