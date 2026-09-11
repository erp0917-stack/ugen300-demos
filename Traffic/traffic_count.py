"""
traffic_count.py —— 路口車流/人流計數:YOLO 偵測 → 追蹤 ID → 越線計數(上行/下行)→ 儀表板。

按鍵:↑/↓(或 w/s)移動計數線 / r 歸零 / q 離開
用法:python traffic_count.py --source 0 [--fullscreen]
      python traffic_count.py --video road.mp4 [--snapshot out.png]   (用影片檔演,現場沒路口也行)
      python traffic_count.py --selftest
輸出:count_log.csv(每筆越線事件)
"""
import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from hailo_detect import ObjectDetector as HailoDetector
from tracker import CentroidTracker, LineCounter
from ui_text import draw_text, badge

WIN = "UGen300 Traffic Count"
W, H, PANEL = 1280, 720, 400
CLASSES = {"person": "行人", "bicycle": "腳踏車", "motorcycle": "機車", "car": "汽車", "bus": "公車", "truck": "卡車"}
COLORS = {"person": (80, 220, 120), "bicycle": (255, 200, 60), "motorcycle": (255, 120, 60), "car": (60, 160, 255), "bus": (200, 80, 200), "truck": (120, 120, 255)}
C_BG, C_PANEL, C_TXT, C_DIM, C_ACC = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150), (255, 170, 40)
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "count_log.csv")


def compose(frame, tracks, line_y_rel, counter, rate_per_min, fps, err=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    h, w = frame.shape[:2]; s = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * s), int(h * s)))
    ly = int(line_y_rel * fr.shape[0])
    cv2.line(fr, (0, ly), (fr.shape[1], ly), C_ACC, 3)
    draw_text(fr, "計數線", (10, ly - 30), 22, C_ACC)
    for tr in tracks:
        x1, y1, x2, y2 = [int(v * s) for v in tr.box]; col = COLORS.get(tr.label, (200, 200, 200))
        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
        draw_text(fr, f"{CLASSES.get(tr.label, tr.label)} #{tr.id}", (x1, max(0, y1 - 26)), 20, col)
        pts = [(int(x * s), int(y * s)) for x, y in tr.history[-15:]]
        for i in range(1, len(pts)): cv2.line(fr, pts[i - 1], pts[i], col, 2)
    x0, y0 = (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2; canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "路口車流計數", (px + 24, 56), 38, C_TXT)
    draw_text(canvas, "偵測 → 追蹤 ID → 越線計數 · 不存影像", (px + 24, 106), 17, C_DIM)
    if err: draw_text(canvas, "錯誤:" + err[:40], (px + 24, 140), 18, (60, 60, 230))
    draw_text(canvas, f"{counter.total()}", (px + PANEL // 2, 230), 120, C_ACC, anchor="mm")
    draw_text(canvas, f"總計 · {rate_per_min:.0f} /分鐘", (px + PANEL // 2, 305), 22, C_DIM, anchor="mm")
    y = 350
    draw_text(canvas, "類別", (px + 24, y), 18, C_DIM); draw_text(canvas, "↓下行", (px + 220, y), 18, C_DIM); draw_text(canvas, "↑上行", (px + 320, y), 18, C_DIM)
    for k, name in CLASSES.items():
        c = counter.counts.get(k, {"up": 0, "down": 0}); y += 40
        draw_text(canvas, name, (px + 24, y), 24, COLORS[k]); draw_text(canvas, f"{c['down']}", (px + 220, y), 24, C_TXT); draw_text(canvas, f"{c['up']}", (px + 320, y), 24, C_TXT)
    draw_text(canvas, f"{fps:.0f} fps · 追蹤中 {len(tracks)}", (px + 24, H - 90), 18, C_DIM)
    draw_text(canvas, "w/s 移動計數線  r 歸零  q 離開", (px + 24, H - 60), 20, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0); ap.add_argument("--video", default="")
    ap.add_argument("--line", type=float, default=0.6, help="計數線位置(畫面高度比例)")
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; det = None
    try:
        det = HailoDetector("yolov8s.hef", conf_threshold=0.35)
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    cap = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not args.video: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.selftest:
        ok, f = cap.read(); d = det.infer(f) if (ok and det) else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if det else 'FAIL'} 物件={[(l, round(s, 2)) for l, s, _ in d][:5]}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    line_rel = args.line; tracker = CentroidTracker(); counter = None; events = deque(); fps = 0.0; t_last = time.time(); frames = 0
    logf = open(LOG_PATH, "a", newline="", encoding="utf-8"); logw = csv.writer(logf)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                frame = np.zeros((720, 1280, 3), np.uint8)
            if counter is None: counter = LineCounter(int(line_rel * frame.shape[0]))
            dets = []
            if det and not err:
                try: dets = [(l, s, b) for l, s, b in det.infer(frame) if l in CLASSES]
                except Exception as e: err = f"推論失敗:{e}"
            tracks = tracker.update(dets)
            now = time.time()
            for tid, lab, d in counter.update(tracks):
                events.append(now); logw.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), lab, d, tid])
            while events and now - events[0] > 60: events.popleft()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last); t_last = now
            canvas = compose(frame, tracks, line_rel, counter, len(events), fps, err)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 40:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k in (ord("w"), 82): line_rel = max(0.1, line_rel - 0.02); counter.line_y = int(line_rel * frame.shape[0])
            if k in (ord("s"), 84): line_rel = min(0.9, line_rel + 0.02); counter.line_y = int(line_rel * frame.shape[0])
            if k == ord("r"): counter = LineCounter(int(line_rel * frame.shape[0])); events.clear()
    finally:
        logf.close(); cap.release(); cv2.destroyAllWindows()
        if det: det.close()


if __name__ == "__main__":
    main()
