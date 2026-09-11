"""
head_count.py —— 即時數人頭:YOLO 偵測「人」→ 畫框編號 → 右側大數字顯示目前人數、峰值、最近 60 秒曲線。

按鍵:r 重設峰值 / q 離開
用法:python head_count.py --source 0 [--fullscreen] [--conf 0.3] [--hef yolov8m.hef]
按 空白鍵 可凍結畫面與人數(掃教室掃到定點時用),再按一次恢復
      python head_count.py --selftest / --snapshot out.png
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from hailo_detect import ObjectDetector
from ui_text import draw_text, badge

WIN = "UGen300 Head Count"
W, H, PANEL = 1280, 720, 420
C_BG, C_PANEL, C_TXT, C_DIM, C_ACC, C_OK = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150), (255, 170, 40), (80, 220, 120)


def compose(frame, people, count, peak, history, fps, err=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    h, w = frame.shape[:2]; s = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * s), int(h * s)))
    for i, (sc, (x1, y1, x2, y2)) in enumerate(people, 1):
        cv2.rectangle(fr, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), C_OK, 3)
        cv2.rectangle(fr, (int(x1 * s), int(y1 * s) - 30), (int(x1 * s) + 44, int(y1 * s)), C_OK, -1)
        draw_text(fr, f"{i}", (int(x1 * s) + 22, int(y1 * s) - 16), 22, (20, 20, 20), anchor="mm", shadow=False)
    x0, y0 = (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2; canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "即時人數", (px + 24, 56), 40, C_TXT)
    draw_text(canvas, "YOLOv8m 偵測 · 不存影像、不辨識身分", (px + 24, 108), 17, C_DIM)
    if err: draw_text(canvas, "錯誤:" + err[:40], (px + 24, 140), 18, (60, 60, 230))
    draw_text(canvas, f"{count}", (px + PANEL // 2, 290), 220, C_OK if count else C_TXT, anchor="mm")
    draw_text(canvas, "人", (px + PANEL // 2 + 120, 340), 40, C_DIM, anchor="mm")
    draw_text(canvas, f"峰值 {peak}", (px + 24, 420), 26, C_ACC)
    # 最近 60 秒曲線
    gx0, gy0, gw, gh = px + 24, 470, PANEL - 48, 120
    cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (50, 50, 58), -1)
    if len(history) >= 2:
        mx = max(max(v for _, v in history), 1)
        t_now = history[-1][0]
        pts = [(int(gx0 + gw * (1 - (t_now - t) / 60.0)), int(gy0 + gh - gh * v / mx)) for t, v in history if t_now - t <= 60]
        for i in range(1, len(pts)): cv2.line(canvas, pts[i - 1], pts[i], C_ACC, 2)
    draw_text(canvas, "最近 60 秒", (gx0, gy0 + gh + 6), 16, C_DIM)
    draw_text(canvas, f"{fps:.0f} fps", (px + 24, H - 90), 18, C_DIM)
    draw_text(canvas, "空白鍵 凍結  r 重設峰值  q 離開", (px + 24, H - 60), 20, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0); ap.add_argument("--conf", type=float, default=0.3); ap.add_argument("--hef", default="yolov8m.hef")
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; det = None
    try:
        det = ObjectDetector(args.hef, conf_threshold=args.conf)
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.selftest:
        ok, f = cap.read(); d = [(s, b) for l, s, b in det.infer(f) if l == "person"] if (ok and det) else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if det else 'FAIL'} 人數={len(d)}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    peak = 0; history = deque(); fps = 0.0; t_last = time.time(); frames = 0; smooth = deque(maxlen=3)
    frozen = None; frozen_people = []; frozen_count = 0
    try:
        while True:
            ok, frame = cap.read(); frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            people = []
            if frozen is not None:
                frame, people, count = frozen, frozen_people, frozen_count
            else:
                if det and not err:
                    try: people = [(s, b) for l, s, b in det.infer(frame) if l == "person"]
                    except Exception as e: err = f"推論失敗:{e}"
                smooth.append(len(people)); count = int(round(np.median(smooth)))   # 3 幀中位數,避免閃爍
            peak = max(peak, count); now = time.time(); history.append((now, count))
            while history and now - history[0][0] > 60: history.popleft()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last); t_last = now
            canvas = compose(frame, people, count, peak, history, fps, err)
            if frozen is not None: draw_text(canvas, "已凍結 · 空白鍵恢復", (30, 24), 28, C_ACC)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 40:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("r"): peak = count
            if k == ord(" "):
                if frozen is None: frozen, frozen_people, frozen_count = frame.copy(), list(people), count
                else: frozen = None
    finally:
        cap.release(); cv2.destroyAllWindows()
        if det: det.close()


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
