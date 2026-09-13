"""
classroom.py —— 教室儀表板:一個視窗同時顯示 人數 / 舉手(含九宮格位置)/ 不專心(低頭、滑手機、趴睡、轉頭)/ 專注度走勢。

模型:yolov8m_pose(骨架 → 舉手、低頭、趴睡、轉頭)+ yolov11m(person 計數、cell phone),兩者交錯逐幀跑。
兩個模型 Hailo-8L 也都有,UGen200 同樣可用。不存影像、不辨識身分。

按鍵:1/2/3/4 開關 低頭/滑手機/趴睡/轉頭 規則 · 空白鍵 凍結 · s 存畫面 PNG · r 重設走勢 · q 離開
用法:python classroom.py [--source auto] [--fullscreen]
      python classroom.py --selftest / --snapshot out.png
"""
import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import attention as A
from hailo_detect import ObjectDetector
from hailo_pose import PoseEstimator
from tracker import CentroidTracker
from ui_text import draw_text, badge
from camera import open_camera

WIN = "UGen300 Classroom"
W, H, PANEL = 1280, 720, 440
HERE = os.path.dirname(os.path.abspath(__file__))
C_BG, C_PANEL, C_TXT, C_DIM, C_LINE = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150), (70, 70, 80)
C_OK, C_RAISE, C_DOWN, C_PHONE, C_SLEEP, C_TURN, C_ACC = (80, 220, 120), (255, 160, 40), (60, 200, 255), (60, 60, 230), (140, 140, 140), (200, 90, 200), (255, 170, 40)
STATE_COLOR = {"ok": C_OK, "head_down": C_DOWN, "phone": C_PHONE, "sleeping": C_SLEEP, "turned": C_TURN}
STATE_TEXT = {"ok": "", "head_down": "低頭", "phone": "滑手機", "sleeping": "趴睡", "turned": "轉頭"}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6)]


def draw_people(fr, s, tracks_kps, states, phones):
    for tid, kps in tracks_kps.items():
        st = states.get(tid); col = STATE_COLOR.get(st.state if st else "ok", C_OK)
        if st and st.raised and st.state == "ok": col = C_RAISE
        for a, b in SKELETON:
            if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                cv2.line(fr, (int(kps[a][0] * s), int(kps[a][1] * s)), (int(kps[b][0] * s), int(kps[b][1] * s)), col, 2)
        if st and st.box:
            x1, y1, x2, y2 = [int(v * s) for v in st.box]
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
            tag = ("舉手 " if st.raised else "") + STATE_TEXT.get(st.state, "")
            if tag: draw_text(fr, tag.strip(), (x1, max(0, y1 - 28)), 22, col)
    for x1, y1, x2, y2 in phones:
        cv2.rectangle(fr, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), C_PHONE, 2)


def draw_grid_map(canvas, x, y, grid, cell=42):
    for r in range(3):
        for c in range(3):
            n = grid[r][c]; x0, y0 = x + c * cell, y + r * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 3, y0 + cell - 3), (60, 60, 68) if n == 0 else (120, 80, 30), -1)
            if n: draw_text(canvas, str(n), (x0 + cell // 2 - 1, y0 + cell // 2 - 1), 24, C_RAISE, anchor="mm", shadow=False)
    for r in range(3): draw_text(canvas, A.GRID_ROW[r], (x + 3 * cell + 4, y + r * cell + 10), 15, C_DIM)
    for c in range(3): draw_text(canvas, A.GRID_COL[c], (x + c * cell + cell // 2 - 1, y + 3 * cell + 2), 15, C_DIM, anchor="ma")


def compose(frame, s, tracks_kps, states, phones, count, summary, enabled, history, fps, err=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    h, w = frame.shape[:2]; sc = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * sc), int(h * sc)))
    draw_people(fr, sc, tracks_kps, states, phones)
    x0, y0 = (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2; canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "教室儀表板", (px + 24, 44), 36, C_TXT)
    draw_text(canvas, "YOLO 姿態 + 偵測 · 不存影像、不辨識身分", (px + 24, 92), 15, C_DIM)
    if err: draw_text(canvas, "錯誤:" + err[:36], (px + 24, 114), 16, C_PHONE)
    # 人數
    draw_text(canvas, f"{count}", (px + 24, 118), 96, C_OK if count else C_TXT)
    draw_text(canvas, "人", (px + 24 + len(str(count)) * 54 + 6, 172), 30, C_DIM)
    # 舉手 + 九宮格
    draw_text(canvas, f"舉手 {summary['raised']}", (px + 24, 250), 30, C_RAISE)
    draw_grid_map(canvas, px + 24, 296, summary["grid"])
    # 不專心
    bx = px + 200
    draw_text(canvas, f"不專心 {summary['inattentive']}", (bx, 250), 30, C_PHONE if summary["inattentive"] else C_TXT)
    yy = 296
    for i, k in enumerate(A.RULES, 1):
        on = enabled.get(k, True); col = STATE_COLOR[k] if on else (80, 80, 88)
        draw_text(canvas, f"{i} {A.RULE_NAMES[k]}", (bx, yy), 21, col)
        draw_text(canvas, f"{summary['by'][k]}" if on else "關", (bx + 200, yy), 21, col, anchor="ra")
        yy += 32
    # 專注度 + 走勢
    att = 100 if count == 0 else int(round(100 * max(0, count - summary["inattentive"]) / count))
    draw_text(canvas, f"專注度 {att}%", (px + 24, 470), 30, C_OK if att >= 80 else C_ACC if att >= 60 else C_PHONE)
    gx0, gy0, gw, gh = px + 24, 516, PANEL - 48, 90
    cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (50, 50, 58), -1)
    if len(history) >= 2:
        t_now = history[-1][0]
        pts = [(int(gx0 + gw * (1 - (t_now - t) / 60.0)), int(gy0 + gh - gh * v / 100)) for t, v in history if t_now - t <= 60]
        for i in range(1, len(pts)): cv2.line(canvas, pts[i - 1], pts[i], C_OK, 2)
    draw_text(canvas, "最近 60 秒專注度", (gx0, gy0 + gh + 4), 15, C_DIM)
    draw_text(canvas, f"{fps:.0f} fps", (px + 24, H - 84), 17, C_DIM)
    draw_text(canvas, "1-4 開關規則  空白鍵 凍結  s 存圖  q 離開", (px + 24, H - 56), 18, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", help="auto=外接優先,否則內建;或指定編號 0/1")
    ap.add_argument("--pose-hef", default="yolov8m_pose.hef"); ap.add_argument("--det-hef", default="yolov11m.hef")
    ap.add_argument("--conf", type=float, default=0.3); ap.add_argument("--flip", action="store_true", help="鏡像(自拍角度用;教室朝學生時不用)")
    ap.add_argument("--image", default="", help="用靜態照片代替鏡頭(教室合照測試用)")
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; pose = det = None
    try:
        pose = PoseEstimator(args.pose_hef, conf_threshold=args.conf); det = ObjectDetector(args.det_hef, conf_threshold=args.conf)
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    still = None
    if args.image:
        still = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if still is None: print("讀不到圖片:", args.image); return
    cap = None if still is not None else open_camera(args.source)[0]
    if args.selftest:
        ok, f = (True, still) if still is not None else cap.read()
        people = pose.infer_multi(f) if (ok and pose) else []; dets = det.infer(f) if (ok and det) else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if (pose and det) else 'FAIL'} 骨架={len(people)} person={sum(l == 'person' for l, _, _ in dets)} phone={sum(l == 'cell phone' for l, _, _ in dets)}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    enabled = {k: True for k in A.RULES}; tracker = CentroidTracker(max_missed=15, iou_thresh=0.2, dist_thresh=120)
    states = {}; people = []; dets = []; phones = []; count_hist = deque(maxlen=5); history = deque()
    frozen = None; fps = 0.0; t_last = time.time(); frames = 0; last_summary = A.summarize({}, W, H); count = 0; tracks_kps = {}
    try:
        while True:
            if still is not None: ok, frame = True, still.copy()
            else: ok, frame = cap.read(); frame = frame if ok else np.zeros((720, 1280, 3), np.uint8)
            if args.flip and ok: frame = cv2.flip(frame, 1)
            if frozen is not None: frame = frozen
            elif pose and det and not err:
                try:
                    if frames % 2 == 0: people = pose.infer_multi(frame) or []
                    else: dets = det.infer(frame)
                except Exception as e:  # noqa: BLE001
                    err = f"推論失敗:{e}"
                phones = [b for l, _, b in dets if l == "cell phone"]
                n_person = sum(1 for l, _, _ in dets if l == "person")
                count_hist.append(max(n_person, len(people))); count = int(np.median(count_hist))
                # 追蹤:用骨架推出的框給 ID,再把 ID 對回關鍵點
                boxed = [(A.bbox_from_kps(k), k) for k in people]; boxed = [(b, k) for b, k in boxed if b]
                tracks = tracker.update([("person", 1.0, b) for b, _ in boxed])
                box2kps = {b: k for b, k in boxed}; tracks_kps = {}; now = time.time()
                for tr in tracks:
                    if tr.missed == 0 and tr.box in box2kps:
                        kps = box2kps[tr.box]; tracks_kps[tr.id] = kps
                        states.setdefault(tr.id, A.PersonState()).update(kps, phones, enabled, now, box=tr.box)
                alive = {tr.id for tr in tracks}
                for tid in list(states):
                    if tid not in alive: del states[tid]
                last_summary = A.summarize({t: s for t, s in states.items() if t in tracks_kps}, frame.shape[1], frame.shape[0])
                att = 100 if count == 0 else 100 * max(0, count - last_summary["inattentive"]) / count
                history.append((now, att))
                while history and now - history[0][0] > 60: history.popleft()
            now = time.time(); fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last); t_last = now
            canvas = compose(frame, None, tracks_kps, states, phones, count, last_summary, enabled, history, fps, err)
            if frozen is not None: draw_text(canvas, "已凍結 · 空白鍵恢復", (30, 24), 28, C_ACC)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 40:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k in (ord("1"), ord("2"), ord("3"), ord("4")): key = A.RULES[k - ord("1")]; enabled[key] = not enabled[key]
            if k == ord(" "): frozen = None if frozen is not None else frame.copy()
            if k == ord("r"): history.clear(); count_hist.clear()
            if k == ord("s"):
                p = os.path.join(HERE, f"classroom_{datetime.now():%H%M%S}.png"); cv2.imwrite(p, canvas); print("[存圖]", p)
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
