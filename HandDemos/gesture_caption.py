"""
gesture_caption.py —— 手勢字幕:鏡頭前比手勢,螢幕即時顯示大字(數字 1–5、OK、讚、比心、搖滾、拳頭…)。

按鍵:q 離開 / h 顯示手勢表
用法:python gesture_caption.py --source 0 [--fullscreen]
      python gesture_caption.py --selftest
      python gesture_caption.py --snapshot out.png
"""
import argparse
import time

import cv2
import numpy as np

from hand_tracker import HandTracker, draw_hand
from gestures import classify, GestureSmoother
from ui_text import draw_text, badge

WIN = "UGen300 Gesture Caption"
CANVAS_W, CANVAS_H, PANEL_W = 1280, 720, 400
C_BG, C_PANEL, C_TXT, C_DIM, C_ACC, C_OK = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150), (255, 170, 40), (80, 220, 120)
CHEAT = ["1–5:伸出手指數量", "手掌張開:5 / 停", "拳頭:0", "OK:拇指食指圈", "讚 / 倒讚", "比心:拇指食指交叉", "搖滾:食指+小指", "小指:打勾勾"]


def compose(frame, hands, labels, fps, err=None):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, np.uint8)
    vw = CANVAS_W - PANEL_W
    h, w = frame.shape[:2]; s = min(vw / w, CANVAS_H / h)
    fr = cv2.resize(frame, (int(w * s), int(h * s))); y0 = (CANVAS_H - fr.shape[0]) // 2; x0 = (vw - fr.shape[1]) // 2
    canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    px = vw
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "手勢字幕", (px + 24, 56), 40, C_TXT)
    draw_text(canvas, "21 點手部關鍵點 · 規則判斷 · 不需訓練", (px + 24, 108), 17, C_DIM)
    if err:
        draw_text(canvas, "發生錯誤", (px + 24, 160), 28, (60, 140, 255))
        for i, l in enumerate(str(err).split("\n")[:6]): draw_text(canvas, l[:30], (px + 24, 200 + i * 26), 18, C_DIM)
        return canvas
    if not hands:
        draw_text(canvas, "把手舉到鏡頭前", (px + PANEL_W // 2, 330), 30, C_DIM, anchor="mm")
    for i, (hand, (label, text)) in enumerate(zip(hands, labels)):
        y = 170 + i * 250
        draw_text(canvas, f"{'右' if hand.side == 'R' else '左'}手", (px + 24, y), 22, C_DIM)
        draw_text(canvas, label if label != "?" else "…", (px + PANEL_W // 2, y + 90), 120 if len(label) <= 2 else 64, C_OK if label != "?" else C_DIM, anchor="mm")
        draw_text(canvas, text, (px + PANEL_W // 2, y + 175), 22, C_ACC, anchor="mm")
    draw_text(canvas, f"{fps:.0f} fps", (px + 24, CANVAS_H - 90), 18, C_DIM)
    draw_text(canvas, "h 手勢表  r 重置  q 離開", (px + 24, CANVAS_H - 60), 20, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0); ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; tr = None
    try:
        tr = HandTracker()
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{type(e).__name__}\n{e}\n請確認 UGen300 已插上。"
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.selftest:
        ok, f = cap.read(); hs = tr.update(cv2.flip(f, 1)) if (ok and tr) else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if tr else 'FAIL'} 手={len(hs)}"); cap.release(); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, CANVAS_W, CANVAS_H)
    smoothers = {"R": GestureSmoother(4), "L": GestureSmoother(4)}
    show_cheat = False; t_last = time.time(); fps = 0.0; frames = 0
    try:
        while True:
            ok, frame = cap.read()
            frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            hands, labels = [], []
            if tr and not err:
                try:
                    hands = tr.update(frame)
                except Exception as e:  # noqa: BLE001
                    err = f"推論失敗:{e}"
                for hd in hands:
                    label, text = classify(hd.pts)
                    stable = smoothers[hd.side].update(label)
                    labels.append((stable or "?", text if stable == label else "辨識中…"))
                    draw_hand(frame, hd)
            now = time.time(); fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last); t_last = now
            canvas = compose(frame, hands, labels, fps, err)
            if show_cheat:
                for i, l in enumerate(CHEAT): draw_text(canvas, l, (30, 30 + i * 34), 24, C_TXT)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 30:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("h"): show_cheat = not show_cheat
            if k == ord("r"): smoothers = {"R": GestureSmoother(4), "L": GestureSmoother(4)}
    finally:
        cap.release(); cv2.destroyAllWindows()
        if tr: tr.close()


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
