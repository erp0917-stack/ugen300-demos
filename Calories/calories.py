"""
calories.py —— 離線食物熱量估算:拍一張 → Qwen2-VL 估品項/份量/熱量 → 右側表格 + 今日累計。

按鍵:空白鍵 3 秒倒數拍照 / r 回到即時畫面 / c 清除今日累計 / q 離開
用法:python calories.py --source 0
      python calories.py --selftest                      (抓一幀、跑一次 VLM、印解析結果)
      python calories.py --image food.jpg --snapshot out.png   (分析指定圖片、截圖後離開)
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np

from nutrition import PROMPT, parse_items, totals, DailyLog
from ui_text import draw_text, badge

WIN = "UGen300 Calories"
CANVAS_W, CANVAS_H, PANEL_W = 1280, 720, 470
C_BG, C_PANEL, C_TXT, C_DIM = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150)
C_OK, C_WARN, C_ACC, C_LINE = (80, 220, 120), (60, 140, 255), (255, 170, 40), (70, 70, 80)
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "today.json")


class Analyzer:
    """在背景執行緒呼叫 VLM,主迴圈只讀狀態。"""
    def __init__(self):
        self.busy = False; self.items = None; self.raw = ""; self.error = None; self.elapsed = 0.0

    def start(self, frame):
        self.busy = True; self.items = None; self.error = None; self.raw = ""
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame):
        t = time.time()
        try:
            from vlm_client import ask_json_stream
            self.raw, _ = ask_json_stream(frame, PROMPT, temperature=0.15, max_tokens=220)
            self.items = parse_items(self.raw)
            if not self.items and self.raw.startswith("[vlm_client]"):
                self.error = self.raw
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        self.elapsed = time.time() - t; self.busy = False


def fit(frame, w, h):
    fh, fw = frame.shape[:2]; s = min(w / fw, h / fh)
    fr = cv2.resize(frame, (int(fw * s), int(fh * s)))
    canvas = np.full((h, w, 3), C_BG, np.uint8)
    y0 = (h - fr.shape[0]) // 2; x0 = (w - fr.shape[1]) // 2
    canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    return canvas


def compose(view, phase, countdown, an, log, spin):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, np.uint8)
    vw = CANVAS_W - PANEL_W
    canvas[:, :vw] = fit(view, vw, CANVAS_H)
    px = vw
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), C_PANEL, -1)
    badge(canvas)
    draw_text(canvas, "食物熱量估算", (px + 24, 56), 38, C_TXT)
    draw_text(canvas, "AI 估算 ±30%,不上雲、不用訂閱", (px + 24, 108), 18, C_DIM)
    y = 150
    if phase == "live":
        draw_text(canvas, "把餐點對準鏡頭", (px + 24, y), 28, C_ACC)
        draw_text(canvas, "按 空白鍵 拍照分析", (px + 24, y + 44), 24, C_TXT)
    elif phase == "countdown":
        draw_text(canvas, f"{countdown}", (px + PANEL_W // 2, 300), 160, C_ACC, anchor="mm")
        draw_text(canvas, "保持不動…", (px + PANEL_W // 2, 420), 28, C_DIM, anchor="mm")
    elif phase == "analyzing":
        dots = "·" * (spin % 4)
        draw_text(canvas, f"UGen300 分析中{dots}", (px + 24, y), 30, C_ACC)
        draw_text(canvas, "Qwen2-VL 看圖(約 6–10 秒)", (px + 24, y + 44), 20, C_DIM)
        cx, cy = px + PANEL_W // 2, 330
        cv2.ellipse(canvas, (cx, cy), (36, 36), (spin * 12) % 360, 0, 270, C_ACC, 5)
    elif phase == "result":
        if an.error:
            draw_text(canvas, "分析失敗", (px + 24, y), 28, C_WARN)
            for i, line in enumerate(str(an.error)[:200].split("\n")[:5]):
                draw_text(canvas, line[:34], (px + 24, y + 44 + i * 26), 18, C_DIM)
        elif not an.items:
            draw_text(canvas, "沒看到食物", (px + 24, y), 28, C_WARN)
            draw_text(canvas, "換個角度或靠近一點,按 r 重拍", (px + 24, y + 44), 20, C_DIM)
            for i, line in enumerate(an.raw[:160].split("\n")[:4]):
                draw_text(canvas, line[:34], (px + 24, y + 90 + i * 24), 16, C_DIM)
        else:
            tot = totals(an.items)
            draw_text(canvas, f"{tot['kcal']}", (px + 24, y - 10), 96, C_OK)
            draw_text(canvas, "kcal", (px + 24 + len(str(tot['kcal'])) * 54 + 8, y + 46), 30, C_DIM)
            draw_text(canvas, f"蛋白質 {tot['protein_g']} g  ·  {an.elapsed:.1f} 秒", (px + 24, y + 108), 20, C_DIM)
            yy = y + 150
            cv2.line(canvas, (px + 24, yy - 8), (CANVAS_W - 24, yy - 8), C_LINE, 1)
            for it in an.items[:7]:
                draw_text(canvas, it["item"][:12], (px + 24, yy), 24, C_TXT)
                draw_text(canvas, it["portion"][:10], (px + 220, yy + 4), 18, C_DIM)
                draw_text(canvas, f"{it['kcal']}", (CANVAS_W - 24, yy), 24, C_ACC, anchor="ra")
                yy += 40
        draw_text(canvas, "r 重拍  c 清除今日  q 離開", (px + 24, CANVAS_H - 100), 20, C_DIM)
    # 今日累計(底部)
    cv2.rectangle(canvas, (px, CANVAS_H - 64), (CANVAS_W, CANVAS_H), (30, 30, 36), -1)
    draw_text(canvas, f"今日累計  {log.kcal} kcal  ·  {log.meals} 餐", (px + 24, CANVAS_H - 48), 24, C_TXT)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--image", default="", help="分析指定圖片(不用鏡頭)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--snapshot", default="")
    args = ap.parse_args()

    log = DailyLog(LOG_PATH)
    an = Analyzer()
    cap = None; still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            print("讀不到圖片:", args.image); return
    else:
        cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if args.selftest:
        frame = still if still is not None else cap.read()[1]
        an.start(frame)
        while an.busy: time.sleep(0.2)
        print(f"[selftest] {an.elapsed:.1f}s | 錯誤={an.error} | 解析 {len(an.items or [])} 項 | raw={an.raw[:160]!r}")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if args.fullscreen: cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else: cv2.resizeWindow(WIN, CANVAS_W, CANVAS_H)

    phase = "live"; t_count = 0.0; frozen = None; spin = 0; snap_pending = bool(args.snapshot)
    if still is not None:  # 靜態圖模式:直接分析
        frozen = still; phase = "analyzing"; an.start(frozen)
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            else:
                frame = still
            if phase == "countdown":
                left = 3 - int(time.time() - t_count)
                if left <= 0:
                    frozen = frame.copy(); phase = "analyzing"; an.start(frozen)
                    try:
                        import winsound; winsound.Beep(1200, 80)
                    except Exception: pass
                countdown = max(1, left)
            else:
                countdown = 0
            if phase == "analyzing" and not an.busy:
                phase = "result"
                if an.items: log.add(an.items)
            view = frozen if (phase in ("analyzing", "result") and frozen is not None) else frame
            spin += 1
            canvas = compose(view, phase, countdown, an, log, spin)
            cv2.imshow(WIN, canvas)
            if snap_pending and phase == "result":
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"): break
            if k == ord(" ") and phase == "live": phase = "countdown"; t_count = time.time()
            if k == ord("r") and phase in ("result", "live") and cap is not None: phase = "live"; frozen = None
            if k == ord("c"): log.clear()
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        try:
            import vlm_client; vlm_client.close()   # 補丁版:只釋放模型層
        except Exception: pass


if __name__ == "__main__":
    main()
