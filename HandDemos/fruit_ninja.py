"""
fruit_ninja.py —— 手勢切水果:食指指尖就是刀,水果從下方拋起,劃過就切;炸彈扣分;30 秒一局。

按鍵:空白鍵 開始/重玩 / q 離開
用法:python fruit_ninja.py --source 0 [--fullscreen] [--seconds 30]
      python fruit_ninja.py --selftest / --snapshot out.png
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from hand_tracker import HandTracker
from ui_text import draw_text, badge

WIN = "UGen300 Fruit Ninja"
W, H = 1280, 720
G = 1400.0  # 重力 px/s^2
FRUITS = [("蘋果", (60, 60, 230), 34), ("西瓜", (80, 200, 80), 46), ("橘子", (40, 150, 255), 32), ("葡萄", (200, 80, 160), 28), ("檸檬", (60, 230, 230), 30)]
BEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fruit_best.json")


class Fruit:
    def __init__(self, bomb=False):
        self.x = random.uniform(200, W - 200); self.y = H + 40
        self.vx = random.uniform(-160, 160); self.vy = -random.uniform(900, 1150)
        self.bomb = bomb
        self.name, self.color, self.r = ("炸彈", (30, 30, 30), 34) if bomb else random.choice(FRUITS)
        self.alive = True; self.cut_t = None; self.halves = []

    def step(self, dt):
        self.vy += G * dt; self.x += self.vx * dt; self.y += self.vy * dt
        for hv in self.halves:
            hv[3] += G * dt; hv[0] += hv[2] * dt; hv[1] += hv[3] * dt

    def cut(self, now, dirx):
        self.alive = False; self.cut_t = now
        self.halves = [[self.x, self.y, self.vx - 150 * dirx, self.vy * 0.5], [self.x, self.y, self.vx + 150 * dirx, self.vy * 0.5]]

    def gone(self, now):
        return (self.y > H + 80) or (self.cut_t is not None and now - self.cut_t > 0.8)

    def draw(self, canvas, now):
        if self.alive:
            cv2.circle(canvas, (int(self.x), int(self.y)), self.r, self.color, -1)
            cv2.circle(canvas, (int(self.x), int(self.y)), self.r, (255, 255, 255), 2)
            if self.bomb:
                cv2.line(canvas, (int(self.x), int(self.y - self.r)), (int(self.x + 10), int(self.y - self.r - 14)), (0, 200, 255), 3)
            draw_text(canvas, self.name, (int(self.x), int(self.y - self.r - 26)), 20, (255, 255, 255), anchor="mm")
        else:
            for hx, hy, _, _ in self.halves:
                cv2.circle(canvas, (int(hx), int(hy)), self.r // 2 + 4, self.color, -1)
            if self.cut_t and now - self.cut_t < 0.5:
                draw_text(canvas, "-5" if self.bomb else "+1", (int(self.x), int(self.y - 50)), 40, (60, 60, 255) if self.bomb else (80, 220, 120), anchor="mm")


def seg_circle_hit(p0, p1, c, r):
    """線段 p0-p1 是否碰到圓 (c, r)。"""
    p0, p1, c = np.array(p0, float), np.array(p1, float), np.array(c, float)
    d = p1 - p0; L2 = float(d @ d)
    t = 0.0 if L2 == 0 else float(np.clip(((c - p0) @ d) / L2, 0, 1))
    return float(np.linalg.norm(p0 + t * d - c)) <= r


def load_best():
    try:
        return json.load(open(BEST_PATH))["best"]
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0); ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; tr = None
    try:
        tr = HandTracker()
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.selftest:
        ok, f = cap.read(); hs = tr.update(cv2.flip(f, 1)) if (ok and tr) else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if tr else 'FAIL'} 手={len(hs)}"); cap.release(); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)

    best = load_best(); state = "ready"; score = 0; t_start = 0; fruits = []; trails = {}; last_spawn = 0
    t_prev = time.time(); frames = 0
    try:
        while True:
            ok, frame = cap.read()
            frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            now = time.time(); dt = min(0.1, now - t_prev); t_prev = now
            canvas = cv2.resize(frame, (W, H)); canvas = (canvas * 0.55).astype(np.uint8)
            tips = []
            if tr and not err:
                try:
                    for hd in tr.update(frame):
                        p = hd.pts[8] * np.array([W / frame.shape[1], H / frame.shape[0]])
                        tips.append((hd.side, (float(p[0]), float(p[1]))))
                except Exception as e:  # noqa: BLE001
                    err = f"推論失敗:{e}"
            # 刀痕軌跡
            for side, p in tips:
                trails.setdefault(side, []).append((p, now))
            for side in list(trails):
                trails[side] = [(p, t) for p, t in trails[side] if now - t < 0.35]
                pts = [p for p, _ in trails[side]]
                for i in range(1, len(pts)):
                    cv2.line(canvas, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), (255, 255, 255), max(1, 8 - (len(pts) - i)))
            if state == "play":
                remain = args.seconds - (now - t_start)
                if remain <= 0:
                    state = "over"; best = max(best, score)
                    try: json.dump({"best": best}, open(BEST_PATH, "w"))
                    except OSError: pass
                if now - last_spawn > random.uniform(0.5, 0.9):
                    fruits.append(Fruit(bomb=random.random() < 0.18)); last_spawn = now
                for fr in fruits:
                    fr.step(dt)
                    if fr.alive:
                        for side in trails:
                            pts = [p for p, _ in trails[side]]
                            if len(pts) >= 2 and seg_circle_hit(pts[-2], pts[-1], (fr.x, fr.y), fr.r + 6):
                                speed = np.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) / max(dt, 1e-3)
                                if speed > 250:
                                    fr.cut(now, 1 if pts[-1][0] >= pts[-2][0] else -1); score += -5 if fr.bomb else 1
                                    try:
                                        import winsound; winsound.Beep(200 if fr.bomb else 1000, 40)
                                    except Exception: pass
                                    break
                fruits = [f for f in fruits if not f.gone(now)]
                for fr in fruits: fr.draw(canvas, now)
                draw_text(canvas, f"{int(remain) + 1}", (W // 2, 60), 64, (255, 170, 40), anchor="mm")
            elif state == "ready":
                draw_text(canvas, "手勢切水果", (W // 2, H // 2 - 80), 80, (255, 255, 255), anchor="mm")
                draw_text(canvas, "食指就是刀,快速劃過水果;炸彈別碰", (W // 2, H // 2 + 10), 30, (220, 220, 220), anchor="mm")
                draw_text(canvas, "按 空白鍵 開始", (W // 2, H // 2 + 80), 36, (255, 170, 40), anchor="mm")
            else:
                draw_text(canvas, "時間到!", (W // 2, H // 2 - 80), 80, (255, 255, 255), anchor="mm")
                draw_text(canvas, f"得分 {score}   最佳 {best}", (W // 2, H // 2 + 10), 44, (80, 220, 120), anchor="mm")
                draw_text(canvas, "空白鍵 再玩一次", (W // 2, H // 2 + 80), 30, (255, 170, 40), anchor="mm")
            draw_text(canvas, f"分數 {score}", (30, 24), 40, (255, 255, 255))
            draw_text(canvas, f"最佳 {best}", (30, 74), 24, (200, 200, 200))
            badge(canvas)
            if err: draw_text(canvas, err[:60], (30, H - 50), 22, (60, 140, 255))
            if not tips and state == "play": draw_text(canvas, "看不到手,舉高一點", (W // 2, H - 40), 26, (200, 200, 200), anchor="mm")
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames == 5: state = "play"; t_start = now; last_spawn = 0
            if args.snapshot and frames >= 60:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord(" ") and state in ("ready", "over"):
                state = "play"; score = 0; fruits = []; t_start = now; last_spawn = 0
    finally:
        cap.release(); cv2.destroyAllWindows()
        if tr: tr.close()


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
