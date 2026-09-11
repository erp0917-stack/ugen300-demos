"""
air_band.py —— 空氣樂團:畫面分三區(左 鼓 / 中 鋼琴 / 右 吉他),多人一起揮手就能演奏。

- 用姿態的手腕(不需手部細節,延遲最低),支援多人。
- 鼓:手腕快速往下揮 → 依高度 大鼓/小鼓/hi-hat。
- 鋼琴:手腕進入某一鍵(8 鍵,依 x)就發那個音;高度決定八度。
- 吉他:手腕往下刷 → 依高度 4 個和弦。
- 音源:numpy 合成,pygame.mixer 多音混音(pip install pygame)。
按鍵:q 離開
用法:python air_band.py --source 0 [--fullscreen] / --selftest / --snapshot out.png
"""
import argparse
import time

import cv2
import numpy as np

from hailo_pose import PoseEstimator
from ui_text import draw_text, badge

WIN = "UGen300 Air Band"
W, H = 1280, 720
L_WRIST, R_WRIST = 9, 10
SR = 22050
NOTE_NAMES = ["Do", "Re", "Mi", "Fa", "Sol", "La", "Si", "Do'"]
NOTE_FREQ = [261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.3]
CHORDS = [("C", [261.6, 329.6, 392.0]), ("G", [196.0, 246.9, 392.0]), ("Am", [220.0, 261.6, 329.6]), ("F", [174.6, 220.0, 349.2])]


# ---------- 合成音色 ----------
def _env(n, a=0.005, d=0.6):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / d)


def tone_piano(f, dur=1.2):
    n = int(SR * dur); t = np.arange(n) / SR
    y = sum((0.6 ** k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4)) * _env(n, 0.004, 0.45)
    return y


def tone_pluck(f, dur=1.0):
    """Karplus-Strong 撥弦。"""
    n = int(SR * dur); p = max(2, int(SR / f)); buf = np.random.uniform(-1, 1, p); out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]; buf[i % p] = 0.5 * (buf[i % p] + buf[(i + 1) % p]) * 0.996
    return out


def tone_drum(kind, dur=0.4):
    n = int(SR * dur); t = np.arange(n) / SR
    if kind == "kick":
        return np.sin(2 * np.pi * (60 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 9)
    if kind == "snare":
        return (np.random.uniform(-1, 1, n) * 0.6 + np.sin(2 * np.pi * 180 * t) * 0.4) * np.exp(-t * 14)
    return np.random.uniform(-1, 1, n) * np.exp(-t * 40) * 0.6  # hihat


class Mixer:
    def __init__(self):
        import pygame
        pygame.mixer.pre_init(SR, -16, 1, 512); pygame.mixer.init(); pygame.mixer.set_num_channels(32)
        self.pg = pygame; self.cache = {}

    def _snd(self, key, arr):
        if key not in self.cache:
            a = np.clip(arr / max(1e-6, np.abs(arr).max()) * 0.8 * 32767, -32767, 32767).astype(np.int16)
            ch = self.pg.mixer.get_init()[2]          # 實際初始化的聲道數(Windows 常強制 2)
            if ch > 1:
                a = np.repeat(a[:, None], ch, axis=1)
            self.cache[key] = self.pg.sndarray.make_sound(np.ascontiguousarray(a))
        return self.cache[key]

    def piano(self, i): self._snd(("p", i), tone_piano(NOTE_FREQ[i])).play()
    def piano_hi(self, i): self._snd(("ph", i), tone_piano(NOTE_FREQ[i] * 2)).play()
    def drum(self, kind): self._snd(("d", kind), tone_drum(kind)).play()
    def chord(self, i):
        for f in CHORDS[i][1]: self._snd(("g", f), tone_pluck(f)).play()


# ---------- 純邏輯:區域與觸發 ----------
class BandLogic:
    """輸入每隻手(id, x, y, t),回傳觸發事件。id 用 (人序, 左右)。"""
    def __init__(self, w=W, h=H):
        self.w, self.h = w, h; self.prev = {}; self.last_fire = {}; self.last_key = {}

    def zone(self, x):
        return "drum" if x < self.w / 3 else ("piano" if x < 2 * self.w / 3 else "guitar")

    def update(self, hands, now):
        events = []
        for hid, x, y in hands:
            px, py, pt = self.prev.get(hid, (x, y, now))
            vy = (y - py) / max(1e-3, now - pt)          # 往下為正
            z = self.zone(x)
            if z == "drum":
                if vy > 900 and now - self.last_fire.get(hid, 0) > 0.18:
                    kind = "hihat" if y < self.h * 0.4 else ("snare" if y < self.h * 0.7 else "kick")
                    events.append(("drum", kind)); self.last_fire[hid] = now
            elif z == "piano":
                rel = (x - self.w / 3) / (self.w / 3); key = int(np.clip(rel * 8, 0, 7))
                if self.last_key.get(hid) != key and now - self.last_fire.get(hid, 0) > 0.08:
                    events.append(("piano_hi" if y < self.h * 0.45 else "piano", key)); self.last_fire[hid] = now
                self.last_key[hid] = key
            else:
                if vy > 700 and now - self.last_fire.get(hid, 0) > 0.3:
                    ci = int(np.clip((y / self.h) * 4, 0, 3)); events.append(("chord", ci)); self.last_fire[hid] = now
            if z != "piano": self.last_key[hid] = None
            self.prev[hid] = (x, y, now)
        return events


def draw_stage(canvas, flash):
    third = W // 3
    for i, (name, col) in enumerate([("鼓", (60, 60, 200)), ("鋼琴", (60, 160, 60)), ("吉他", (200, 120, 40))]):
        cv2.rectangle(canvas, (i * third, 0), ((i + 1) * third, H), col, 3)
        draw_text(canvas, name, (i * third + third // 2, 40), 40, (255, 255, 255), anchor="mm")
    for k in range(8):  # 鋼琴鍵
        x0 = third + k * third // 8
        cv2.line(canvas, (x0, 80), (x0, H - 60), (90, 90, 90), 1)
        draw_text(canvas, NOTE_NAMES[k], (x0 + third // 16, H - 40), 20, (200, 200, 200), anchor="mm")
    for i in range(3):  # 鼓的三層
        y = int(H * (0.4 if i == 0 else 0.7 if i == 1 else 1.0))
        draw_text(canvas, ["hi-hat", "小鼓", "大鼓"][i], (third // 2, y - 30), 22, (200, 200, 255), anchor="mm")
        cv2.line(canvas, (0, y), (third, y), (90, 90, 120), 1)
    for i, (nm, _) in enumerate(CHORDS):
        y = int(H * (i + 0.5) / 4); draw_text(canvas, nm, (2 * third + third // 2, y), 44, (255, 200, 120), anchor="mm")
    for (x, y, t0), now in flash:
        a = max(0, 1 - (now - t0) / 0.3)
        if a > 0: cv2.circle(canvas, (int(x), int(y)), int(20 + 60 * (1 - a)), (255, 255, 255), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0); ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; pose = None; mixer = None
    try:
        pose = PoseEstimator("yolov8s_pose.hef", conf_threshold=0.3)
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    try:
        mixer = Mixer()
    except Exception as e:  # noqa: BLE001
        err = (err + "\n" if err else "") + f"音效初始化失敗:{e}(pip install pygame)"
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.selftest:
        ok, f = cap.read(); ppl = pose.infer_multi(cv2.flip(f, 1)) if (ok and pose) else []
        if mixer: mixer.piano(0); mixer.drum("kick"); mixer.chord(0); time.sleep(0.6)
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if pose else 'FAIL'} 音效={'OK' if mixer else 'FAIL'} 人={len(ppl)}"); cap.release(); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    logic = BandLogic(); flashes = []; frames = 0; last_event = ""
    try:
        while True:
            ok, frame = cap.read()
            frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            now = time.time()
            canvas = (cv2.resize(frame, (W, H)) * 0.6).astype(np.uint8)
            hands = []
            if pose and not err:
                try:
                    people = pose.infer_multi(frame) or []
                except Exception as e:  # noqa: BLE001
                    err = f"推論失敗:{e}"; people = []
                sx, sy = W / frame.shape[1], H / frame.shape[0]
                for pi, kps in enumerate(people):
                    for side, idx in (("L", L_WRIST), ("R", R_WRIST)):
                        x, y, c = kps[idx]
                        if c >= 0.35:
                            hands.append(((pi, side), x * sx, y * sy))
                            cv2.circle(canvas, (int(x * sx), int(y * sy)), 14, (255, 255, 255), -1)
                for ev in logic.update(hands, now):
                    kind, val = ev
                    if mixer:
                        {"drum": mixer.drum, "piano": mixer.piano, "piano_hi": mixer.piano_hi, "chord": mixer.chord}[kind](val)
                    last_event = f"{kind}:{val if kind != 'chord' else CHORDS[val][0]}"
                    hx = [h for h in hands];
                    if hx: flashes.append((hx[0][1], hx[0][2], now))
            flashes = [f for f in flashes if now - f[2] < 0.3]
            draw_stage(canvas, [(f, now) for f in flashes])
            badge(canvas)
            draw_text(canvas, "空氣樂團 · 多人可同時演奏 · q 離開", (W // 2, H - 14), 20, (200, 200, 200), anchor="mm")
            if last_event: draw_text(canvas, last_event, (20, H - 40), 20, (180, 180, 180))
            if err: draw_text(canvas, err[:70], (20, 60), 22, (60, 140, 255))
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 30:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            if (cv2.waitKey(1) & 0xFF) == ord("q"): break
    finally:
        cap.release(); cv2.destroyAllWindows()
        if pose: pose.close()


if __name__ == "__main__":
    main()
