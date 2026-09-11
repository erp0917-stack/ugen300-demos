"""
wakeup.py —— 早八起床驗證:鬧鐘響了,做 N 下深蹲才能關。

流程:時鐘 → 鬧鈴 + 指示 → 偵測到人開始計次 → 達標 → 停鈴、勝利音效、「驗證通過,早安!」
按鍵:空白鍵 跳過時鐘 / r 重來 / q 離開
用法:python wakeup.py --reps 5 --source 0
      python wakeup.py --selftest            (只載模型、抓一幀、推論一次,不開視窗)
      python wakeup.py --snapshot out.png    (跑 40 幀後存一張畫面並離開,檢查版面用)
"""
import argparse
import threading
import time

import cv2
import numpy as np

from hailo_pose import PoseEstimator
from squat_counter import SquatCounter
from ui_text import draw_text, badge
from camera import open_camera
from victory_sound import play_victory

WIN = "UGen300 WakeUp"
CANVAS_W, CANVAS_H = 1280, 720
PANEL_W = 420
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
C_BG, C_PANEL, C_TXT, C_DIM = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150)
C_OK, C_WARN, C_ACC = (80, 220, 120), (60, 140, 255), (255, 170, 40)


class Alarm:
    """鬧鈴:獨立執行緒循環嗶嗶,stop() 立即安靜。"""
    def __init__(self):
        self._on = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()

    def _loop(self):
        try:
            import winsound
        except ImportError:
            return
        while True:
            self._on.wait()
            for f in (880, 1175):
                if not self._on.is_set(): break
                winsound.Beep(f, 180)
            time.sleep(0.25)

    def start(self): self._on.set()
    def stop(self): self._on.clear()
    @property
    def ringing(self): return self._on.is_set()


def draw_skeleton(frame, people):
    for kps in people:
        for a, b in SKELETON:
            if kps[a][2] > 0.35 and kps[b][2] > 0.35:
                cv2.line(frame, (int(kps[a][0]), int(kps[a][1])), (int(kps[b][0]), int(kps[b][1])), C_OK, 3)
        for x, y, c in kps:
            if c > 0.35:
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
    return frame


def compose(frame, phase, counter, reps, clock_txt, ringing, err=None):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, np.uint8)
    # 左:鏡頭畫面等比放進 (CANVAS_W-PANEL_W) x CANVAS_H
    vw = CANVAS_W - PANEL_W
    h, w = frame.shape[:2]
    s = min(vw / w, CANVAS_H / h)
    fr = cv2.resize(frame, (int(w * s), int(h * s)))
    y0 = (CANVAS_H - fr.shape[0]) // 2; x0 = (vw - fr.shape[1]) // 2
    canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    # 右:面板
    px = vw
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), C_PANEL, -1)
    badge(canvas)
    draw_text(canvas, "早八起床驗證", (px + 24, 60), 40, C_TXT)
    if err:
        draw_text(canvas, "發生錯誤", (px + 24, 130), 30, C_WARN)
        for i, line in enumerate(err.split("\n")[:8]):
            draw_text(canvas, line, (px + 24, 175 + i * 30), 20, C_DIM)
        return canvas
    if phase == "clock":
        draw_text(canvas, clock_txt, (px + PANEL_W // 2, 260), 110, C_TXT, anchor="mm")
        draw_text(canvas, "鬧鐘即將響起…", (px + PANEL_W // 2, 360), 30, C_DIM, anchor="mm")
        draw_text(canvas, "空白鍵:直接響鈴", (px + 24, CANVAS_H - 60), 22, C_DIM)
    elif phase in ("alarm", "count"):
        draw_text(canvas, clock_txt, (px + 24, 110), 44, C_WARN if ringing else C_TXT)
        draw_text(canvas, f"做 {reps} 下深蹲才能關鬧鐘", (px + 24, 170), 28, C_ACC)
        big = f"{counter.count}"
        draw_text(canvas, big, (px + PANEL_W // 2 - 40, 400), 200, C_OK if counter.count else C_TXT, anchor="mm")
        draw_text(canvas, f"/ {reps}", (px + PANEL_W // 2 + 110, 430), 48, C_DIM, anchor="mm")
        st = {"stand": "站好,蹲下去!", "down": "很好,站起來!"}[counter.state]
        if phase == "alarm":
            st = "請站到鏡頭前(全身入鏡)"
        draw_text(canvas, st, (px + PANEL_W // 2, 540), 30, C_TXT, anchor="mm")
        if counter.angle is not None:
            draw_text(canvas, f"膝角 {counter.angle:.0f}°", (px + 24, CANVAS_H - 100), 22, C_DIM)
        draw_text(canvas, "r 重來  q 離開", (px + 24, CANVAS_H - 60), 22, C_DIM)
    elif phase == "done":
        draw_text(canvas, "驗證通過", (px + PANEL_W // 2, 230), 64, C_OK, anchor="mm")
        draw_text(canvas, "早安!", (px + PANEL_W // 2, 320), 90, C_TXT, anchor="mm")
        draw_text(canvas, f"{reps} 下深蹲完成,鬧鐘已關閉", (px + PANEL_W // 2, 420), 28, C_DIM, anchor="mm")
        draw_text(canvas, "r 重來  q 離開", (px + 24, CANVAS_H - 60), 22, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8s_pose.hef")
    ap.add_argument("--source", default="auto", help="auto=外接優先,否則內建;或指定編號 0/1")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--clock", type=float, default=4.0, help="時鐘畫面秒數後自動響鈴")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--snapshot", default="")
    args = ap.parse_args()

    err = None
    pose = None
    try:
        pose = PoseEstimator(args.hef, conf_threshold=args.conf)
    except Exception as e:  # 讓畫面顯示錯誤而不是閃退
        err = f"模型載入失敗:{type(e).__name__}\n{e}\n\n請確認 UGen300 已插上、\n{args.hef} 存在。"

    cap, cam_idx, cam_name = open_camera(args.source)
    if not cap.isOpened():
        err = (err or "") + f"\n鏡頭 {args.source} 打不開,試 --source 1"

    if args.selftest:
        ok, frame = cap.read()
        people = pose.infer_multi(frame) if (ok and pose) else None
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if pose else 'FAIL'} 偵測到 {len(people) if people else 0} 人")
        cap.release(); return

    counter = SquatCounter(target=args.reps, conf=0.35)
    alarm = Alarm()
    phase = "clock"; t0 = time.time(); frames = 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN, CANVAS_W, CANVAS_H)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((720, 1280, 3), np.uint8)
            frame = cv2.flip(frame, 1)
            people = []
            if pose and phase in ("alarm", "count", "clock") and not err:
                try:
                    people = pose.infer_multi(frame) or []
                except Exception as e:
                    err = f"推論失敗:{e}"
            draw_skeleton(frame, people)

            # 狀態推進
            elapsed = time.time() - t0
            if phase == "clock":
                secs = 59 - min(59, int(elapsed / args.clock * 60)) if args.clock > 0 else 0
                clock_txt = f"07:{secs:02d}"
                if elapsed >= args.clock:
                    phase = "alarm"; alarm.start()
            else:
                clock_txt = "08:00  響鈴中" if alarm.ringing else "08:00"
            if phase == "alarm" and people:
                phase = "count"
            if phase == "count":
                kps = max(people, key=lambda p: sum(c for _, _, c in p)) if people else None
                r = counter.update(kps)
                if r["just_counted"]:
                    try:
                        import winsound; winsound.Beep(1500, 60)
                    except Exception:
                        pass
                if r["done"]:
                    phase = "done"; alarm.stop(); play_victory()

            canvas = compose(frame, phase, counter, args.reps, clock_txt, alarm.ringing, err)
            cv2.imshow(WIN, canvas)
            frames += 1
            if args.snapshot and frames >= 40:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" ") and phase == "clock":
                phase = "alarm"; alarm.start()
            if k == ord("r"):
                counter.reset(); phase = "alarm"; alarm.start()
    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        if pose:
            pose.close()  # 補丁版:預設不 release VDevice


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
