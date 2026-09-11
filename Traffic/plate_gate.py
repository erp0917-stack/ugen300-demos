"""
plate_gate.py —— 車牌辨識停車柵欄:鏡頭看到車 → 找車牌 → OCR → 白名單比對 → 柵欄開/擋 + 進出紀錄。

按鍵:空白鍵 手動辨識目前畫面 / a 把最後辨識到的車牌加入白名單 / q 離開
用法:python plate_gate.py --source 0 [--fullscreen]
      python plate_gate.py --image car.jpg [--snapshot out.png]   (靜態圖,現場沒車也能演)
      python plate_gate.py --selftest
白名單:whitelist.json;紀錄:gate_log.csv(時間,車牌,結果)
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from hailo_detect import ObjectDetector as HailoDetector
from ocr_engine import OCREngine, normalize_plate
from ui_text import draw_text, badge
from camera import open_camera

WIN = "UGen300 Plate Gate"
W, H, PANEL = 1280, 720, 440
HERE = os.path.dirname(os.path.abspath(__file__))
WL_PATH, LOG_PATH = os.path.join(HERE, "whitelist.json"), os.path.join(HERE, "gate_log.csv")
VEHICLES = {"car", "truck", "bus", "motorcycle"}
C_BG, C_PANEL, C_TXT, C_DIM, C_OK, C_NO, C_ACC = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150), (80, 220, 120), (60, 60, 230), (255, 170, 40)


WL_WARNING = None   # 白名單檔壞掉時的訊息,顯示在面板上而不是靜默吞掉


def load_wl():
    """沒有檔案 → 用預設示範白名單;檔案存在但壞掉 → 空白名單 + 面板警告(不能靜默放行/擋人)。"""
    global WL_WARNING
    if not os.path.exists(WL_PATH):
        return {"ABC-1234"}
    try:
        data = json.load(open(WL_PATH, encoding="utf-8"))
        return set(str(x).upper() for x in data)
    except (OSError, ValueError) as e:
        WL_WARNING = f"whitelist.json 讀取失敗:{type(e).__name__}"; return set()


def save_wl(wl):
    json.dump(sorted(wl), open(WL_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def log(plate, result):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f); new and w.writerow(["time", "plate", "result"]); w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), plate, result])


class Gate:
    """柵欄動畫狀態:closed / opening / open / denied。"""
    def __init__(self): self.state = "closed"; self.t = 0
    def open(self): self.state, self.t = "opening", time.time()
    def deny(self): self.state, self.t = "denied", time.time()
    def angle(self):
        if self.state == "opening":
            a = min(90, (time.time() - self.t) * 90 / 1.0)
            if a >= 90: self.state = "open"
            return a
        if self.state == "open":
            if time.time() - self.t > 5: self.state = "closed"
            return 90
        if self.state == "denied" and time.time() - self.t > 3: self.state = "closed"
        return 0


def draw_gate(canvas, gate, x, y):
    a = gate.angle(); L = 220
    ex, ey = int(x + L * np.cos(np.radians(a))), int(y - L * np.sin(np.radians(a)))
    cv2.rectangle(canvas, (x - 14, y - 10), (x + 14, y + 60), (80, 80, 80), -1)
    cv2.line(canvas, (x, y), (ex, ey), (60, 60, 230) if gate.state == "denied" else (240, 240, 240), 12)
    for k in range(1, 6):
        px, py = int(x + L * k / 6 * np.cos(np.radians(a))), int(y - L * k / 6 * np.sin(np.radians(a)))
        cv2.circle(canvas, (px, py), 6, (60, 60, 230), -1)


def compose(frame, dets, plates, last, gate, wl, mode, err=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    h, w = frame.shape[:2]; s = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * s), int(h * s)))
    x0, y0 = (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2
    for lab, sc, (x1, y1, x2, y2) in dets:
        cv2.rectangle(fr, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), (255, 170, 40), 2)
        draw_text(fr, f"{lab} {sc:.2f}", (int(x1 * s), max(0, int(y1 * s) - 26)), 20, (255, 170, 40))
    for p in plates:
        x1, y1, x2, y2 = p["box"]; cv2.rectangle(fr, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), C_OK, 3)
        draw_text(fr, p["text"], (int(x1 * s), int(y2 * s) + 4), 26, C_OK)
    canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "車牌辨識柵欄", (px + 24, 56), 38, C_TXT)
    draw_text(canvas, "YOLO 找車 → OCR 讀牌 → 白名單 → 開柵", (px + 24, 106), 17, C_DIM)
    if err:
        draw_text(canvas, "錯誤:" + err[:40], (px + 24, 150), 20, C_NO)
    draw_gate(canvas, gate, px + 60, 300)
    if last:
        color = C_OK if last["ok"] else C_NO
        draw_text(canvas, last["plate"], (px + PANEL // 2, 420), 64, color, anchor="mm")
        draw_text(canvas, "白名單 · 柵欄開啟" if last["ok"] else "不在白名單 · 禁止進入", (px + PANEL // 2, 480), 26, color, anchor="mm")
        draw_text(canvas, f"{last['dt']*1000:.0f} ms", (px + PANEL // 2, 515), 18, C_DIM, anchor="mm")
    else:
        draw_text(canvas, "等待車輛…" if mode == "auto" else "按空白鍵辨識", (px + PANEL // 2, 440), 28, C_DIM, anchor="mm")
    draw_text(canvas, f"白名單 {len(wl)} 筆" + (f"  警告:{WL_WARNING}" if WL_WARNING else ""), (px + 24, H - 130), 20, C_NO if WL_WARNING else C_DIM)
    draw_text(canvas, "空白鍵 辨識  a 加入白名單  r 清除  q 離開", (px + 24, H - 60), 20, C_DIM)
    return canvas


def find_plates(ocr, frame, dets):
    """在車輛框內(或整張圖)找車牌字串。"""
    regions = [b for lab, sc, b in dets if lab in VEHICLES] or [(0, 0, frame.shape[1], frame.shape[0])]
    found = []
    for x1, y1, x2, y2 in regions:
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        for r in ocr.read_all(crop, min_conf=0.4):
            p = normalize_plate(r["text"])
            if p:
                bx = r["box"]; found.append(dict(text=p, conf=r["conf"], box=(bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1)))
    found.sort(key=lambda r: -r["conf"])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", help="auto=外接優先,否則內建;或指定編號 0/1"); ap.add_argument("--image", default="")
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    ap.add_argument("--auto", action="store_true", help="看到車就自動辨識(預設按空白鍵)")
    args = ap.parse_args()
    err = None; det = None; ocr = None
    try:
        det = HailoDetector("yolov8s.hef", conf_threshold=0.4); ocr = OCREngine()
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    still = cv2.imread(args.image) if args.image else None
    cap = None if still is not None else open_camera(args.source)[0]
    wl = load_wl(); gate = Gate(); last = None; plates = []; dets = []
    if args.selftest:
        frame = still if still is not None else cap.read()[1]
        t = time.time(); dets = det.infer(frame) if det else []; plates = find_plates(ocr, frame, dets) if ocr else []
        print(f"[selftest] 偵測 {len(dets)} 物件,車牌 {[p['text'] for p in plates]},{(time.time()-t)*1000:.0f} ms"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    frames = 0; last_auto = 0
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read(); frame = frame if ok else np.zeros((720, 1280, 3), np.uint8)
            else:
                frame = still.copy()
            if det and not err:
                try: dets = det.infer(frame)
                except Exception as e: err = f"推論失敗:{e}"; dets = []
            has_vehicle = any(lab in VEHICLES for lab, _, _ in dets)
            do = False
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord(" ") or (args.snapshot and frames == 3): do = True
            if args.auto and has_vehicle and time.time() - last_auto > 4: do = True
            if k == ord("a") and last: wl.add(last["plate"]); save_wl(wl)
            if k == ord("r"): last = None; plates = []; gate = Gate()
            if do and ocr:
                t = time.time(); plates = find_plates(ocr, frame, dets); dt = time.time() - t; last_auto = time.time()
                if plates:
                    p = plates[0]["text"]; okp = p in wl
                    last = dict(plate=p, ok=okp, dt=dt); gate.open() if okp else gate.deny(); log(p, "open" if okp else "deny")
                    try:
                        import winsound; winsound.Beep(1200 if okp else 300, 150)
                    except Exception: pass
                else:
                    last = dict(plate="未辨識到車牌", ok=False, dt=dt)
            canvas = compose(frame, dets, plates, last, gate, wl, "auto" if args.auto else "manual", err)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 45:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        if det: det.close()


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
