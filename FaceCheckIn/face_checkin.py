"""
face_checkin.py —— 30 秒建檔的人臉報到:SCRFD 找臉 → ArcFace 512 維向量 → 本機人臉庫比對 → 「已報到」名單。

建檔兩種方式:
  1. 現場:按 e → 3 秒倒數 → 自動抓 5 幀取平均向量 → 跳出輸入框打名字(支援中文)→ 存檔
  2. 預先:把照片放進 faces\ 資料夾,檔名就是名字(王小明.jpg),啟動時自動建檔
存的是向量 + 64px 縮圖(faces.json),不存原始照片;報到紀錄寫 checkin.csv。

按鍵:e 建檔 / d 刪除最後一筆建檔 / r 清空報到名單 / s 存畫面 / q 離開
用法:python face_checkin.py [--source auto] [--fullscreen] [--thr 0.45]
      python face_checkin.py --selftest / --snapshot out.png
"""
import argparse
import csv
import glob
import os
import time
from datetime import datetime

import cv2
import numpy as np

from face_engine import FaceDetector, FaceEmbedder, FaceDB
from ui_text import draw_text, badge
from camera import open_camera

WIN = "UGen300 Face Check-In"
W, H, PANEL = 1280, 720, 440
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH, LOG_PATH, FACES_DIR = os.path.join(HERE, "faces.json"), os.path.join(HERE, "checkin.csv"), os.path.join(HERE, "faces")
C_BG, C_PANEL, C_TXT, C_DIM = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150)
C_OK, C_ACC, C_NO, C_LINE = (80, 220, 120), (255, 170, 40), (60, 140, 255), (70, 70, 80)
ENROLL_FRAMES, CONFIRM_FRAMES = 5, 3


def beep(f=1200, ms=120):
    try:
        import winsound; winsound.Beep(f, ms)
    except Exception: pass


def ask_name(default=""):
    """用 tkinter 對話框輸入名字(OpenCV 視窗打不了中文)。取消回傳 None。"""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        name = simpledialog.askstring("人臉建檔", "請輸入名字:", initialvalue=default, parent=root)
        root.destroy()
        return name.strip() if name else None
    except Exception as e:  # noqa: BLE001
        print("[輸入名字失敗]", e); return None


def preload_faces(det, emb, db):
    """faces\ 內的照片 → 檔名當名字 → 建檔(已存在的名字略過)。"""
    added = []
    for p in sorted(glob.glob(os.path.join(FACES_DIR, "*.*"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in db.names() or not p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")): continue
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        faces = det.detect(img)
        if not faces: print(f"[預先建檔] {name}:照片裡找不到臉,略過"); continue
        f = faces[0]; x1, y1, x2, y2 = f["box"]
        db.add(name, emb.embed(img, f["kps"]), crop(img, f["box"])); added.append(name)
    if added: print("[預先建檔]", "、".join(added))


def crop(img, box, pad=0.2):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad)); x2, y2 = min(img.shape[1], int(x2 + w * pad)), min(img.shape[0], int(y2 + h * pad))
    c = img[y1:y2, x1:x2]
    return c if c.size else img[:64, :64]


def log_checkin(name):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); new and w.writerow(["time", "name"]); w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name])


def compose(frame, faces, results, db, checked, phase, countdown, msg, fps, err=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    h, w = frame.shape[:2]; s = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * s), int(h * s)))
    for f, (name, sim) in zip(faces, results):
        x1, y1, x2, y2 = [int(v * s) for v in f["box"]]
        col = C_OK if name else C_ACC
        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
        label = f"{name}" if name else "訪客"
        if name and name in checked: label += "  已報到"
        draw_text(fr, label, (x1, max(0, y1 - 34)), 26, col)
        draw_text(fr, f"{sim:.2f}", (x2 - 4, y2 + 2), 16, C_DIM, anchor="ra")
    x0, y0 = (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2; canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    if phase == "countdown":
        draw_text(canvas, f"{countdown}", (x0 + fr.shape[1] // 2, y0 + fr.shape[0] // 2), 200, C_ACC, anchor="mm")
        draw_text(canvas, "看鏡頭,保持不動", (x0 + fr.shape[1] // 2, y0 + fr.shape[0] // 2 + 140), 32, C_TXT, anchor="mm")
    elif phase == "capture":
        draw_text(canvas, "擷取中…", (x0 + fr.shape[1] // 2, y0 + 60), 40, C_ACC, anchor="mm")
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "人臉報到", (px + 24, 56), 40, C_TXT)
    draw_text(canvas, "SCRFD 找臉 → ArcFace 向量 → 本機比對 · 不上傳", (px + 24, 108), 16, C_DIM)
    if err: draw_text(canvas, "錯誤:" + err[:36], (px + 24, 136), 18, C_NO)
    draw_text(canvas, f"已報到  {len(checked)} / 建檔 {len(db.people)}", (px + 24, 160), 28, C_ACC)
    cv2.line(canvas, (px + 24, 200), (W - 24, 200), C_LINE, 1)
    yy = 214
    for name, t in list(checked.items())[-9:]:
        p = next((p for p in db.people if p["name"] == name), None)
        if p and p["thumb"] is not None:
            th = cv2.resize(p["thumb"], (44, 44)); canvas[yy:yy + 44, px + 24:px + 68] = th
        draw_text(canvas, name, (px + 82, yy + 2), 26, C_TXT)
        draw_text(canvas, t, (W - 24, yy + 10), 18, C_DIM, anchor="ra")
        yy += 52
    if not checked: draw_text(canvas, "站到鏡頭前即可報到", (px + PANEL // 2, 330), 24, C_DIM, anchor="mm")
    if msg: draw_text(canvas, msg, (px + 24, H - 130), 22, C_OK)
    draw_text(canvas, f"{fps:.0f} fps", (px + 24, H - 96), 18, C_DIM)
    draw_text(canvas, "e 建檔  d 刪最後一筆  r 清單重來  q 離開", (px + 24, H - 60), 19, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", help="auto=外接優先,否則內建;或指定編號 0/1")
    ap.add_argument("--det", default="scrfd_10g.hef"); ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; det = emb = None
    try:
        det = FaceDetector(args.det, conf=0.5); emb = FaceEmbedder()
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    db = FaceDB(DB_PATH)
    if det and emb: preload_faces(det, emb, db)
    cap, cam_idx, cam_name = open_camera(args.source)
    if args.selftest:
        ok, f = cap.read(); faces = det.detect(cv2.flip(f, 1)) if (ok and det) else []
        res = [db.match(emb.embed(cv2.flip(f, 1), x["kps"]), args.thr) for x in faces] if faces else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if det else 'FAIL'} 臉={len(faces)} 建檔={db.names()} 比對={res}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    checked = {}; phase = "live"; t0 = 0; samples = []; sample_thumb = None; msg = ""; msg_t = 0
    streak = {}; fps = 0.0; t_last = time.time(); frames = 0
    try:
        while True:
            ok, frame = cap.read(); frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            faces, results = [], []
            if det and not err:
                try:
                    faces = det.detect(frame)
                    for f in faces:
                        v = emb.embed(frame, f["kps"]); results.append(db.match(v, args.thr))
                except Exception as e:  # noqa: BLE001
                    err = f"推論失敗:{e}"
            # 報到:同一名字連續 CONFIRM_FRAMES 幀才算
            seen = set()
            for name, sim in results:
                if not name: continue
                seen.add(name); streak[name] = streak.get(name, 0) + 1
                if streak[name] >= CONFIRM_FRAMES and name not in checked:
                    checked[name] = datetime.now().strftime("%H:%M:%S"); log_checkin(name); beep(1400, 120)
                    msg, msg_t = f"{name} 已報到", time.time()
            for n in list(streak):
                if n not in seen: streak[n] = 0
            # 建檔流程
            countdown = 0
            if phase == "countdown":
                left = 3 - int(time.time() - t0); countdown = max(1, left)
                if left <= 0: phase = "capture"; samples = []; sample_thumb = None; beep(1000, 80)
            elif phase == "capture":
                if faces:
                    f = faces[0]; samples.append(emb.embed(frame, f["kps"]))
                    if sample_thumb is None: sample_thumb = crop(frame, f["box"])
                if len(samples) >= ENROLL_FRAMES:
                    v = np.mean(samples, 0); v /= np.linalg.norm(v) + 1e-6
                    cv2.imshow(WIN, compose(frame, faces, results, db, checked, "live", 0, "請在對話框輸入名字", fps, err)); cv2.waitKey(1)
                    name = ask_name()
                    if name: db.add(name, v, sample_thumb); streak.pop(name, None); msg, msg_t = f"已建檔:{name}", time.time(); beep(1500, 100)
                    else: msg, msg_t = "已取消建檔", time.time()
                    phase = "live"
                elif time.time() - t0 > 8: phase = "live"; msg, msg_t = "沒抓到臉,再按 e 重試", time.time()
            now = time.time(); fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last); t_last = now
            if msg and now - msg_t > 4: msg = ""
            canvas = compose(frame, faces, results, db, checked, phase, countdown, msg, fps, err)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 40:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("e") and phase == "live": phase = "countdown"; t0 = time.time()
            if k == ord("d"):
                n = db.remove_last(); msg, msg_t = (f"已刪除:{n}" if n else "沒有建檔資料"), time.time(); checked.pop(n, None)
            if k == ord("r"): checked.clear(); streak.clear(); msg, msg_t = "報到名單已清空", time.time()
            if k == ord("s"):
                p = os.path.join(HERE, f"checkin_{datetime.now():%H%M%S}.png"); cv2.imwrite(p, canvas); msg, msg_t = "已存畫面", time.time()
    finally:
        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
