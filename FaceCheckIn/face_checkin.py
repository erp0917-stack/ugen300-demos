"""
face_checkin.py —— 30 秒建檔的人臉報到:SCRFD 找臉 → ArcFace 512 維向量 → 本機人臉庫比對 → 「已報到」名單。

建檔兩種方式:
  1. 現場:按 e → 3 秒倒數 → 鎖定倒數結束時最大的那張臉,1.5 秒內取 5 個樣本平均 → 對話框輸入名字(支援中文)→ 存檔
  2. 預先:把照片放進 faces\ 資料夾,檔名就是名字(王小明.jpg),啟動時自動建檔(已有同名者略過)
存的是向量 + 64px 縮圖(faces.json),不存原始照片;報到紀錄寫 checkin.csv(寫不進去只提示,不會當掉)。
兩個模型(scrfd、arcface_mobilefacenet)Hailo-8L 也有,UGen200 同樣可用。

按鍵:e 建檔 / d 刪除最後一筆建檔 / r 清空報到名單 / v 顯示相似度 / s 存畫面 / q 或 Esc 離開
用法:python face_checkin.py [--source auto] [--fullscreen] [--thr 0.45]
      python face_checkin.py --selftest / --snapshot out.png
"""
import argparse
import csv
import glob
import os
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

from face_engine import FaceDetector, FaceEmbedder, FaceDB, square_crop, iou
from ui_text import draw_text, badge
from camera import open_camera

WIN = "UGen300 Face Check-In"
W, H, PANEL = 1280, 720, 440
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH, LOG_PATH, FACES_DIR = os.path.join(HERE, "faces.json"), os.path.join(HERE, "checkin.csv"), os.path.join(HERE, "faces")
C_BG, C_PANEL, C_TXT, C_DIM = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150)
C_OK, C_ACC, C_NO, C_LINE, C_GRAY = (80, 220, 120), (255, 170, 40), (60, 140, 255), (70, 70, 80), (110, 110, 118)
ENROLL_SAMPLES, ENROLL_EVERY, ENROLL_TIMEOUT = 5, 3, 6.0     # 5 個樣本、每 3 幀取一個、最多等 6 秒
CONFIRM_SEC, RESET_GRACE, BANNER_SEC, LIST_MAX = 0.6, 3.0, 2.5, 6
ERR_RETRY, ERR_CLEAR_SEC = 5, 3.0


def beep(f=1200, ms=120):
    try:
        import winsound; winsound.Beep(f, ms)
    except Exception: pass


def disable_ime(title):
    """把 OpenCV 視窗的中文輸入法關掉:否則打完中文名字後 e/q/d 會被 IME 吃掉。"""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd: ctypes.windll.imm32.ImmAssociateContext(hwnd, 0)
    except Exception: pass


def ask_name(default=""):
    """用 tkinter 對話框輸入名字(OpenCV 視窗打不了中文)。取消回傳 None。"""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        root.option_add("*Font", "{Microsoft JhengHei} 18"); root.option_add("*Dialog.msg.font", "{Microsoft JhengHei} 18")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight(); root.geometry(f"+{sw // 2 - 200}+{sh // 2 - 80}")
        name = simpledialog.askstring("人臉建檔", "請輸入名字,按 Enter 存檔:", initialvalue=default, parent=root)
        root.destroy()
        return name.strip() if name and name.strip() else None
    except Exception as e:  # noqa: BLE001
        print("[輸入名字失敗]", e); return None


def preload_faces(det, emb, db):
    """faces\ 內的照片 → 檔名當名字 → 建檔(已存在的名字略過,所以按 d 刪掉後要連照片一起移走才不會回來)。"""
    added, skipped = [], []
    for p in sorted(glob.glob(os.path.join(FACES_DIR, "*.*"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in db.names() or not p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")): continue
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None: skipped.append(name); continue
        if max(img.shape[:2]) > 1600:
            s = 1600 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s)
        faces = det.detect(img)
        v = emb.embed(img, faces[0]["kps"]) if faces else None
        if v is None: skipped.append(name); print(f"[預先建檔] {name}:照片裡找不到臉,略過"); continue
        db.add(name, v, square_crop(img, faces[0]["box"])); added.append(name)
    if added: print("[預先建檔]", "、".join(added))
    return added, skipped


def log_checkin(name):
    """寫 checkin.csv;被 Excel 鎖住時回傳錯誤字串而不是崩潰。"""
    try:
        new = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); new and w.writerow(["time", "name"]); w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name])
        return None
    except OSError as e:
        return f"checkin.csv 無法寫入(被 Excel 開著?):{type(e).__name__}"


def fit(frame, vw):
    h, w = frame.shape[:2]; s = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * s), int(h * s)))
    return fr, s, (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2


def compose(frame, faces, results, db, checked, ui, fatal=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    fr, s, x0, y0 = fit(frame, vw)
    if ui["phase"] == "live" and not fatal:
        for f, (name, sim) in zip(faces, results):
            x1, y1, x2, y2 = [int(v * s) for v in f["box"]]
            if name:
                col = C_OK; cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
                draw_text(fr, name + ("  已報到" if name in checked else ""), (x1, max(0, y1 - 36)), 30, col)
            else:
                cv2.rectangle(fr, (x1, y1), (x2, y2), C_GRAY, 1)
                draw_text(fr, "未建檔", (x1, max(0, y1 - 26)), 20, C_GRAY)
            if ui["show_sim"]: draw_text(fr, f"{sim:.2f}", (x2 - 4, y2 + 2), 16, C_DIM, anchor="ra")
    canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    cx, cy = x0 + fr.shape[1] // 2, y0 + fr.shape[0] // 2
    if fatal:
        cv2.rectangle(canvas, (x0, y0), (x0 + fr.shape[1], y0 + fr.shape[0]), (20, 20, 90), -1)
        draw_text(canvas, "無法啟動", (cx, cy - 60), 60, C_NO, anchor="mm")
        for i, line in enumerate(fatal.split("\n")[:4]): draw_text(canvas, line[:40], (cx, cy + 10 + i * 34), 24, C_TXT, anchor="mm")
    elif ui["phase"] == "countdown":
        draw_text(canvas, f"{ui['countdown']}", (cx, cy), 220, C_ACC, anchor="mm")
        draw_text(canvas, "看鏡頭,保持不動", (cx, cy + 150), 34, C_TXT, anchor="mm")
    elif ui["phase"] == "capture":
        if ui.get("lock_box"):
            x1, y1, x2, y2 = [int(v * s) for v in ui["lock_box"]]; cv2.rectangle(canvas, (x0 + x1, y0 + y1), (x0 + x2, y0 + y2), C_ACC, 3)
        draw_text(canvas, f"擷取中 {ui['n_samples']}/{ENROLL_SAMPLES}", (cx, y0 + 60), 44, C_ACC, anchor="mm")
    elif ui["phase"] == "naming":
        cv2.rectangle(canvas, (x0, cy - 60), (x0 + fr.shape[1], cy + 60), (30, 30, 36), -1)
        draw_text(canvas, "請在跳出的對話框輸入名字", (cx, cy), 44, C_ACC, anchor="mm")
    if ui.get("banner") and time.time() < ui["banner_until"]:
        cv2.rectangle(canvas, (x0, cy - 70), (x0 + fr.shape[1], cy + 70), (30, 70, 40), -1)
        draw_text(canvas, ui["banner"], (cx, cy), 64, C_OK, anchor="mm")
    if ui.get("warn"):
        draw_text(canvas, ui["warn"], (x0 + 16, y0 + 12), 22, C_NO)
    # 右側面板
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "人臉報到", (px + 24, 50), 40, C_TXT)
    draw_text(canvas, "只存 512 個數字,不存照片、不上傳", (px + 24, 100), 20, C_OK)
    draw_text(canvas, "SCRFD + ArcFace · UGen200 亦可執行", (px + 24, 128), 16, C_DIM)
    draw_text(canvas, f"{len(checked)}", (px + 24, 150), 96, C_OK if checked else C_TXT)
    draw_text(canvas, "人已報到", (px + 24 + len(str(len(checked))) * 54 + 10, 208), 28, C_DIM)
    draw_text(canvas, f"已建檔 {len(db.people)} 人", (W - 24, 216), 18, C_DIM, anchor="ra")
    cv2.line(canvas, (px + 24, 266), (W - 24, 266), C_LINE, 1)
    yy = 280
    for name, t in list(checked.items())[-LIST_MAX:]:
        p = next((p for p in db.people if p["name"] == name), None)
        if p and p["thumb"] is not None:
            canvas[yy:yy + 44, px + 24:px + 68] = cv2.resize(p["thumb"], (44, 44))
        draw_text(canvas, name[:8], (px + 82, yy + 2), 28, C_TXT)
        draw_text(canvas, t, (W - 24, yy + 12), 18, C_DIM, anchor="ra")
        yy += 54
    if len(checked) > LIST_MAX: draw_text(canvas, f"…另有 {len(checked) - LIST_MAX} 人", (px + 82, yy), 16, C_DIM)
    if not checked and not fatal: draw_text(canvas, "站到鏡頭前即可報到", (px + PANEL // 2, 400), 28, C_DIM, anchor="mm")
    if ui.get("msg"): draw_text(canvas, ui["msg"][:22], (px + 24, H - 132), 24, C_OK if not ui.get("msg_bad") else C_NO)
    draw_text(canvas, f"偵測 {ui['ms']:.0f} ms", (px + 24, H - 92), 16, C_DIM)
    draw_text(canvas, "e 建檔  d 刪最後一筆  r 清單重來  q 離開", (px + 24, H - 62), 18, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", help="auto=外接優先,否則內建;或指定編號 0/1")
    ap.add_argument("--det", default="scrfd_10g.hef"); ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--image", default="", help="用靜態照片代替鏡頭(測試用)")
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    fatal = None; det = emb = None
    try:
        det = FaceDetector(args.det, conf=0.5); emb = FaceEmbedder()
    except Exception as e:  # noqa: BLE001
        fatal = f"模型載入失敗:{type(e).__name__}\n請確認 UGen300 已插上,且沒有其他 demo 正在使用它"; print("[模型]", e)
    db = FaceDB(DB_PATH)
    if det and emb: preload_faces(det, emb, db)
    still = None
    if args.image:
        still = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if still is None: print("讀不到圖片:", args.image); return
        cap, cam_idx, cam_name = cv2.VideoCapture(), -1, "靜態圖"; cam_ok = True
    else:
        cap, cam_idx, cam_name = open_camera(args.source); cam_ok = cap.isOpened()
    if not cam_ok and not fatal: fatal = "找不到可用鏡頭\n請關閉其他正在用鏡頭的程式(Teams、相機)後重開"

    def grab():
        if still is not None: return True, still.copy()
        return cap.read() if cam_ok else (False, None)

    if args.selftest:
        ok, f = grab(); faces = det.detect(cv2.flip(f, 1)) if (ok and det) else []
        res = [db.match(emb.embed(cv2.flip(f, 1), x["kps"]), args.thr) for x in faces] if faces else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'}({cam_name}) 模型={'OK' if det else 'FAIL'} 臉={len(faces)} 建檔={db.names()} "
              f"比對={[(n or '未建檔', round(s, 2)) for n, s in res]}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    cv2.waitKey(1); disable_ime(WIN)
    checked = {}; first_seen = {}; frames = 0; n_err = 0; err_at = 0.0
    ui = dict(phase="live", countdown=0, n_samples=0, lock_box=None, banner="", banner_until=0.0, warn="", msg="", msg_bad=False, show_sim=False, ms=0.0)
    if db.last_error: ui.update(msg=db.last_error, msg_bad=True); msg_t = time.time() + 6
    else: msg_t = 0.0
    t0 = 0.0; samples = []; sample_thumb = None; reset_until = 0.0
    try:
        while True:
            ok, frame = grab()
            frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            faces, results = [], []
            now = time.time()
            if det and not fatal and ok:
                try:
                    t = time.time(); faces = det.detect(frame); ui["ms"] = 0.8 * ui["ms"] + 0.2 * (time.time() - t) * 1000
                    results = [db.match(emb.embed(frame, f["kps"]), args.thr) for f in faces]
                    n_err = 0
                except Exception as e:  # noqa: BLE001  一次抖動不該讓整支 demo 停擺:顯示警告、繼續嘗試
                    n_err += 1; err_at = now; ui["warn"] = f"推論失敗({n_err}):{type(e).__name__}"; print("[推論]", e)
                    if n_err >= ERR_RETRY: fatal = f"UGen300 連續推論失敗 {n_err} 次\n請重新插拔後重開程式"
            if ui["warn"] and now - err_at > ERR_CLEAR_SEC: ui["warn"] = ""
            # 報到:同一名字連續看到 CONFIRM_SEC 秒才算;按 r 之後 RESET_GRACE 秒內不報到
            seen = set()
            for name, sim in results:
                if not name: continue
                seen.add(name); first_seen.setdefault(name, now)
                if now - first_seen[name] >= CONFIRM_SEC and name not in checked and now >= reset_until:
                    checked[name] = datetime.now().strftime("%H:%M:%S"); beep(1400, 120)
                    ui.update(banner=f"{name}  已報到", banner_until=now + BANNER_SEC)
                    e = log_checkin(name)
                    if e: ui.update(msg=e, msg_bad=True); msg_t = now + 6
            for n in list(first_seen):
                if n not in seen: del first_seen[n]
            # 建檔流程
            if ui["phase"] == "countdown":
                left = 3 - int(now - t0); ui["countdown"] = max(1, left)
                if left <= 0:
                    if faces:
                        ui.update(phase="capture", n_samples=0, lock_box=faces[0]["box"]); samples = []; sample_thumb = None; t0 = now; beep(1000, 80)
                    else:
                        ui.update(phase="live", msg="沒抓到臉,再按 e 重試", msg_bad=True); msg_t = now + 4
            elif ui["phase"] == "capture":
                if faces and frames % ENROLL_EVERY == 0:
                    f = max(faces, key=lambda f: iou(f["box"], ui["lock_box"]))
                    if iou(f["box"], ui["lock_box"]) > 0.3:
                        ui["lock_box"] = f["box"]
                        try: v = emb.embed(frame, f["kps"])
                        except Exception: v = None  # noqa: BLE001
                        if v is not None:
                            samples.append(v); ui["n_samples"] = len(samples)
                            if sample_thumb is None: sample_thumb = square_crop(frame, f["box"])
                if len(samples) >= ENROLL_SAMPLES:
                    v = np.mean(samples, 0); v /= np.linalg.norm(v) + 1e-6
                    ui["phase"] = "naming"
                    cv2.imshow(WIN, compose(frame, faces, results, db, checked, ui, fatal)); cv2.waitKey(1)
                    name = ask_name(); disable_ime(WIN)
                    if name:
                        existed, saved = db.add(name, v, sample_thumb); checked.pop(name, None); first_seen.pop(name, None)
                        ui.update(msg=(f"已更新建檔:{name}" if existed else f"已建檔:{name}") if saved else db.last_error, msg_bad=not saved); beep(1500, 100)
                    else: ui.update(msg="已取消建檔", msg_bad=True)
                    msg_t = now + 4; ui["phase"] = "live"
                elif now - t0 > ENROLL_TIMEOUT:
                    ui.update(phase="live", msg="臉離開了畫面,再按 e 重試", msg_bad=True); msg_t = now + 4
            if ui["msg"] and now > msg_t: ui.update(msg="", msg_bad=False)
            canvas = compose(frame, faces, results, db, checked, ui, fatal)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 40:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            key = chr(k).lower() if 32 <= k < 127 else ("esc" if k == 27 else "")
            if key in ("q", "esc"): break
            if not args.fullscreen and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break   # 按了視窗的 X
            if fatal: continue
            if key == "e" and ui["phase"] == "live": ui["phase"] = "countdown"; t0 = now
            if key == "d":
                n = db.remove_last(); checked.pop(n, None); first_seen.pop(n, None)
                ui.update(msg=(f"已刪除建檔:{n}" if n else "沒有建檔資料"), msg_bad=not n); msg_t = now + 4
            if key == "r":
                checked.clear(); first_seen.clear(); reset_until = now + RESET_GRACE
                ui.update(msg="報到名單已清空", msg_bad=False, banner=""); msg_t = now + 4
            if key == "v": ui["show_sim"] = not ui["show_sim"]
            if key == "s":
                p = os.path.join(HERE, f"checkin_{datetime.now():%H%M%S}.png"); cv2.imwrite(p, canvas); ui.update(msg="已存畫面", msg_bad=False); msg_t = now + 3
    finally:
        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    import hailo_vdevice
    code = 0
    try:
        main()
    except Exception:  # noqa: BLE001  任何未預期錯誤都要走硬退出,否則 HailoRT 收尾會讓 UGen300 掉線
        traceback.print_exc(); code = 1
    except KeyboardInterrupt:
        pass
    hailo_vdevice.exit_now(code)
