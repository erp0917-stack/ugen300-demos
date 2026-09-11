"""
edge_vs_cloud.py —— 邊緣 vs 雲端對照儀表:同一支鏡頭,左邊 UGen300 即時偵測,右邊算「如果送雲端」要多久、多少錢。

即時實測:UGen300 每幀推論毫秒、fps、累計張數;網路往返(TCP 連線到雲端端點的時間,每 2 秒量一次)。
標示為「估算/參考」的:雲端 API 推論時間、每千張單價、功耗(UGen300 USB 版無法量功耗,用規格典型值)。

按鍵:r 歸零累計 / q 離開
用法:python edge_vs_cloud.py --source 0 [--fullscreen] / --selftest / --snapshot out.png
"""
import argparse
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np

from hailo_detect import ObjectDetector
from ui_text import draw_text, badge

WIN = "UGen300 Edge vs Cloud"
W, H = 1280, 720
VIDEO_W = 640
C_BG, C_TXT, C_DIM, C_ACC, C_OK, C_NO = (24, 24, 28), (240, 240, 240), (150, 150, 150), (255, 170, 40), (80, 220, 120), (60, 60, 230)
# 參考值(可在現場口頭說明來源)
CLOUD_INFER_MS = 250          # 雲端視覺 API 典型伺服器端處理時間(不含網路)
CLOUD_USD_PER_1000 = 1.5      # 主流雲端影像辨識 API 每 1000 張約 US$1–1.5
USD_TWD = 32.0
UGEN_WATT = 2.5               # Hailo-10H 典型功耗(規格值)
CLOUD_WATT = 300              # 一張雲端推論 GPU 卡的典型功耗(參考)
CLOUD_HOST = ("1.1.1.1", 443)


class NetProbe:
    """背景量測到雲端端點的 TCP 連線時間(近似網路往返)。"""
    def __init__(self):
        self.rtt_ms = None; self.online = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t = time.time()
            try:
                socket.create_connection(CLOUD_HOST, timeout=1.5).close()
                self.rtt_ms = (time.time() - t) * 1000; self.online = True
            except OSError:
                self.rtt_ms = None; self.online = False
            time.sleep(2)


def bar(canvas, x, y, w, h, frac, color):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 58), -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * min(1.0, frac)), y + h), color, -1)


def compose(frame, dets, edge_ms, fps, frames_total, net, elapsed, err=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8)
    # 左:鏡頭 + 偵測框(縮到 640 寬)
    h, w = frame.shape[:2]; s = VIDEO_W / w; fr = cv2.resize(frame, (VIDEO_W, int(h * s)))
    for lab, sc, (x1, y1, x2, y2) in dets:
        cv2.rectangle(fr, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), C_OK, 2)
        draw_text(fr, lab, (int(x1 * s), max(0, int(y1 * s) - 24)), 18, C_OK)
    y0 = 90; canvas[y0:y0 + fr.shape[0], 20:20 + VIDEO_W] = fr
    draw_text(canvas, "同一支鏡頭,兩種做法", (20, 30), 34, C_TXT)
    badge(canvas)
    if err: draw_text(canvas, "錯誤:" + err[:50], (20, y0 + fr.shape[0] + 10), 18, C_NO)

    # 右:兩欄對照
    xE, xC, cw = 690, 990, 270
    draw_text(canvas, "UGen300 邊緣", (xE, 90), 28, C_OK); draw_text(canvas, "雲端 API", (xC, 90), 28, C_ACC)
    rows = []
    cloud_total = (net.rtt_ms or 0) * 2 + CLOUD_INFER_MS if net.online else None
    rows.append(("每張延遲", f"{edge_ms:.0f} ms", f"{cloud_total:.0f} ms" if cloud_total else "離線:無法使用", edge_ms / 1000, (cloud_total or 1000) / 1000))
    rows.append(("其中網路往返", "0 ms(不用網路)", f"{net.rtt_ms * 2:.0f} ms(實測)" if net.rtt_ms else "—", 0, ((net.rtt_ms or 0) * 2) / 1000))
    rows.append(("處理速度", f"{fps:.0f} 張/秒", f"{1000 / cloud_total:.1f} 張/秒" if cloud_total else "—", min(1, fps / 60), min(1, (1000 / cloud_total if cloud_total else 0) / 60)))
    usd = frames_total * CLOUD_USD_PER_1000 / 1000
    monthly = fps * 3600 * 8 * 22 * CLOUD_USD_PER_1000 / 1000  # 每天 8 小時、每月 22 天
    rows.append(("目前已處理", f"{frames_total:,} 張", f"{frames_total:,} 張", 0, 0))
    rows.append(("費用", "0 元(一次買斷)", f"US${usd:,.2f} ≈ NT${usd * USD_TWD:,.0f}", 0, min(1, usd / 5)))
    rows.append(("以此速度跑一個月", "0 元", f"US${monthly:,.0f} ≈ NT${monthly * USD_TWD:,.0f}", 0, 1))
    rows.append(("功耗(規格參考)", f"≈{UGEN_WATT} W", f"≈{CLOUD_WATT} W(GPU)", UGEN_WATT / CLOUD_WATT, 1))
    rows.append(("影像去哪", "不出這台電腦", "上傳到別人的伺服器", 0, 0))
    y = 140
    for name, ev, cv_, ef, cf in rows:
        draw_text(canvas, name, (xE, y), 17, C_DIM)
        draw_text(canvas, ev, (xE, y + 22), 24, C_TXT); draw_text(canvas, cv_, (xC, y + 22), 24, C_TXT)
        if ef or cf:
            bar(canvas, xE, y + 56, cw, 8, ef, C_OK); bar(canvas, xC, y + 56, cw, 8, cf, C_ACC)
        y += 72
    draw_text(canvas, f"網路:{'連線中' if net.online else '已斷線'} · 雲端單價 US${CLOUD_USD_PER_1000}/千張、伺服器端 {CLOUD_INFER_MS} ms 為參考值 · 已跑 {elapsed:.0f} 秒", (20, H - 40), 16, C_DIM)
    draw_text(canvas, "r 歸零  q 離開", (W - 160, H - 40), 16, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=0); ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    err = None; det = None
    try:
        det = ObjectDetector("yolov8s.hef", conf_threshold=0.4)
    except Exception as e:  # noqa: BLE001
        err = f"模型載入失敗:{e}"
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    net = NetProbe()
    if args.selftest:
        ok, f = cap.read(); t = time.time(); d = det.infer(f) if (ok and det) else []; ms = (time.time() - t) * 1000
        time.sleep(2.2); print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if det else 'FAIL'} 推論 {ms:.0f} ms 物件 {len(d)} 網路 RTT {net.rtt_ms}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    lat = deque(maxlen=30); fps = 0.0; t_last = time.time(); frames_total = 0; t0 = time.time(); frames = 0
    try:
        while True:
            ok, frame = cap.read(); frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
            dets = []
            if det and not err:
                t = time.time()
                try: dets = det.infer(frame)
                except Exception as e: err = f"推論失敗:{e}"
                lat.append((time.time() - t) * 1000); frames_total += 1
            now = time.time(); fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last); t_last = now
            canvas = compose(frame, dets, float(np.mean(lat)) if lat else 0.0, fps, frames_total, net, now - t0, err)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= 60:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("r"): frames_total = 0; t0 = time.time()
    finally:
        cap.release(); cv2.destroyAllWindows()
        if det: det.close()


if __name__ == "__main__":
    main()
