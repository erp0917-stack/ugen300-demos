# -*- coding: utf-8 -*-
"""
missing_check.py  ——  #5 漏件防呆（開場記基準 / 少件紅閃 / 多件黃閃 / 雙音效）
================================================================================
⚠️ 放置位置：這支檔請放在「漏件防呆」資料夾內，跟 Y1 副本的
   hailo_detect.py / coco_labels.py / yolov8m.hef 放在一起跑。
   ❗ 不要複製回原本的 Y1 資料夾、也不要覆蓋 Y1 的 main.py。
      Y1（保命 demo）永遠維持原樣，這支是獨立的 #5。

操作流程：
   1. 鏡頭照著桌面，把要看守的東西擺好（例：滑鼠＋手機），其他東西移出鏡頭。
   2. 按「空白鍵」→ 程式把此刻看到的物品記起來，當成「應有清單」基準。
   3. 之後：
        - 有東西被拿走（少件）→ 紅框閃 + 低音長嗶，列出少了什麼。
        - 多放了基準以外的東西（多件）→ 黃框閃 + 高音短嗶兩聲，列出多了什麼。
        - 又少又多時：紅色優先顯示。
   4. 東西恢復原樣 → 警報解除，回到綠色 ALL OK。
   5. 想重設基準：隨時再按一次「空白鍵」。

鏡頭建議：筆電內建鏡頭照不到桌面，請用外接 USB webcam（如 BRIO）
         架高往下俯瞰桌面。跑的時候用 --source 1 指定外接鏡頭。

跑法：
   先 cd 到「漏件防呆」資料夾，然後：
       python missing_check.py --source 1
       --source 1 = 外接 webcam（內建鏡頭是 0）
       --debug    = 終端機印出實際偵測到的物件名（對不上時用來查）

操作鍵：空白鍵 = 設定/重設基準   ｜   q = 離開
"""

import argparse
import time
import threading
from collections import deque

import cv2

from hailo_detect import ObjectDetector

# Windows 內建嗶聲（不用裝套件、不用外接喇叭）。非 Windows 則自動跳過。
try:
    import winsound
    _HAS_BEEP = True
except Exception:
    _HAS_BEEP = False


# 最近這麼多幀內有看到就算「在場」（消除單幀閃爍，判定才穩）
PRESENCE_WINDOW = 8

# 警報節流：每隔幾秒響一次（避免一直響到吵）
ALARM_INTERVAL = 1.2

# 兩種音效（靠頻率高低、長短、次數做出區別）
MISS_FREQ, MISS_MS = 600, 350          # 少件：低音、長 → 一聲
EXTRA_FREQ, EXTRA_MS = 1100, 120       # 多件：高音、短 → 兩聲

# 顏色 (B, G, R)
C_OK    = (90, 200, 90)      # 綠
C_MISS  = (60, 60, 230)      # 紅（少件）
C_EXTRA = (40, 200, 240)     # 黃（多件）
C_PANEL = (32, 32, 40)       # 面板深底
C_TEXT  = (235, 235, 235)    # 淺字
C_INFO  = (0, 212, 216)      # FPS 字色


def _norm(s):
    """類別名正規化：小寫、去空白，讓 'cell phone' / 'cellphone' 都能對上。"""
    return "".join(str(s).lower().split())


def _beep_miss():
    """少件音：低音長嗶一聲（背景執行緒，不卡畫面）。"""
    if not _HAS_BEEP:
        return
    def run():
        try:
            winsound.Beep(MISS_FREQ, MISS_MS)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def _beep_extra():
    """多件音：高音短嗶兩聲（背景執行緒，不卡畫面）。"""
    if not _HAS_BEEP:
        return
    def run():
        try:
            winsound.Beep(EXTRA_FREQ, EXTRA_MS)
            time.sleep(0.06)
            winsound.Beep(EXTRA_FREQ, EXTRA_MS)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def draw_alert_icon(frame, kind, cx, cy, size=48):
    """畫大警示圖案：kind='miss' 紅三角驚嘆號；kind='extra' 黃圓加號。"""
    import numpy as np
    if kind == "miss":
        col = C_MISS
        # 警告三角形（實心紅 + 白邊）
        pts = np.array([[cx, cy - size],
                        [cx - size, cy + size],
                        [cx + size, cy + size]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], col)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 3, cv2.LINE_AA)
        # 中間驚嘆號
        cv2.line(frame, (cx, cy - size//2), (cx, cy + size//4),
                 (255, 255, 255), 6, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy + size//2), 5, (255, 255, 255), -1, cv2.LINE_AA)
    else:  # extra
        col = C_EXTRA
        cv2.circle(frame, (cx, cy), size, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), size, (0, 0, 0), 3, cv2.LINE_AA)
        # 中間加號（黑）
        cv2.line(frame, (cx - size//2, cy), (cx + size//2, cy), (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - size//2), (cx, cy + size//2), (0, 0, 0), 7, cv2.LINE_AA)


def draw(frame, baseline, present_map, boxes_map, extras, fps, armed, flash_on):
    """
    baseline    : 應有清單 [(coco, show), ...]；尚未設定時為空 list
    present_map : {coco: True/False} 該基準項目目前是否在場
    boxes_map   : {coco: ((x1,y1,x2,y2), score)} 基準項目這幀的框（沒有則 None）
    extras      : [(show_label, (x1,y1,x2,y2), score), ...] 多出來（基準以外）的東西
    armed       : 是否已設定基準、進入看守狀態
    flash_on    : 這一幀閃爍要不要亮
    """
    h, w = frame.shape[:2]
    missing = [c for c, _ in baseline if not present_map.get(c, False)]
    has_extra = armed and len(extras) > 0
    all_ok = armed and len(missing) == 0 and not has_extra

    # --- 基準項目在場：畫綠框 ---
    for coco, show in baseline:
        b = boxes_map.get(coco)
        if b is not None:
            (x1, y1, x2, y2), score = b
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_OK, 3)
            tag = "{} {:.0%}".format(show, score)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), C_OK, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- 多出來的東西：畫黃框 + "EXTRA" 標籤 ---
    if armed:
        for show, (x1, y1, x2, y2), score in extras:
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_EXTRA, 3)
            tag = "EXTRA: {}".format(show)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), C_EXTRA, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- 邊框閃爍：少件紅色優先，否則多件黃色 ---
    if armed and flash_on:
        if missing:
            cv2.rectangle(frame, (4, 4), (w - 5, h - 5), C_MISS, 14)
        elif has_extra:
            cv2.rectangle(frame, (4, 4), (w - 5, h - 5), C_EXTRA, 14)

    # --- 大警示圖案（畫在攝影區中央上方，不蓋到右側面板）---
    if armed:
        cam_w = max(0, w - 250)          # 面板寬 250
        icx, icy = cam_w // 2, 110
        if missing:
            draw_alert_icon(frame, "miss", icx, icy, 50)
        elif has_extra:
            draw_alert_icon(frame, "extra", icx, icy, 46)

    # --- 右側半透明面板 ---
    panel_w = 250
    px = max(0, w - panel_w)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (w, h), C_PANEL, -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, "PARTS CHECK", (px + 16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2)

    if not armed:
        cv2.putText(frame, "Place items,", (px + 16, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_INFO, 2)
        cv2.putText(frame, "press SPACE", (px + 16, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_INFO, 2)
        cv2.putText(frame, "to set baseline", (px + 16, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)
    else:
        y = 78
        for coco, show in baseline:
            ok = present_map.get(coco, False)
            color = C_OK if ok else C_MISS
            mark = "OK" if ok else "X "
            cv2.putText(frame, "[{}] {}".format(mark, show), (px + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 34
        # 多件列在清單下方（黃字）
        for show, _b, _s in extras:
            cv2.putText(frame, "[+] {}".format(show), (px + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_EXTRA, 2)
            y += 34

        # 底部大狀態列
        sy = h - 70
        if missing:
            banner, txt, tcol = C_MISS, "MISSING {}".format(len(missing)), (255, 255, 255)
        elif has_extra:
            banner, txt, tcol = C_EXTRA, "EXTRA {}".format(len(extras)), (0, 0, 0)
        else:
            banner, txt, tcol = C_OK, "ALL OK", (0, 0, 0)
        cv2.rectangle(frame, (px, sy), (w, h), banner, -1)
        cv2.putText(frame, txt, (px + 24, sy + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, tcol, 3)

    # --- 左上角 FPS（不蓋到面板）---
    cv2.rectangle(frame, (0, 0), (px, 38), (26, 26, 46), -1)
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_INFO, 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8m.hef", help="偵測模型檔（沿用 Y1 的）")
    ap.add_argument("--source", default="auto",
                    help="auto 自動找鏡頭（外接優先，預設）/ 0 筆電 / 1 外接 / 影片檔")
    ap.add_argument("--conf", type=float, default=0.4, help="信心度門檻（小物件可調低）")
    ap.add_argument("--debug", action="store_true", help="終端機印出實際偵測到的物件名")
    args = ap.parse_args()

    # 開鏡頭：auto 會依序試外接(1)再筆電(0)，挑第一個開得起來又拍得到畫面的
    cap = None
    if str(args.source).lower() == "auto":
        for idx in (1, 0):
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                ok, _ = c.read()
                if ok:
                    where = "外接 webcam" if idx == 1 else "筆電內建鏡頭"
                    print("[missing_check] 自動選用鏡頭 #{}（{}）".format(idx, where))
                    cap = c
                    break
                c.release()
        if cap is None:
            print("[missing_check] 找不到可用鏡頭，請確認 webcam 已接上。")
            return
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[missing_check] 無法開啟影像來源：{}".format(args.source))
            print("       外接 webcam 通常是 1（內建鏡頭是 0），試試 --source 1")
            return

    det = ObjectDetector(args.hef, conf_threshold=args.conf)

    baseline = []          # [(coco, show), ...]
    baseline_norm = set()  # 基準項目的正規化名稱集合（用來判斷「多件」）
    history = {}           # {coco: deque} 每個基準項目的在場歷史

    print("[missing_check] 開始。鏡頭照好桌面、擺好東西後，按【空白鍵】設基準；q 離開。")
    prev = time.time()
    last_debug = 0.0
    last_miss_beep = 0.0
    last_extra_beep = 0.0
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            detections = det.infer(frame)

            # 這一幀：名稱 -> 信心度最高的 (原始名, 框, 分數)
            frame_best = {}
            detected_labels = []
            for label, score, box in detections:
                detected_labels.append(label)
                n = _norm(label)
                if n not in frame_best or score > frame_best[n][2]:
                    frame_best[n] = (label, box, score)

            present_map = {}
            boxes_map = {}
            extras = []
            if baseline:
                # 基準項目在不在
                for coco, show in baseline:
                    n = _norm(coco)
                    seen = n in frame_best
                    history[coco].append(seen)
                    present_map[coco] = any(history[coco])
                    boxes_map[coco] = (frame_best[n][1], frame_best[n][2]) if seen else None
                # 多出來的東西 = 這幀看到、但不在基準裡的
                for n, (label, box, score) in frame_best.items():
                    if n not in baseline_norm:
                        extras.append((label, box, score))

            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
            prev = now

            armed = len(baseline) > 0
            missing = [c for c, _ in baseline if not present_map.get(c, False)]
            has_extra = armed and len(extras) > 0
            flash_on = (int(now * 3) % 2 == 0)

            # 音效：少件低音、多件高音；各自節流
            if armed and missing and (now - last_miss_beep) > ALARM_INTERVAL:
                _beep_miss()
                last_miss_beep = now
            if armed and has_extra and not missing and (now - last_extra_beep) > ALARM_INTERVAL:
                _beep_extra()
                last_extra_beep = now

            if args.debug and (now - last_debug) > 1.0:
                print("[debug] 偵測到：{}".format(sorted(set(detected_labels))))
                last_debug = now

            frame = draw(frame, baseline, present_map, boxes_map, extras, fps, armed, flash_on)
            cv2.imshow("UGen300 Missing-Item Check (Space = set baseline, q = quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                baseline = []
                baseline_norm = set()
                history = {}
                for n, (label, box, score) in frame_best.items():
                    baseline.append((label, label))
                    baseline_norm.add(n)
                    history[label] = deque(maxlen=PRESENCE_WINDOW)
                names = [c for c, _ in baseline]
                if names:
                    print("[missing_check] 已鎖定基準：{}".format(names))
                else:
                    print("[missing_check] 目前畫面沒偵測到任何東西，基準是空的；擺好東西再按一次空白鍵。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        det.close()
        print("[missing_check] 已結束。")


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
