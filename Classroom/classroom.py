"""
classroom.py —— 教室儀表板:一個視窗同時顯示 人數 / 舉手(含九宮格位置)/ 分心訊號(低頭、手機、趴下、轉頭)/ 專注度走勢。

模型:yolov8m_pose(骨架 → 舉手、低頭、趴下、轉頭)+ yolov11m(person 計數、cell phone),兩者交錯逐幀跑。
兩個模型 Hailo-8L 也都有,UGen200 同樣可用。不存影像、不辨識身分;追蹤編號只是暫時的,人離開就消失。

按鍵:1/2/3/4 開關 低頭/手機/趴下/轉頭 規則 · 空白鍵 凍結(計時器同步暫停)· f 鏡像 · s 存畫面 PNG · r 重設 · q 或 Esc 離開
用法:python classroom.py [--source auto] [--fullscreen] [--flip]
      python classroom.py --image classroom.jpg          (靜態合照測試)
      python classroom.py --selftest / --snapshot out.png [--snapshot-frames 40]
"""
import argparse
import os
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import attention as A
from hailo_detect import ObjectDetector
from hailo_pose import PoseEstimator
from tracker import CentroidTracker
from ui_text import draw_text, badge, text_width
from camera import open_camera

WIN = "UGen300 Classroom"
W, H, PANEL = 1280, 720, 440
HERE = os.path.dirname(os.path.abspath(__file__))
C_BG, C_PANEL, C_TXT, C_DIM, C_LINE = (24, 24, 28), (36, 36, 42), (240, 240, 240), (150, 150, 150), (70, 70, 80)
C_OK, C_RAISE, C_DOWN, C_PHONE, C_SLEEP, C_TURN, C_ACC = (80, 220, 120), (255, 160, 40), (60, 200, 255), (60, 60, 230), (140, 140, 140), (200, 90, 200), (255, 170, 40)
STATE_COLOR = {"ok": C_OK, "head_down": C_DOWN, "phone": C_PHONE, "sleeping": C_SLEEP, "turned": C_TURN}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6)]
PHONE_CONF, ERR_FATAL, ERR_CLEAR_SEC, ATT_WINDOW, LABEL_MAX_PEOPLE = 0.5, 5, 3.0, 5.0, 8


def draw_people(fr, s, tracks_kps, states, phones):
    many = len(tracks_kps) > LABEL_MAX_PEOPLE      # 人多時:專心的人只畫細框,有狀態的才畫骨架與字(避免一團線)
    for tid, kps in tracks_kps.items():
        st = states.get(tid); state = st.state if st else "ok"; col = STATE_COLOR.get(state, C_OK)
        flagged = st is not None and (st.raised or state != "ok")
        if flagged: col = C_RAISE if (st.raised and state == "ok") else col
        if not many or flagged:
            for a, b in SKELETON:
                if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                    cv2.line(fr, (int(kps[a][0] * s), int(kps[a][1] * s)), (int(kps[b][0] * s), int(kps[b][1] * s)), col, 2)
        if st and st.box:
            x1, y1, x2, y2 = [int(v * s) for v in st.box]
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2 if (not many or flagged) else 1)
            tag = (("舉手 " if st.raised else "") + A.RULE_NAMES.get(state, "")).strip()
            if tag and (not many or flagged):
                ty = max(0, y1 - 36); tw = text_width(tag, 26)
                cv2.rectangle(fr, (x1, ty), (x1 + tw + 12, ty + 34), (20, 20, 24), -1)
                draw_text(fr, tag, (x1 + 6, ty + 2), 26, col, shadow=False)
    for x1, y1, x2, y2 in phones:
        cv2.rectangle(fr, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), C_PHONE, 2)


def draw_grid_map(canvas, x, y, grid, cell=46):
    for r in range(3):
        for c in range(3):
            n = grid[r][c]; x0, y0 = x + c * cell, y + r * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 3, y0 + cell - 3), (60, 60, 68) if n == 0 else (120, 80, 30), -1)
            if n: draw_text(canvas, str(n), (x0 + cell // 2 - 1, y0 + cell // 2 - 1), 26, C_RAISE, anchor="mm", shadow=False)
    for r in range(3): draw_text(canvas, A.GRID_ROW[r], (x + 3 * cell + 4, y + r * cell + 12), 17, C_DIM)
    for c in range(3): draw_text(canvas, A.GRID_COL[c], (x + c * cell + cell // 2 - 1, y + 3 * cell + 2), 17, C_DIM, anchor="ma")


def compose(frame, tracks_kps, states, phones, count, summary, enabled, history, ui, fatal=None):
    canvas = np.full((H, W, 3), C_BG, np.uint8); vw = W - PANEL
    h, w = frame.shape[:2]; sc = min(vw / w, H / h); fr = cv2.resize(frame, (int(w * sc), int(h * sc)))
    if not fatal: draw_people(fr, sc, tracks_kps, states, phones)
    x0, y0 = (vw - fr.shape[1]) // 2, (H - fr.shape[0]) // 2; canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    cx, cy = x0 + fr.shape[1] // 2, y0 + fr.shape[0] // 2
    if fatal:
        cv2.rectangle(canvas, (x0, y0), (x0 + fr.shape[1], y0 + fr.shape[0]), (20, 20, 90), -1)
        draw_text(canvas, "無法啟動", (cx, cy - 60), 60, C_PHONE, anchor="mm")
        for i, line in enumerate(fatal.split("\n")[:4]): draw_text(canvas, line[:40], (cx, cy + 10 + i * 34), 24, C_TXT, anchor="mm")
    if ui.get("warn"): draw_text(canvas, ui["warn"], (x0 + 16, y0 + fr.shape[0] - 40), 22, C_PHONE)
    if ui.get("frozen"):
        cv2.rectangle(canvas, (cx - 170, y0 + fr.shape[0] - 52), (cx + 170, y0 + fr.shape[0] - 8), (20, 20, 24), -1)
        draw_text(canvas, "已凍結 · 空白鍵恢復", (cx, y0 + fr.shape[0] - 30), 28, C_ACC, anchor="mm", shadow=False)
    draw_text(canvas, "鏡像" if ui.get("flip") else "鏡頭視角", (x0 + fr.shape[1] - 12, y0 + fr.shape[0] - 30), 17, C_DIM, anchor="ra")
    # 右側面板
    px = vw; cv2.rectangle(canvas, (px, 0), (W, H), C_PANEL, -1); badge(canvas)
    draw_text(canvas, "教室儀表板", (px + 24, 40), 36, C_TXT)
    draw_text(canvas, "不存影像、不認人,只出統計數字", (px + 24, 88), 20, C_OK)
    draw_text(canvas, "YOLO 姿態 + 偵測 · UGen200 亦可執行", (px + 24, 114), 15, C_DIM)
    # 人數
    count_txt = f"{count}" if count else "--"
    draw_text(canvas, count_txt, (px + 24, 128), 96, C_OK if count else C_DIM)
    draw_text(canvas, "人", (px + 24 + len(count_txt) * 54 + 6, 182), 30, C_DIM)
    # 舉手 + 九宮格
    draw_text(canvas, f"舉手 {summary['raised']}", (px + 24, 252), 30, C_RAISE if summary["raised"] else C_TXT)
    draw_text(canvas, "舉手位置", (px + 24, 292), 15, C_DIM)
    draw_grid_map(canvas, px + 24, 312, summary["grid"])
    # 分心訊號
    bx = px + 214; inatt = min(summary["inattentive"], count)
    draw_text(canvas, f"分心 {inatt}", (bx, 252), 30, C_PHONE if inatt else C_TXT)
    yy = 296
    for i, k in enumerate(A.RULES, 1):
        on = enabled.get(k, True); col = STATE_COLOR[k] if on else (80, 80, 88)
        draw_text(canvas, f"[{i}]", (bx, yy + 4), 15, (110, 110, 118)); draw_text(canvas, A.RULE_NAMES[k], (bx + 34, yy), 22, col)
        draw_text(canvas, f"{summary['by'][k]}" if on else "已關閉", (W - 24, yy + 2), 20 if on else 16, col, anchor="ra")
        yy += 34
    # 專注度(最近 ATT_WINDOW 秒中位數)+ 走勢
    att = ui.get("att")
    if att is None: draw_text(canvas, "專注度 --", (px + 24, 478), 34, C_DIM)
    else: draw_text(canvas, f"專注度 {att}%", (px + 24, 478), 34, C_OK if att >= 80 else C_ACC if att >= 60 else C_PHONE)
    gx0, gy0, gw, gh = px + 24, 528, PANEL - 48, 84
    cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (50, 50, 58), -1)
    cv2.line(canvas, (gx0, gy0 + 2), (gx0 + gw, gy0 + 2), (70, 90, 76), 1)
    if len(history) >= 2:
        t_now = history[-1][0]
        pts = [(int(gx0 + gw * (1 - (t_now - t) / 60.0)), int(gy0 + gh - 2 - (gh - 4) * v / 100)) for t, v in history if t_now - t <= 60]
        for i in range(1, len(pts)): cv2.line(canvas, pts[i - 1], pts[i], C_OK, 2)
    draw_text(canvas, "最近 60 秒專注度", (gx0, gy0 + gh + 4), 16, C_DIM)
    draw_text(canvas, f"姿態 {ui.get('ms_pose', 0):.0f} ms · 偵測 {ui.get('ms_det', 0):.0f} ms", (px + 24, H - 84), 16, C_DIM)
    draw_text(canvas, "UGen200 亦可執行", (W - 24, H - 84), 16, C_DIM, anchor="ra")
    draw_text(canvas, "1-4 開關  空白鍵 凍結  f 鏡像  r 重設  q 離開", (px + 24, H - 56), 17, C_DIM)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", help="auto=外接優先,否則內建;或指定編號 0/1")
    ap.add_argument("--pose-hef", default="yolov8m_pose.hef"); ap.add_argument("--det-hef", default="yolov11m.hef")
    ap.add_argument("--conf", type=float, default=0.3); ap.add_argument("--flip", action="store_true", help="鏡像(自己對著筆電測試時用;鏡頭朝教室時不用)")
    ap.add_argument("--image", default="", help="用靜態照片代替鏡頭(教室合照測試用)")
    ap.add_argument("--fullscreen", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--snapshot", default=""); ap.add_argument("--snapshot-frames", type=int, default=40)
    args = ap.parse_args()
    fatal = None; pose = det = None
    try:
        pose = PoseEstimator(args.pose_hef, conf_threshold=args.conf); det = ObjectDetector(args.det_hef, conf_threshold=args.conf)
    except Exception as e:  # noqa: BLE001
        fatal = f"模型載入失敗:{type(e).__name__}\n請確認 UGen300 已插上,且沒有其他 demo 正在使用它"; print("[模型]", repr(e))
    still = None; cap = None
    if args.image:
        still = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if still is None: print("讀不到圖片:", args.image); return
    else:
        try: cap = open_camera(args.source)[0]
        except (ValueError, TypeError) as e: fatal = fatal or f"鏡頭參數錯誤:{e}"; cap = cv2.VideoCapture()
        if not cap.isOpened() and not fatal: fatal = "找不到可用鏡頭\n請關閉其他正在用鏡頭的程式(Teams、相機)後重開"
    if args.selftest:
        ok, f = (True, still) if still is not None else (cap.read() if cap.isOpened() else (False, None))
        people = pose.infer_multi(f) if (ok and pose) else []; dets = det.infer(f) if (ok and det) else []
        print(f"[selftest] 鏡頭={'OK' if ok else 'FAIL'} 模型={'OK' if (pose and det) else 'FAIL'} 骨架={len(people)} "
              f"person={sum(l == 'person' for l, _, _ in dets)} phone={sum(l == 'cell phone' and s >= PHONE_CONF for l, s, _ in dets)}"); return
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) if args.fullscreen else cv2.resizeWindow(WIN, W, H)
    enabled = {k: True for k in A.RULES}; tracker = CentroidTracker(max_missed=15, iou_thresh=0.2, dist_thresh=160)
    states = {}; people = []; dets = []; phones = []; count_hist = deque(maxlen=5); history = deque(); att_hist = deque()
    frozen = None; freeze_t = 0.0; frames = 0; last_summary = A.summarize({}, W, H); count = 0; tracks_kps = {}
    n_err = 0; err_at = 0.0; ui = dict(warn="", frozen=False, flip=args.flip, ms_pose=0.0, ms_det=0.0, att=None)
    try:
        while True:
            if still is not None: ok, frame = True, still.copy()
            else: ok, frame = cap.read(); frame = frame if ok else np.zeros((720, 1280, 3), np.uint8)
            if ui["flip"] and ok: frame = cv2.flip(frame, 1)
            now = time.time()
            if frozen is not None: frame = frozen
            elif pose and det and not fatal:
                try:
                    t = time.time()
                    if frames % 2 == 0: people = pose.infer_multi(frame) or []; ui["ms_pose"] = 0.8 * ui["ms_pose"] + 0.2 * (time.time() - t) * 1000
                    else: dets = det.infer(frame); ui["ms_det"] = 0.8 * ui["ms_det"] + 0.2 * (time.time() - t) * 1000
                    n_err = 0
                except Exception as e:  # noqa: BLE001  單次抖動不該讓 demo 永久停擺
                    n_err += 1; err_at = now; ui["warn"] = f"推論失敗({n_err}):{type(e).__name__}"; print("[推論]", repr(e))
                    if n_err >= ERR_FATAL: fatal = f"UGen300 連續推論失敗 {n_err} 次\n請重新插拔後重開程式"
                phones = [b for l, s, b in dets if l == "cell phone" and s >= PHONE_CONF]
                n_person = sum(1 for l, _, _ in dets if l == "person")
                count_hist.append(max(n_person, len(people))); count = int(np.median(count_hist))
                # 追蹤:用「頭肩框」給 ID(手伸出去不會讓框劇變),完整框留給手機歸屬與九宮格
                boxed = [(A.bbox_from_kps(k, idx=A.HEAD_SHOULDER), A.bbox_from_kps(k), k) for k in people]
                boxed = [(hb, fb, k) for hb, fb, k in boxed if hb and fb]
                tracks = tracker.update([("person", 1.0, hb) for hb, _, _ in boxed])
                hb2 = {hb: (fb, k) for hb, fb, k in boxed}; tracks_kps = {}; persons = []
                for tr in tracks:
                    if tr.missed == 0 and tr.box in hb2:
                        fb, kps = hb2[tr.box]; tracks_kps[tr.id] = (fb, kps)
                        persons.append((tr.id, fb, A.shoulder_line(kps)[0]))
                my_phones = A.assign_phones(phones, persons)
                for tid, (fb, kps) in tracks_kps.items():
                    states.setdefault(tid, A.PersonState()).update(kps, my_phones.get(tid, []), enabled, now, box=fb, frame_h=frame.shape[0])
                alive = {tr.id for tr in tracks}
                for tid in list(states):
                    if tid not in alive: del states[tid]
                tracks_kps = {tid: kps for tid, (fb, kps) in tracks_kps.items()}
                last_summary = A.summarize({t: s for t, s in states.items() if t in tracks_kps}, frame.shape[1], frame.shape[0])
                n_judged = len(tracks_kps)          # 專注度只看「有骨架、真的被判斷過」的人,不拿 YOLO 人數當分母(後排沒骨架不算專心)
                if n_judged > 0:
                    att_now = 100 * max(0, n_judged - min(last_summary["inattentive"], n_judged)) / n_judged
                    att_hist.append((now, att_now))
                    while att_hist and now - att_hist[0][0] > ATT_WINDOW: att_hist.popleft()
                    ui["att"] = int(round(float(np.median([v for _, v in att_hist])))); history.append((now, ui["att"]))
                else:
                    ui["att"] = None; att_hist.clear()
                while history and now - history[0][0] > 60: history.popleft()
            if ui["warn"] and now - err_at > ERR_CLEAR_SEC: ui["warn"] = ""
            canvas = compose(frame, tracks_kps, states, phones, count, last_summary, enabled, history, ui, fatal)
            cv2.imshow(WIN, canvas); frames += 1
            if args.snapshot and frames >= args.snapshot_frames:
                cv2.imwrite(args.snapshot, canvas); print("[snapshot]", args.snapshot); break
            k = cv2.waitKey(1) & 0xFF
            key = chr(k).lower() if 32 <= k < 127 else ("esc" if k == 27 else "")
            if key in ("q", "esc"): break
            if not args.fullscreen and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break   # 按了視窗的 X
            if fatal: continue
            if key in ("1", "2", "3", "4"): rk = A.RULES[int(key) - 1]; enabled[rk] = not enabled[rk]
            if key == " ":
                if frozen is None: frozen = frame.copy(); freeze_t = now; ui["frozen"] = True
                else:
                    dt = time.time() - freeze_t
                    for s in states.values(): s.shift(dt)      # 凍結期間不算持續時間
                    frozen = None; ui["frozen"] = False
            if key == "f": ui["flip"] = not ui["flip"]
            if key == "r": history.clear(); att_hist.clear(); count_hist.clear(); states.clear(); tracker = CentroidTracker(max_missed=15, iou_thresh=0.2, dist_thresh=160)
            if key == "s":
                p = os.path.join(HERE, f"classroom_{datetime.now():%Y%m%d_%H%M%S}.png"); cv2.imwrite(p, canvas); print("[存圖]", p)
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()


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
