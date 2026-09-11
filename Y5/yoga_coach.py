# -*- coding: utf-8 -*-
"""
yoga_coach.py  ——  瑜伽姿勢維持指導（Y5）
=========================================================
隨機出題（一輪內不重複）、擺對維持滿秒過關、自動換下一個。
右側大火柴人示意圖 + 箭頭 + 英文分步說明；擺不對時即時提示「還差哪一步」。

⚠️ 畫面一律英文（OpenCV 畫不出中文會亂碼）。已移除全身入鏡門禁。
操作：按 q 離開、按 n 跳下一題。
跑法：python yoga_coach.py --hef yolov8s_pose.hef --source 0
"""

import argparse
import cv2

from hailo_pose import PoseEstimator
from yoga_logic import YogaCoach, RandomCoach, POSES
from victory_sound import play_victory

SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

# ── 火柴人示意圖座標 (0~1) ───────────────────────────────────────────
_ICON_POSES = {
    "tree": {"head":(.50,.12),"neck":(.50,.24),"shL":(.40,.26),"shR":(.60,.26),
        "elL":(.40,.13),"elR":(.60,.13),"wrL":(.42,.02),"wrR":(.58,.02),
        "hipC":(.50,.55),"hipL":(.44,.56),"hipR":(.56,.56),
        "knL":(.44,.76),"knR":(.56,.76),"anL":(.44,.97),"anR":(.56,.97)},
    "warrior": {"head":(.50,.12),"neck":(.50,.24),"shL":(.40,.27),"shR":(.60,.27),
        "elL":(.26,.27),"elR":(.74,.27),"wrL":(.10,.27),"wrR":(.90,.27),
        "hipC":(.50,.55),"hipL":(.43,.56),"hipR":(.57,.56),
        "knL":(.30,.74),"knR":(.70,.74),"anL":(.20,.97),"anR":(.80,.97)},
    "warrior1": {"head":(.50,.12),"neck":(.50,.24),"shL":(.42,.26),"shR":(.58,.26),
        "elL":(.42,.13),"elR":(.58,.13),"wrL":(.45,.02),"wrR":(.55,.02),
        "hipC":(.50,.54),"hipL":(.45,.55),"hipR":(.55,.55),
        "knL":(.34,.70),"anL":(.32,.96),"knR":(.66,.74),"anR":(.86,.94)},
    "triangle": {"head":(.40,.14),"neck":(.43,.26),"shL":(.36,.30),"shR":(.50,.24),
        "elR":(.52,.12),"wrR":(.55,.02),"elL":(.34,.45),"wrL":(.30,.62),
        "hipC":(.55,.55),"hipL":(.48,.56),"hipR":(.62,.55),
        "knL":(.34,.74),"knR":(.78,.74),"anL":(.24,.96),"anR":(.86,.95)},
    "squat": {"head":(.50,.20),"neck":(.50,.30),"shL":(.40,.33),"shR":(.60,.33),
        "elL":(.42,.40),"elR":(.58,.40),"wrL":(.48,.40),"wrR":(.52,.40),
        "hipC":(.50,.62),"hipL":(.42,.62),"hipR":(.58,.62),
        "knL":(.30,.72),"knR":(.70,.72),"anL":(.33,.95),"anR":(.67,.95)},
    "tree1": {"head":(.50,.12),"neck":(.50,.24),"shL":(.40,.26),"shR":(.60,.26),
        "elL":(.40,.13),"elR":(.60,.13),"wrL":(.42,.02),"wrR":(.58,.02),
        "hipC":(.50,.55),"hipL":(.44,.56),"hipR":(.56,.56),
        "knR":(.56,.76),"anR":(.56,.97),"knL":(.36,.64),"anL":(.50,.72)},
}
_ICON_BONES = [("head","neck"),("neck","shL"),("neck","shR"),
               ("shL","elL"),("elL","wrL"),("shR","elR"),("elR","wrR"),
               ("neck","hipC"),("hipC","hipL"),("hipC","hipR"),
               ("hipL","knL"),("knL","anL"),("hipR","knR"),("knR","anR")]

# 每個姿勢的「動作箭頭」：(起點, 終點) 標出關鍵部位要往哪動（黃色箭頭）
_ICON_ARROWS = {
    "tree":     [((.42,.20),(.42,.04)), ((.58,.20),(.58,.04))],          # 手往上
    "warrior":  [((.30,.27),(.12,.27)), ((.70,.27),(.88,.27))],          # 手往兩側
    "warrior1": [((.45,.18),(.45,.04)), ((.40,.62),(.34,.70))],          # 手往上 + 前膝彎
    "triangle": [((.53,.14),(.55,.03)), ((.32,.50),(.30,.62))],          # 一手上一手下
    "squat":    [((.40,.55),(.34,.70)), ((.60,.55),(.66,.70)), ((.50,.34),(.50,.40))],  # 膝往下蹲
    "tree1":    [((.42,.20),(.42,.04)), ((.46,.62),(.50,.72))],          # 手往上 + 抬腳
}


def draw_pose_icon(frame, pose_key, box, color=(255,255,255), thick=8):
    pts = _ICON_POSES.get(pose_key)
    if not pts:
        return
    x0, y0, bw, bh = box
    def P(p):
        return (int(x0+p[0]*bw), int(y0+p[1]*bh))
    for a, b in _ICON_BONES:
        if a in pts and b in pts:
            cv2.line(frame, P(pts[a]), P(pts[b]), color, thick, cv2.LINE_AA)
    hx, hy = P(pts["head"])
    cv2.circle(frame, (hx, hy), max(10, int(bw*0.10)), color, thick, cv2.LINE_AA)
    # 黃色動作箭頭
    for s, e in _ICON_ARROWS.get(pose_key, []):
        cv2.arrowedLine(frame, P(s), P(e), (0, 230, 255), max(3, thick-3),
                        cv2.LINE_AA, tipLength=0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8s_pose.hef")
    ap.add_argument("--source", default="0")
    ap.add_argument("--pose", default=None, choices=list(POSES.keys()))
    ap.add_argument("--hold", type=float, default=8.0)
    ap.add_argument("--switch-delay", type=float, default=2.0)
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[yoga] 無法開啟影像來源：{args.source}（外接 webcam 試 --source 1）")
        return

    pose = PoseEstimator(args.hef)
    WIN = "Yoga Coach (n=skip / q=quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    random_mode = args.pose is None
    coach = RandomCoach(hold_target=args.hold, switch_delay=args.switch_delay) if random_mode \
            else YogaCoach(pose_name=args.pose, hold_target=args.hold)
    print(f"[yoga] 啟動，維持 {args.hold:.0f} 秒。模式={'隨機' if random_mode else '單一'}。q 離開。")

    frame_idx = 0
    victory_played = False   # 只在「剛完成一輪」時響一次過關音效，避免每幀重複
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame = cv2.flip(frame, 1)
            keypoints = pose.infer(frame)
            state = coach.update(keypoints)

            # ── 完成一輪：恭喜結束畫面（按 q 離開、n/空白鍵重玩）──
            if random_mode and state.get("finished"):
                # 剛進入完成狀態 → 響一次過關音效（背景播，不卡畫面）
                if not victory_played:
                    play_victory()
                    victory_played = True
                h, w = frame.shape[:2]
                ov = frame.copy()
                cv2.rectangle(ov, (0,0), (w,h), (15,20,15), -1)
                frame = cv2.addWeighted(ov, 0.72, frame, 0.28, 0)
                # 慶祝圖示：一個雙手高舉的火柴人（綠色）
                bw, bh = int(w*0.18), int(h*0.34)
                draw_pose_icon(frame, "tree", (w//2 - bw//2, int(h*0.10), bw, bh),
                               color=(0,230,0), thick=8)
                lines = [("CONGRATULATIONS!", 1.6, (0,230,0), 4),
                         (f"All {coach.total} poses completed!", 1.0, (255,255,255), 3),
                         ("Great job!", 0.9, (0,212,216), 2),
                         ("press N = play again    Q = quit", 0.7, (200,200,200), 2)]
                yy = int(h*0.56)
                for txt, sc, c, th in lines:
                    (tw,_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
                    cv2.putText(frame, txt, ((w-tw)//2, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
                    yy += int(58*sc)
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("n"), ord(" ")):
                    coach.restart()
                    victory_played = False   # 重玩 → 下次完成時再響一次
                    print("[yoga] 重玩一輪")
                continue

            cur_key = state.get("pose_name") if random_mode else coach.pose_name

            # 真人骨架
            col = (0,220,0) if (state["in_pose"] or state["done"]) else (255,180,0)
            if keypoints:
                for a, b in SKELETON:
                    xa, ya, ca = keypoints[a]; xb, yb, cb = keypoints[b]
                    if ca >= 0.5 and cb >= 0.5:
                        cv2.line(frame, (int(xa),int(ya)), (int(xb),int(yb)), col, 3)

            h, w = frame.shape[:2]
            # 上方資訊條（再壓矮版：40px）
            cv2.rectangle(frame, (0,0), (w,40), (26,26,46), -1)
            cv2.putText(frame, "Target: " + state["label"], (10,16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,212,216), 1)
            if state["done"]:
                if random_mode:
                    status = f"DONE!  Next pose in {state.get('celebrate_remain',0):.0f}..."
                else:
                    status = "DONE!  (press n for next)"
                scol = (0,220,0)
            elif state["in_pose"]:
                status, scol = f"Hold it!  {state['held']:.1f}/{state['target']:.0f}s", (0,220,0)
            else:
                # 即時回饋：告訴使用者還差哪一步
                status, scol = ">> " + (state.get("hint") or "Get into pose"), (0,200,255)
            cv2.putText(frame, status, (10,33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, scol, 1)
            if random_mode:
                dt = f"Done: {state.get('completed',0)}"
                (tw,_), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.putText(frame, dt, (w-tw-10,16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,220,0), 1)
            if state["in_pose"] and not state["done"]:
                ratio = min(1.0, state["held"]/state["target"])
                cv2.rectangle(frame, (10,36), (10+int((w-20)*ratio),38), (0,220,0), -1)

            # 右側面板：火柴人 + 箭頭 + 英文分步說明（再窄版：20% 寬）
            panel_w = max(190, int(w*0.20)); panel_h = h - 48
            px = w - panel_w - 8; py = 46
            overlay = frame.copy()
            cv2.rectangle(overlay, (px,py), (px+panel_w, py+panel_h), (18,18,30), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            cv2.putText(frame, "DO THIS:", (px+10, py+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,212,216), 1)
            # 火柴人（佔面板上 ~48%）
            icon_col = (0,220,0) if (state["in_pose"] or state["done"]) else (255,255,255)
            fig_h = int(panel_h*0.48)
            draw_pose_icon(frame, cur_key, (px+12, py+26, panel_w-24, fig_h),
                           color=icon_col, thick=5)
            # 分步說明（火柴人下方）
            sy = py + 26 + fig_h + 12
            for i, line in enumerate(state.get("steps", [])):
                cv2.putText(frame, line, (px+12, sy + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235,235,235), 1, cv2.LINE_AA)

            # 底部除錯讀數
            if random_mode:
                dbg = f"frame={frame_idx} phase={getattr(coach,'_phase','?')} cf={getattr(coach,'_celebrate_frames',0)} mode=RANDOM"
            else:
                dbg = f"frame={frame_idx} done={state['done']} mode=SINGLE"
            cv2.putText(frame, dbg, (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1, cv2.LINE_AA)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                if random_mode:
                    coach.skip(); print(f"[yoga] 跳題 → {coach._coach.label}")
                else:
                    names = list(POSES.keys())
                    nxt = names[(names.index(coach.pose_name)+1) % len(names)]
                    coach = YogaCoach(pose_name=nxt, hold_target=args.hold)
                    print(f"[yoga] 換姿勢：{POSES[nxt][0]}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("[yoga] 已結束。")


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
