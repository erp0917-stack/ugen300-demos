# -*- coding: utf-8 -*-
"""
main.py  ——  D4 手勢翻頁主程式
==============================
把三塊串起來：
   webcam 畫面 → UGen300 抓骨架 → 判斷手勢 → 控制翻頁 → 螢幕顯示

跑法（週一環境好了之後）：
   python main.py --hef yolov8s_pose.hef --source 0
       --source 0   用第一支 webcam（外接的通常是 1，內建是 0，可試）
       --source demo.mp4   也可以餵預錄影片測試

操作：
   舉右手 → 下一頁 ｜ 舉左手 → 上一頁 ｜ 雙手舉高 → 從頭播放
   視窗按 q 離開

⚠️ 注意：要能真的翻頁，需先把 PowerPoint（或簡報）切到「播放模式」並讓它在最前面。
"""

import argparse
import cv2

from hailo_pose import PoseEstimator
from gesture_control import (
    detect_raw_gesture, GestureController, trigger_action,
    GESTURE_NONE, GESTURE_RIGHT, GESTURE_LEFT, GESTURE_BOTH, KP,
)

# 骨架連線（哪些關鍵點要連成線，畫出人形）
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),      # 兩隻手臂
    (5, 6), (5, 11), (6, 12), (11, 12),    # 軀幹
    (11, 13), (13, 15), (12, 14), (14, 16),  # 兩條腿
    (0, 5), (0, 6),                        # 頭連到肩
]

GESTURE_LABEL = {
    GESTURE_NONE: "（無）",
    GESTURE_RIGHT: "舉右手 → 下一頁",
    GESTURE_LEFT: "舉左手 → 上一頁",
    GESTURE_BOTH: "雙手舉高 → 從頭播放",
}


def draw_overlay(frame, keypoints, raw_gesture, conf_threshold=0.5):
    """把骨架和目前手勢狀態畫到畫面上（純顯示，已寫好）"""
    h, w = frame.shape[:2]

    if keypoints is not None:
        # 畫關鍵點
        for (x, y, c) in keypoints:
            if c >= conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        # 畫骨架連線
        for a, b in SKELETON:
            xa, ya, ca = keypoints[a]
            xb, yb, cb = keypoints[b]
            if ca >= conf_threshold and cb >= conf_threshold:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 180, 0), 2)

    # 左上角顯示目前手勢（讓你 demo 時看得到狀態）
    label = GESTURE_LABEL.get(raw_gesture, raw_gesture)
    cv2.rectangle(frame, (0, 0), (w, 40), (26, 26, 46), -1)
    cv2.putText(frame, f"Gesture: {label}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 216), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8s_pose.hef", help="pose 模型檔路徑")
    ap.add_argument("--source", default="0", help="webcam 編號（0/1）或影片檔路徑")
    ap.add_argument("--conf", type=float, default=0.5, help="關鍵點可信度門檻")
    ap.add_argument("--confirm-frames", type=int, default=4, help="連續幾格才確認手勢")
    ap.add_argument("--cooldown", type=float, default=1.2, help="翻頁後冷卻秒數")
    args = ap.parse_args()

    # 開啟影像來源（webcam 或影片）
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[main] 無法開啟影像來源：{args.source}")
        print("       外接 webcam 通常是 1（內建鏡頭是 0），試試 --source 1")
        return

    pose = PoseEstimator(args.hef, conf_threshold=args.conf)
    ctrl = GestureController(confirm_frames=args.confirm_frames, cooldown_sec=args.cooldown)

    print("[main] 開始。舉右手=下一頁、舉左手=上一頁、雙手=從頭播放，按 q 離開。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 鏡像，讓畫面像照鏡子，比較直覺

            keypoints = pose.infer(frame)              # 叫 UGen300 抓骨架
            raw = detect_raw_gesture(keypoints, args.conf) if keypoints else GESTURE_NONE
            action = ctrl.update(raw)                  # 防呆判斷
            if action:
                trigger_action(action)                 # 真的翻頁

            frame = draw_overlay(frame, keypoints, raw, args.conf)
            cv2.imshow("D4 手勢翻頁（按 q 離開）", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("[main] 已結束。")


if __name__ == "__main__":
    main()
