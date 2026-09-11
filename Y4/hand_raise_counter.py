# -*- coding: utf-8 -*-
"""
hand_raise_counter.py  ——  舉手統計 / 即時投票（S4 / 對應發想 W-08）
==================================================================
偵測畫面中有幾個人舉手，即時顯示「舉手 X 人 / 共 Y 人」。
課堂、會議、活動現場即時投票，不用發表單。

操作：幾個人在鏡頭前舉手，畫面即時統計。按 q 離開。
跑法：python hand_raise_counter.py --hef yolov8s_pose.hef --source 1

⚠️ 多人偵測：這個應用要「同時偵測多個人」。
   單人版的 hailo_pose.py 目前回傳一個人；週一對接時，跟 Claude Code 說：
   「把 hailo_pose 改成回傳『多個人』的關鍵點清單（infer_multi），舉手統計要用」。
   官方 pose 範例本來就支援多人，改動不大。
   （在還沒改成多人前，本程式會把單人當成 1 人清單，邏輯仍可驗證。）
"""

import argparse
import cv2

from hailo_pose import PoseEstimator
from hand_raise_logic import count_raised_hands, is_hand_raised

SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(0,5),(0,6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8s_pose.hef")
    ap.add_argument("--source", default="0")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[handraise] 無法開啟影像來源：{args.source}（外接 webcam 試 --source 1）")
        return

    pose = PoseEstimator(args.hef)
    print("[handraise] 舉手統計啟動。請大家舉手試試，按 q 離開。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            # 🔧 對接點：取得「多人」關鍵點
            # 對接後 pose.infer_multi(frame) 回傳 list（每人 17 點）
            # 目前先用單人版包成 1 人清單，讓邏輯可驗證
            if hasattr(pose, "infer_multi"):
                people = pose.infer_multi(frame) or []
            else:
                one = pose.infer(frame)
                people = [one] if one else []

            raised, total = count_raised_hands(people)

            # 畫每個人的骨架 + 舉手者高亮
            for person in people:
                up = is_hand_raised(person)
                col = (0, 220, 0) if up else (180, 180, 180)
                for a, b in SKELETON:
                    xa, ya, ca = person[a]
                    xb, yb, cb = person[b]
                    if ca >= 0.5 and cb >= 0.5:
                        cv2.line(frame, (int(xa),int(ya)), (int(xb),int(yb)), col, 2)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 50), (26, 26, 46), -1)
            cv2.putText(frame, f"Hands up: {raised} / {total}", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 216), 2)
            cv2.imshow("UGen300 Hand-Raise Counter (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("[handraise] 已結束。")


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
