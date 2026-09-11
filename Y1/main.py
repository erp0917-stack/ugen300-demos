# -*- coding: utf-8 -*-
"""
main.py  ——  B1 即時物件偵測（暖場 / 保命 demo）
================================================
webcam 畫面 → UGen300 偵測物件 → 把每個物件框起來、標名稱 → 顯示 FPS。
直觀證明「UGen300 真的在本機跑 AI」。

跑法（週一環境好了之後）：
   python main.py --hef yolov8m.hef --source 1
       --source 1  外接 webcam（內建是 0，可試）

操作：視窗按 q 離開

demo 重點：跑起來後當眾關 Wi-Fi / 拔網路線，框框照畫 → 證明完全離線。
"""

import argparse
import time
import cv2

from hailo_detect import ObjectDetector

# 給不同類別不同顏色（看起來更專業）
COLORS = [
    (0, 212, 216), (0, 200, 120), (255, 180, 0), (220, 80, 80),
    (180, 120, 255), (120, 200, 255), (255, 120, 200), (200, 200, 0),
]


def draw_detections(frame, detections, fps):
    """把偵測框、名稱、信心度、FPS 畫上去（已寫好）"""
    h, w = frame.shape[:2]
    for i, (label, score, box) in enumerate(detections):
        x1, y1, x2, y2 = box
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.0%}"
        # 標籤底色塊
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 右上角 FPS + 物件數
    info = f"FPS: {fps:.1f}   Objects: {len(detections)}"
    cv2.rectangle(frame, (0, 0), (w, 38), (26, 26, 46), -1)
    cv2.putText(frame, info, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 216), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8m.hef", help="偵測模型檔路徑")
    ap.add_argument("--source", default="0", help="webcam 編號（0/1）或影片檔路徑")
    ap.add_argument("--conf", type=float, default=0.5, help="信心度門檻")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[main] 無法開啟影像來源：{args.source}")
        print("       外接 webcam 通常是 1（內建鏡頭是 0），試試 --source 1")
        return

    det = ObjectDetector(args.hef, conf_threshold=args.conf)

    print("[main] 開始即時偵測。按 q 離開。")
    prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 鏡像，比較直覺

            detections = det.infer(frame)

            # 計算 FPS（平滑一點）
            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
            prev = now

            frame = draw_detections(frame, detections, fps)
            cv2.imshow("UGen300 Real-time Detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        det.close()
        print("[main] 已結束。")


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
