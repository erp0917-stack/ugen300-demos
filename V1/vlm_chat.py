# -*- coding: utf-8 -*-
"""
vlm_chat.py  ——  VLM 看圖問答（最吸睛的 demo）
==============================================
webcam 拍下當前畫面 → 問 UGen300 上的 VLM 一個問題 → 它用人話回答。
全程在本機 / UGen300 上跑，畫面不上雲。

操作（視窗有焦點時按鍵）：
   空白鍵 → 拍當前畫面，問「描述這個畫面」
   1      → 拍照問「這張圖裡有什麼？用繁體中文回答」
   2      → 拍照問「畫面中有幾個人？他們在做什麼？」
   3      → 拍照問「描述畫面中人物的穿著和動作」
   a      → 自訂問題（讓客人出題，你在終端機打字）
   q      → 離開

跑法（週一環境好了之後）：
   python vlm_chat.py --source 1     # 外接 webcam 通常是 1

demo 重點：問答前後當眾關 Wi-Fi / 拔網路 → VLM 照答 → 證明離線。
"""

import argparse
import cv2

from vlm_client import ask_about_image

# 預設問題（這些是「安全題」，VLM 答得最穩）
PRESET = {
    ord("1"): "請用繁體中文描述這張圖裡有什麼。",
    ord("2"): "畫面中有幾個人？他們在做什麼？請用繁體中文回答。",
    ord("3"): "請用繁體中文描述畫面中人物的穿著和動作。",
    ord(" "): "請用繁體中文簡短描述這個畫面。",
}


def ask_and_print(frame, question):
    """送一張畫面 + 問題給 VLM，把答案印在終端機（中文顯示最完整）"""
    print("\n" + "=" * 56)
    print(f"問題：{question}")
    print("（UGen300 正在看圖思考，請稍候幾秒…）")
    answer = ask_about_image(frame, question)
    print(f"\nVLM 回答：\n{answer}")
    print("=" * 56 + "\n（畫面視窗按鍵繼續：空白=描述 1/2/3=預設題 a=自訂 q=離開）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam 編號（0/1）或影片檔")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[vlm_chat] 無法開啟影像來源：{args.source}（外接 webcam 試 --source 1）")
        return

    print("[vlm_chat] 開始。把視窗點到最前面，按鍵操作：")
    print("  空白=描述畫面　1/2/3=預設問題　a=自訂問題（客人出題）　q=離開")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            # 畫面上方顯示操作提示（英文，避免 OpenCV 中文亂碼）
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 36), (26, 26, 46), -1)
            cv2.putText(frame, "SPACE=describe  1/2/3=preset  a=ask  q=quit",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 216), 2)
            cv2.imshow("VLM 看圖問答（按 q 離開）", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in PRESET:
                ask_and_print(frame.copy(), PRESET[key])
            elif key == ord("a"):
                # 自訂問題：讓客人出題，你在終端機打字
                q = input("\n請輸入客人的問題（中文/英文都行），按 Enter：\n> ").strip()
                if q:
                    ask_and_print(frame.copy(), q)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[vlm_chat] 已結束。")


if __name__ == "__main__":
    main()
