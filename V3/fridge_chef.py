# -*- coding: utf-8 -*-
"""
fridge_chef.py  ——  冰箱食材食譜助手（S2 / 對應發想 M-13）
=========================================================
把食材給筆電鏡頭看 → 按鍵 → VLM 辨識食材 + 建議能做的菜。全程離線。

兩種輸入方式都支援（程式完全一樣，看鏡頭當下的畫面即可）：
   方式 A：手機顯示「一張擺好幾樣食材的照片」給鏡頭看（食材多時最方便）
   方式 B：手拿實物給鏡頭看（一兩樣食材時很自然，例如一手蘋果一手蛋）

操作（視窗有焦點時按鍵）：
   空白鍵 → 拍當前畫面，辨識食材 + 建議食譜
   1      → 只「辨識有哪些食材」
   2      → 問「這些食材能做什麼菜」（附步驟）
   a      → 自訂問題
   q      → 離開

跑法：python fridge_chef.py --source 1
小撇步：食材擺整齊、拍清楚，辨識最準、食譜建議最好。
"""

import argparse
import cv2
from vlm_client import ask_about_image

PRESET = {
    ord(" "): "你是料理助手。請用繁體中文先列出畫面中看到的食材，再建議 1 到 2 道可以用這些食材做的家常菜，每道附簡單步驟。",
    ord("1"): "請用繁體中文列出畫面中你看到的所有食材。",
    ord("2"): "請用繁體中文根據畫面中的食材，建議 2 道家常菜，並各附 3-4 個簡單步驟。",
}


def ask_and_print(frame, question):
    print("\n" + "=" * 56)
    print("（UGen300 正在看圖思考，請稍候幾秒…）")
    answer = ask_about_image(frame, question)
    print(f"\n🍳 冰箱大廚：\n{answer}")
    print("=" * 56 + "\n（空白=辨識+食譜 1=只辨識 2=食譜含步驟 a=自訂 q=離開）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam 編號（外接通常 1，內建 0）")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[fridge] 無法開啟影像來源：{args.source}（外接 webcam 試 --source 1）")
        return

    print("[fridge] 冰箱大廚啟動。把食材（手機照片或實物）給鏡頭，按空白鍵開始。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 36), (26, 26, 46), -1)
            cv2.putText(frame, "SPACE=identify+recipe  1=identify 2=recipe  a=ask  q=quit",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 216), 2)
            cv2.imshow("UGen300 Fridge Chef (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in PRESET:
                ask_and_print(frame.copy(), PRESET[key])
            elif key == ord("a"):
                q = input("\n請輸入問題（中/英文）按 Enter：\n> ").strip()
                if q:
                    ask_and_print(frame.copy(), q)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[fridge] 已結束。")


if __name__ == "__main__":
    main()
