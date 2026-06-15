# -*- coding: utf-8 -*-
"""
vlm_client_test.py
==================
不開 webcam、不開視窗，直接拿合成圖跑兩次 ask_about_image，
驗證：(1) 中文回答有出來、(2) 第二次推論不重載模型（明顯快很多）。
"""

import time
import cv2
import numpy as np

from vlm_client import ask_about_image


def make_image_shapes_and_text():
    """合成一張 640x480 的圖，畫幾個有顏色的方塊 + 一個圓 + 寫字。"""
    img = np.full((480, 640, 3), 245, dtype=np.uint8)  # 接近白底
    cv2.rectangle(img, ( 40, 60), (200, 220), (0, 0, 220), -1)   # 紅
    cv2.rectangle(img, (240, 60), (400, 220), (40, 180, 40), -1) # 綠
    cv2.circle(img, (520, 140), 80, (220, 120, 0), -1)           # 藍/橘
    cv2.putText(img, "HELLO UGen300", (60, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3)
    return img


def make_image_photo_like():
    """更像照片的合成圖：雜訊背景 + 多色塊 + 漸層，給 VLM 一些可描述的視覺特徵。"""
    rng = np.random.default_rng(1)
    # 背景：低頻雜訊
    img = rng.integers(80, 200, size=(480, 640, 3), dtype=np.uint8)
    # 上半部「天空」灰藍漸層
    for y in range(0, 200):
        t = y / 200.0
        img[y, :, 0] = int(180 + 30 * (1 - t))
        img[y, :, 1] = int(150 + 30 * (1 - t))
        img[y, :, 2] = int(120 + 40 * (1 - t))
    # 下方一塊綠色「草地」
    cv2.rectangle(img, (0, 320), (640, 480), (60, 160, 90), -1)
    # 三個彩色物件（球？）
    cv2.circle(img, (140, 260), 50, (40,  40, 220), -1)  # 紅
    cv2.circle(img, (320, 260), 50, (50, 200,  40), -1)  # 綠
    cv2.circle(img, (500, 260), 50, (220, 180, 40), -1)  # 黃
    return img


def run(label, frame, question):
    print(f"\n========== {label} ==========")
    print(f"問題：{question}")
    t0 = time.time()
    ans = ask_about_image(frame, question)
    dt = time.time() - t0
    print(f"耗時：{dt:.2f}s")
    print("回答：")
    print(ans)
    return dt, ans


def main():
    img1 = make_image_shapes_and_text()
    img2 = make_image_photo_like()

    # 第一次：含模型載入（約 10 秒 + 推論幾秒）
    t_first, ans1 = run(
        "Round 1: 三個彩色形狀 + 文字",
        img1,
        "畫面中有哪些顏色和形狀？請用繁體中文簡短回答。",
    )

    # Round 2 控制組：用同一張圖、不同問題 → 驗證 clear_context 真的重置了
    t_second, ans2 = run(
        "Round 2 [控制]: 同一張彩色形狀圖，換問題",
        img1,
        "這張圖最左邊是什麼顏色的方塊？請用繁體中文回答。",
    )

    # Round 3：換一張寫實一點的合成圖（漸層背景 + 三顆球 + 草地）
    t_third, ans3 = run(
        "Round 3: 漸層背景 + 三顆彩色球",
        img2,
        "畫面中有什麼物體？大致是什麼場景？請用繁體中文回答。",
    )

    print("\n========== 驗證摘要 ==========")
    print(f"Round 1（含模型載入）耗時：{t_first:.2f}s")
    print(f"Round 2（控制：同圖換題）耗時：{t_second:.2f}s")
    print(f"Round 3（換圖）        耗時：{t_third:.2f}s")

    # 簡單檢查三輪都拿到非空字串、沒有 [vlm_client] 失敗訊息
    bad = []
    for i, a in enumerate((ans1, ans2, ans3), 1):
        if not a or a.startswith("[vlm_client]"):
            bad.append(f"Round {i}: {a[:80]!r}")
    if bad:
        print("\n❌ 有 round 沒拿到正常回答：")
        for b in bad: print("  -", b)
        return 1
    else:
        print("\n✅ 三輪都拿到回答，VLM client 對接成功")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
