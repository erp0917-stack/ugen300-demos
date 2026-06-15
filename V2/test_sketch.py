# -*- coding: utf-8 -*-
"""
test_sketch.py — 線稿草圖風格比較（純 CPU、不用 UGen300、不用生成模型）
======================================================================
吃一張照片，產出 3 種「從真實照片描出來」的線稿，讓你挑風格。

放在 V2 資料夾，照片同層 face.jpg。
用法：python test_sketch.py face.jpg
產出（會存在同一個資料夾）：
    sketch_1_dodge.png   鉛筆素描（最接近手繪線稿感）
    sketch_2_pencil.png  OpenCV 內建素描
    sketch_3_edge.png    Canny 邊緣（純線條輪廓）
"""

import sys
import cv2
import numpy as np


def dodge_sketch(img):
    """經典 dodge 法：灰階→反相→模糊→顏色減淡，出鉛筆素描感（白底灰線）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(gray, 255 - blur, scale=256)


def pencil_sketch(img):
    """OpenCV 內建 pencilSketch（photo 模組）。"""
    try:
        gray, _color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07,
                                        shade_factor=0.05)
        return gray
    except Exception as e:
        print(f"[!] 這個 OpenCV 沒有 pencilSketch（{e}），改用 dodge 代替")
        return dodge_sketch(img)


def edge_sketch(img):
    """Canny 邊緣：純線條輪廓（白底黑線）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 130)
    return 255 - edges


def main():
    if len(sys.argv) < 2:
        print("用法：python test_sketch.py face.jpg")
        return
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"[X] 讀不到照片：{sys.argv[1]}")
        return

    # 小圖描線會糊，先放大到長邊 512
    h, w = img.shape[:2]
    if max(h, w) < 512:
        s = 512 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)

    outs = [
        ("sketch_1_dodge.png", dodge_sketch(img), "鉛筆素描（最接近手繪線稿感）"),
        ("sketch_2_pencil.png", pencil_sketch(img), "OpenCV 內建素描"),
        ("sketch_3_edge.png", edge_sketch(img), "Canny 邊緣（純線條輪廓）"),
    ]
    for fn, im, desc in outs:
        cv2.imwrite(fn, im)
        print(f"[V] {fn}  — {desc}")
    print("\n三張都產好了，在 V2 資料夾打開比較，挑一個你喜歡的（1/2/3）告訴我。")


if __name__ == "__main__":
    main()
