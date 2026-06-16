# -*- coding: utf-8 -*-
"""
sketch.py — Dodge 鉛筆素描 + 三庭五眼裝飾網格，純 CPU
"""

import cv2
import numpy as np


def make_dodge_sketch(frame_bgr, blur_ksize=21):
    """
    灰階 → 反相 → 高斯模糊 → cv2.divide（dodge 算法）
    回傳 grayscale uint8 numpy array（單通道）。
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    inv    = cv2.bitwise_not(gray)
    blur   = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    return sketch


def draw_grid_on_sketch(sketch_gray):
    """
    在鉛筆素描（灰階 array）上疊「三庭五眼」裝飾網格：
    - 3 條等距橫線（模擬髮際/眉/鼻底/下巴三庭分隔）
    - 2 條等距豎線（模擬五眼示意的內眼角對齊線）
    - 細灰線、半透明疊加
    - 左上標「三庭五眼」文字
    回傳 BGR uint8 array（方便 matplotlib imshow）。
    """
    # 轉 BGR 才能畫彩色線
    canvas = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    H, W   = canvas.shape[:2]

    # 半透明疊加層
    overlay = canvas.copy()
    line_color = (160, 160, 160)   # 灰
    lw = max(1, W // 200)          # 線寬自適應

    # 三庭：4 等分的第 1、2、3 條橫線
    for i in range(1, 4):
        y = int(H * i / 4)
        cv2.line(overlay, (0, y), (W, y), line_color, lw)

    # 五眼：5 等分的第 1、2 條豎線（中段，模擬眼角對齊）
    for i in range(1, 5):
        x = int(W * i / 5)
        cv2.line(overlay, (x, int(H * 0.15)), (x, int(H * 0.85)), line_color, lw)

    # 半透明混合
    alpha  = 0.45
    result = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # 左上標籤「三庭五眼」
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, W / 500)
    thickness  = max(1, W // 300)
    label      = "san-ting wu-yan"   # ASCII fallback（PIL 中文需另裝）
    # 嘗試用 PIL 畫中文標籤
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)
        fsize   = max(12, W // 20)
        try:
            font_pil = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", fsize)
        except Exception:
            print("Microsoft JhengHei font not found (C:/Windows/Fonts/msjh.ttc), falling back to default font")
            font_pil = ImageFont.load_default()
        draw.text((4, 4), "三庭五眼", font=font_pil, fill=(100, 100, 100))
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(result, label, (4, int(font_scale * 28)),
                    font, font_scale, (100, 100, 100), thickness, cv2.LINE_AA)

    return result
