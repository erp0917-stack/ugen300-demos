"""
ui_text.py —— 在 OpenCV 影像上畫繁體中文(OpenCV 內建字型不支援中文)。
用 Pillow + 微軟正黑體;找不到字型時退回 OpenCV 英文字。
"""
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

_FONT_PATHS = [r"C:\Windows\Fonts\msjhbd.ttc", r"C:\Windows\Fonts\msjh.ttc",
               r"C:\Windows\Fonts\mingliu.ttc", r"C:\Windows\Fonts\arial.ttf"]
_font_cache = {}


def _font(size):
    if size in _font_cache:
        return _font_cache[size]
    f = None
    if _HAS_PIL:
        for p in _FONT_PATHS:
            try:
                f = ImageFont.truetype(p, size); break
            except OSError:
                continue
    _font_cache[size] = f
    return f


def draw_text(img, text, xy, size=32, color=(255, 255, 255), anchor="la", shadow=True):
    """在 BGR 影像上畫文字(就地修改並回傳)。anchor 同 Pillow:la=左上, mm=正中, ra=右上。"""
    f = _font(size)
    if f is None:
        cv2.putText(img, text, (int(xy[0]), int(xy[1]) + size), cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 2, cv2.LINE_AA)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])
    if shadow:
        d.text((xy[0] + 2, xy[1] + 2), text, font=f, fill=(0, 0, 0), anchor=anchor)
    d.text(xy, text, font=f, fill=rgb, anchor=anchor)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def badge(img, text="離線 · UGen300", size=22):
    """右上角徽章。"""
    h, w = img.shape[:2]
    pad = 10
    f = _font(size)
    tw = int(f.getlength(text)) if f else len(text) * size // 2
    x0, y0 = w - tw - pad * 3, pad
    cv2.rectangle(img, (x0, y0), (w - pad, y0 + size + pad), (40, 40, 40), -1)
    cv2.rectangle(img, (x0, y0), (w - pad, y0 + size + pad), (0, 200, 120), 1)
    draw_text(img, text, (x0 + pad, y0 + pad // 2), size, (0, 230, 140), shadow=False)
    return img
