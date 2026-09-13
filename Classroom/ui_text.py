"""
ui_text.py —— 在 OpenCV 影像上畫繁體中文(OpenCV 內建字型不支援中文)。
用 Pillow + 微軟正黑體;找不到字型時退回 OpenCV 英文字。

效能:只把「文字所在的小區塊」轉成 PIL 畫完再貼回,不轉整張畫布
(舊版每次呼叫都轉整張 1280×720,一幀畫 20 個字串就掉到 5 fps)。
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
    if not text:
        return img
    f = _font(size)
    if f is None:
        cv2.putText(img, text, (int(xy[0]), int(xy[1]) + size), cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 2, cv2.LINE_AA)
        return img
    # 文字的外框(含陰影位移與一點邊距)
    try:
        l, t, r, b = f.getbbox(text, anchor=anchor)
    except (TypeError, ValueError):
        l, t, r, b = f.getbbox(text)
    pad = 4
    x0 = int(np.floor(xy[0] + l)) - pad; y0 = int(np.floor(xy[1] + t)) - pad
    x1 = int(np.ceil(xy[0] + r)) + pad + 2; y1 = int(np.ceil(xy[1] + b)) + pad + 2
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return img
    region = img[cy0:cy1, cx0:cx1]
    pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])
    ox, oy = xy[0] - cx0, xy[1] - cy0
    if shadow:
        d.text((ox + 2, oy + 2), text, font=f, fill=(0, 0, 0), anchor=anchor)
    d.text((ox, oy), text, font=f, fill=rgb, anchor=anchor)
    img[cy0:cy1, cx0:cx1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def text_width(text, size):
    """文字寬度(像素),給底板用。"""
    f = _font(size)
    return int(f.getlength(text)) if f else len(text) * size // 2


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


def screen_size(default=(1280, 720)):
    """主螢幕解析度(全螢幕補黑邊用);取不到就回預設。"""
    try:
        import ctypes; u = ctypes.windll.user32; return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return default


def fit_to_screen(canvas, fullscreen, screen=None):
    """全螢幕時 HighGUI 會把畫布直接拉到整個螢幕(KEEPRATIO 無效),16:10 螢幕會把 16:9 畫布垂直拉長 11%;
    這裡先等比縮放並補黑邊,再交給 imshow。非全螢幕原樣回傳。"""
    if not fullscreen: return canvas
    sw, sh = screen or screen_size()
    h, w = canvas.shape[:2]
    if abs(sw / sh - w / h) < 0.01: return canvas
    s = min(sw / w, sh / h); cw, ch = int(w * s), int(h * s)
    out = np.zeros((sh, sw, 3), np.uint8); ox, oy = (sw - cw) // 2, (sh - ch) // 2
    out[oy:oy + ch, ox:ox + cw] = cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_AREA)
    return out
