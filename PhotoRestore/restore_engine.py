"""
restore_engine.py —— 老照片修復引擎:三個單圖進單圖出的模型 + 分塊(tile)推論 + 重疊融合。

模型(HAILO10H):
  zero_dce.hef           提亮   輸入/輸出 400×600×3   (輸出偏亮,以 strength 混合)
  dncnn_color_blind.hef  去雜訊 輸入/輸出 321×481×3
  real_esrgan_x2.hef     放大   輸入 512×512×3 → 輸出 1024×1024×3

分塊與融合是純 numpy,可離線測試(TileRunner 接受任何 callable 當推論函式)。
"""
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "brighten": dict(hef=os.path.join(_HERE, "zero_dce.hef"), tile=(400, 600), scale=1, label="提亮"),
    "denoise":  dict(hef=os.path.join(_HERE, "dncnn_color_blind.hef"), tile=(321, 481), scale=1, label="去雜訊"),
    "upscale":  dict(hef=os.path.join(_HERE, "real_esrgan_x2.hef"), tile=(512, 512), scale=2, label="放大 2 倍"),
}
ORDER = ("brighten", "denoise", "upscale")   # 疊加時的固定順序


# ---------- 純邏輯:分塊 / 融合 ----------
def plan_tiles(h, w, th, tw, overlap=16):
    """回傳 [(y, x)] 左上角座標,覆蓋整張圖;最後一塊貼齊邊界。圖比 tile 小時只有 (0,0)。"""
    def starts(n, t):
        if n <= t:
            return [0]
        step = max(1, t - overlap)
        s = list(range(0, n - t, step)) + [n - t]
        return sorted(set(s))
    return [(y, x) for y in starts(h, th) for x in starts(w, tw)]


def _weight(th, tw, overlap):
    """線性漸變權重,重疊區平滑融合。"""
    wy = np.ones(th, np.float32); wx = np.ones(tw, np.float32)
    r = max(1, min(overlap, th // 2, tw // 2))
    ramp = np.linspace(0.1, 1.0, r, dtype=np.float32)
    wy[:r] = ramp; wy[-r:] = ramp[::-1]; wx[:r] = ramp; wx[-r:] = ramp[::-1]
    return (wy[:, None] * wx[None, :])[:, :, None]


class TileRunner:
    """把任意 (th,tw,3)→(th*s,tw*s,3) 的推論函式套到整張大圖。"""
    def __init__(self, infer, tile, scale=1, overlap=16):
        self.infer, (self.th, self.tw), self.scale, self.overlap = infer, tile, scale, overlap

    def run(self, img, on_progress=None):
        h, w = img.shape[:2]; s = self.scale
        # 比 tile 小的圖:先補到 tile 大小(邊緣反射),推完再裁回
        pad_b, pad_r = max(0, self.th - h), max(0, self.tw - w)
        src = cv2.copyMakeBorder(img, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT) if (pad_b or pad_r) else img
        H, W = src.shape[:2]
        acc = np.zeros((H * s, W * s, 3), np.float32); wsum = np.zeros((H * s, W * s, 1), np.float32)
        wt = _weight(self.th * s, self.tw * s, self.overlap * s)
        tiles = plan_tiles(H, W, self.th, self.tw, self.overlap)
        for i, (y, x) in enumerate(tiles):
            out = self.infer(np.ascontiguousarray(src[y:y + self.th, x:x + self.tw])).astype(np.float32)
            ys, xs = y * s, x * s
            acc[ys:ys + self.th * s, xs:xs + self.tw * s] += out * wt
            wsum[ys:ys + self.th * s, xs:xs + self.tw * s] += wt
            if on_progress:
                on_progress(i + 1, len(tiles))
        res = np.rint(acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
        return res[:h * s, :w * s]


def adaptive_brighten_strength(img, user_strength=1.0):
    """Zero-DCE 對不太暗的圖也會猛提亮 → 依原圖平均亮度給上限。
    平均 <40 → 最多 0.9;≈90 → 約 0.45;>150 → 0.15。取與使用者設定的較小值。"""
    mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    cap = float(np.clip((150.0 - mean) / 125.0, 0.15, 0.9))
    return min(float(user_strength), cap)


def blend(original, processed, strength):
    """strength 0~1:0 = 原圖,1 = 完全處理後。尺寸不同時把原圖縮放到處理後尺寸。"""
    if original.shape != processed.shape:
        original = cv2.resize(original, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(processed, float(strength), original, 1.0 - float(strength), 0)


# ---------- 需要裝置 ----------
class HailoModel:
    def __init__(self, hef):
        import hailo_vdevice
        self.vd = hailo_vdevice.get()
        self.m = self.vd.create_infer_model(hef); self.m.set_batch_size(1)
        self.c = self.m.configure()
        self.ishape = tuple(self.m.input().shape); self.oname = self.m.output_names[0]
        self.oshape = tuple(self.m.output(self.oname).shape)

    def __call__(self, tile_bgr):
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        assert rgb.shape == self.ishape, (rgb.shape, self.ishape)
        b = self.c.create_bindings(); b.input().set_buffer(np.ascontiguousarray(rgb))
        out = np.zeros(self.oshape, np.uint8); b.output(self.oname).set_buffer(out)
        self.c.run([b], 20000)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class RestoreEngine:
    def __init__(self):
        self.models = {}

    def load(self, keys=ORDER, on_status=None):
        for k in keys:
            if k not in self.models:
                if on_status: on_status(f"載入 {MODELS[k]['label']} 模型…")
                self.models[k] = HailoModel(MODELS[k]["hef"])
        if on_status: on_status("模型就緒")

    def apply(self, img, key, strength=1.0, on_progress=None):
        cfg = MODELS[key]
        runner = TileRunner(self.models[key], cfg["tile"], cfg["scale"])
        out = runner.run(img, on_progress)
        if key == "brighten":
            strength = adaptive_brighten_strength(img, strength)
        return blend(img, out, strength) if strength < 1.0 else out

    def pipeline(self, img, steps, strengths=None, on_status=None):
        """steps: 要套用的 key 集合,依 ORDER 順序執行。回傳最終影像。"""
        strengths = strengths or {}
        cur = img
        for k in ORDER:
            if k in steps:
                if on_status: on_status(f"{MODELS[k]['label']}中…")
                cur = self.apply(cur, k, strengths.get(k, 1.0),
                                 on_progress=(lambda i, n, k=k: on_status(f"{MODELS[k]['label']}中… {i}/{n} 塊")) if on_status else None)
        return cur
