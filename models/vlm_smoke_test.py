# -*- coding: utf-8 -*-
"""
vlm_smoke_test.py
=================
驗證 HailoRT GenAI VLM 在 UGen300 / Hailo-10H 上能跑：
  1) import hailo_platform.VDevice / hailo_platform.genai.VLM
  2) 開 VDevice
  3) 載入 .hef
  4) 印出 input_frame_shape / format
  5) 用合成 numpy 圖跑一次 generate_all

每步獨立 try/except，成功印 ✓、失敗印 traceback 但不中斷後面的測試（除非無法繼續）。
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen2-VL-2B-Instruct.hef")

step = 0
def step_header(title):
    global step
    step += 1
    print(f"\n========== [{step}] {title} ==========", flush=True)

def ok(msg): print(f"  ✓ {msg}", flush=True)
def fail(label, exc):
    print(f"  ✗ {label} FAILED: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()


# ---- 1. imports ----------------------------------------------------------
step_header("import hailo_platform / genai")
try:
    from hailo_platform import VDevice
    ok(f"VDevice: {VDevice}")
except Exception as e:
    fail("import VDevice", e); sys.exit(1)

try:
    from hailo_platform.genai import VLM
    ok(f"VLM: {VLM}")
except Exception as e:
    fail("import VLM", e); sys.exit(1)


# ---- 2. .hef 檔在不在 ----------------------------------------------------
step_header(".hef file present & size sanity")
hef = Path(HEF_PATH)
try:
    if not hef.exists():
        raise FileNotFoundError(hef)
    size_gb = hef.stat().st_size / (1024**3)
    ok(f"{hef} exists, size={size_gb:.2f} GB")
    if size_gb < 1.0:
        print(f"  ⚠️ 檔案小於 1 GB，可能還沒下載完整", flush=True)
except Exception as e:
    fail("hef file check", e); sys.exit(1)


# ---- 3. 開 VDevice -------------------------------------------------------
step_header("open VDevice")
vdevice = None
try:
    t0 = time.time()
    vdevice = VDevice()
    ok(f"VDevice opened in {time.time()-t0:.2f}s")
except Exception as e:
    fail("VDevice()", e); sys.exit(1)


# ---- 4. 載入 VLM ---------------------------------------------------------
step_header("load VLM(vdevice, hef_path)")
vlm = None
try:
    t0 = time.time()
    vlm = VLM(vdevice, str(hef))
    ok(f"VLM loaded in {time.time()-t0:.2f}s")
except Exception as e:
    fail("VLM init", e)
    try: vdevice.release()
    except Exception: pass
    sys.exit(1)


# ---- 5. 印 input_frame_shape / format -----------------------------------
step_header("inspect input_frame_shape / format / size")
try:
    shape = vlm.input_frame_shape()
    ok(f"input_frame_shape() = {shape}")
except Exception as e:
    fail("input_frame_shape()", e)

try:
    ftype = vlm.input_frame_format_type()
    ok(f"input_frame_format_type() = {ftype}")
except Exception as e:
    fail("input_frame_format_type()", e)

try:
    forder = vlm.input_frame_format_order()
    ok(f"input_frame_format_order() = {forder}")
except Exception as e:
    fail("input_frame_format_order()", e)

try:
    fsize = vlm.input_frame_size()
    ok(f"input_frame_size() = {fsize}")
except Exception as e:
    fail("input_frame_size()", e)


# ---- 6. 用合成 numpy 圖跑 generate_all -----------------------------------
step_header("run generate_all on a synthetic 336x336 RGB image")
try:
    # 嘗試用 shape 配對；不行就 fallback 336x336x3 uint8（官方範例固定 336x336）
    try:
        shape = vlm.input_frame_shape()
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            h, w = int(shape[0]), int(shape[1])
            c = int(shape[2]) if len(shape) >= 3 else 3
        else:
            h = w = 336; c = 3
    except Exception:
        h = w = 336; c = 3

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
    print(f"  using synthetic image shape={img.shape} dtype={img.dtype}", flush=True)

    prompt = [
        {"role": "system",
         "content": [{"type": "text", "text": "You are a concise helper."}]},
        {"role": "user",
         "content": [
            {"type": "image"},
            {"type": "text", "text": "What is in this image? Reply in <=15 words."},
         ]},
    ]

    t0 = time.time()
    resp = vlm.generate_all(
        prompt=prompt,
        frames=[img],
        temperature=0.1,
        seed=42,
        max_generated_tokens=64,
    )
    dt = time.time() - t0
    text = resp if isinstance(resp, str) else str(resp)
    # 官方範例會 split <|im_end|> 之類；這裡先全印再 split
    print(f"  ✓ generate_all returned in {dt:.2f}s", flush=True)
    print("  ---- raw response ----", flush=True)
    print(text, flush=True)
    print("  ---- end ----", flush=True)
except Exception as e:
    fail("generate_all", e)


# ---- 7. cleanup ----------------------------------------------------------
step_header("cleanup")
try:
    if vlm:
        try: vlm.clear_context()
        except Exception: pass
        vlm.release()
        ok("VLM released")
except Exception as e:
    fail("vlm.release()", e)
try:
    if vdevice:
        vdevice.release()
        ok("VDevice released")
except Exception as e:
    fail("vdevice.release()", e)

print("\n[smoke test 結束]", flush=True)
