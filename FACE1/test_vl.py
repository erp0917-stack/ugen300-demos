# -*- coding: utf-8 -*-
"""
test_vl.py — Block 1 驗證：裁臉 → VL 看圖 → 印 scores + obs → close VDevice

用法：
    python test_vl.py
    python test_vl.py ..\V2\face.jpg tw
"""

import sys
import cv2
import vlm_client
from face_helpers import LANG, _build_vl_prompts, parse_vl_json
from face_report import detect_and_crop

img_path = sys.argv[1] if len(sys.argv) > 1 else r"..\V2\face.jpg"
lang     = sys.argv[2] if len(sys.argv) > 2 else "tw"

frame = cv2.imread(img_path)
if frame is None:
    print(f"[X] 讀不到圖片：{img_path}")
    sys.exit(1)
print(f"[OK] 圖片 {img_path}  {frame.shape[1]}x{frame.shape[0]}")

crop, found = detect_and_crop(frame)
print(f"[偵臉] crop {crop.shape[1]}x{crop.shape[0]}  found={found}")

vl_lang = "tw" if lang == "cn" else lang
system, question = _build_vl_prompts(vl_lang)
print(f"[VL] 送進 Qwen2-VL-2B（第一次載入約 10 秒）……")
raw = vlm_client.ask_about_image(crop, question, system_prompt=system,
                                  max_tokens=400, temperature=0.10)

print("\n===== VL 原始輸出 =====")
print(raw)

vl_data, strict = parse_vl_json(raw, LANG[vl_lang]["purify"])

print(f"\n===== VL 解析結果（{'JSON 一次過' if strict else 'regex 救援'}）=====")
print(f"scores: {vl_data['scores']}")
print("obs:")
for k, v in vl_data["obs"].items():
    print(f"  {k}: {v}")

scores_ok = all(isinstance(v, int) for v in vl_data["scores"].values())
obs_ok    = all(vl_data["obs"].get(k) for k in ["shape", "eyes", "nose", "mouth",
                                                   "skin_state", "brow", "overall_impression"])
vals = list(vl_data["scores"].values())
monotone = vals == sorted(vals, reverse=True)

print(f"\n[{'V' if scores_ok else 'X'}] 分數都是整數")
print(f"[{'V' if obs_ok    else 'X'}] obs 七個欄位有值")
print(f"[{'V' if not monotone else '!'}] 分數{'不是' if not monotone else '★仍是★'}單調遞減")

vlm_client.close()
print("\n[VL] VDevice 已釋放。Block 1 完成。")
