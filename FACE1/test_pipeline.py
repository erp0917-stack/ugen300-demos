# -*- coding: utf-8 -*-
"""
test_pipeline.py — Block 2 驗證
流程：裁臉 → VL (Qwen2-VL-2B) → close → LLM (Qwen2.5-1.5B) → close → 印合併 JSON

用法：
    python test_pipeline.py
    python test_pipeline.py ..\V2\face.jpg tw
"""

import sys
import json
import cv2

import vlm_client
import llm_client
from face_helpers import (
    LANG, SCORE_KEYS, TIP_KEYS, STRUCT_KEYS,
    _build_vl_prompts, parse_vl_json,
    _build_llm_prompts, parse_llm_json,
    merge_vl_llm,
)
from face_report import detect_and_crop

img_path = sys.argv[1] if len(sys.argv) > 1 else r"..\V2\face.jpg"
lang     = sys.argv[2] if len(sys.argv) > 2 else "tw"

# ── 讀圖 + 偵臉 ──────────────────────────────────────────────────────────
frame = cv2.imread(img_path)
if frame is None:
    print(f"[X] 讀不到圖片：{img_path}"); sys.exit(1)
print(f"[OK] {img_path}  {frame.shape[1]}x{frame.shape[0]}  lang={lang}")
crop, found = detect_and_crop(frame)
print(f"[偵臉] crop {crop.shape[1]}x{crop.shape[0]}  found={found}")

# ── VL 階段 ───────────────────────────────────────────────────────────────
vl_lang = "tw" if lang == "cn" else lang
system_vl, question_vl = _build_vl_prompts(vl_lang)
import hashlib, numpy as np
_digest = hashlib.md5(np.ascontiguousarray(crop).tobytes()).hexdigest()
_img_seed = int(_digest, 16) % (2 ** 31)
print("\n[VL] 載入 Qwen2-VL-2B……")
raw_vl = vlm_client.ask_about_image(crop, question_vl, system_prompt=system_vl,
                                     max_tokens=400, temperature=0.10,
                                     seed=_img_seed)
print("===== VL 原始輸出 ====="); print(raw_vl)

vl_data, vl_strict = parse_vl_json(raw_vl, LANG[vl_lang]["purify"])
print(f"\n[VL] 解析：{'JSON 一次過' if vl_strict else 'regex 救援'}")
print(f"scores: {vl_data['scores']}")
for k, v in vl_data["obs"].items():
    print(f"  obs.{k}: {v}")

print("\n[VL] 釋放 VDevice……")
vlm_client.close()
print("[VL] Done.\n")

# ── LLM 階段 ──────────────────────────────────────────────────────────────
llm_lang = "tw" if lang == "cn" else lang
system_llm, user_msg = _build_llm_prompts(llm_lang, vl_data)
print("[LLM] 載入 Qwen2.5-1.5B……")
raw_llm = llm_client.generate(user_msg, system_prompt=system_llm,
                               max_tokens=800, temperature=0.15)
print("===== LLM 原始輸出 ====="); print(raw_llm)

purify = LANG[lang]["purify"]
llm_data, llm_strict = parse_llm_json(raw_llm, vl_data["scores"], vl_data["obs"], purify, lang=lang)
print(f"\n[LLM] 解析：{'JSON 一次過' if llm_strict else 'regex 救援'}")

print("\n[LLM] 釋放 VDevice……")
llm_client.close()
print("[LLM] Done.\n")

# ── 合併 + 驗證 ───────────────────────────────────────────────────────────
data = merge_vl_llm(vl_data, llm_data)

print("=" * 60)
print("合併後完整 JSON")
print("=" * 60)
print(json.dumps(data, ensure_ascii=False, indent=2))

print("\n" + "=" * 60)
print("欄位驗證")
print("=" * 60)

checks = {
    "overall 是整數":         isinstance(data["overall"], int),
    "scores 五項都是整數":     all(isinstance(v, int) for v in data["scores"].values()),
    "meta.gender 非空":       bool(data["meta"].get("gender")),
    "meta.age_range 非空":    bool(data["meta"].get("age_range")),
    "summary 非空":           bool(data["summary"]),
    "face_struct 五項非空":   all(data["face_struct"].get(k) for k in STRUCT_KEYS),
    "parts 有 6 筆":          len(data["parts"]) == 6,
    "parts 每筆有 note":      all(p.get("note") for p in data["parts"]),
    "pros 有 4+ 點":          len(data["pros"]) >= 4,
    "cons 有 4+ 點（不可空）": len(data["cons"]) >= 4,
    "tips 有 5 類":           all(k in data["tips"] for k in TIP_KEYS),
}

all_pass = True
for label, ok in checks.items():
    mark = "V" if ok else "X"
    print(f"  [{mark}] {label}")
    if not ok:
        all_pass = False

print()
print("[V] 全部通過！" if all_pass else "[!] 有欄位未通過，把 LLM 原始輸出貼回來調。")
