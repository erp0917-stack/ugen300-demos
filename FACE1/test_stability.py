# -*- coding: utf-8 -*-
"""
test_stability.py — 連跑兩次，確認分數穩定且落在 65-99。
第一次：VL 推論 + 寫入快取。第二次：讀快取，分數完全相同。
用法：python test_stability.py [圖片路徑]
"""
import sys
import cv2
import vlm_client
import llm_client
from face_helpers import (
    LANG, _build_vl_prompts, parse_vl_json,
    _build_llm_prompts, parse_llm_json, merge_vl_llm,
)
from face_report import detect_and_crop, run_vl

img = sys.argv[1] if len(sys.argv) > 1 else r"..\V2\face.jpg"
frame = cv2.imread(img)
crop, _ = detect_and_crop(frame)
purify = LANG["tw"]["purify"]

results = []
for run in range(1, 3):
    print(f"\n{'='*50}")
    print(f"Run {run}")
    print('='*50)

    # run_vl 內部已含快取邏輯，第2次直接讀快取不呼叫 VDevice
    vl_data = run_vl(crop, "tw")
    try:
        vlm_client.close()   # 快取命中時 VDevice 未開啟，close() 是 no-op
    except Exception:
        pass
    print(f"VL raw scores: {vl_data['scores']}")

    sys_l, u_l = _build_llm_prompts("tw", vl_data)
    raw_llm = llm_client.generate(u_l, system_prompt=sys_l,
                                   max_tokens=800, temperature=0.0)
    llm_data, _ = parse_llm_json(raw_llm, vl_data["scores"], vl_data["obs"],
                                  purify, lang="tw")
    llm_client.close()

    data = merge_vl_llm(vl_data, llm_data)
    part_scores = [p["score"] for p in data["parts"]]

    print(f"overall : {data['overall']}")
    print(f"scores  : {data['scores']}")
    print(f"parts   : {part_scores}")

    all_in_range = all(65 <= v <= 99 for v in data["scores"].values())
    all_in_range &= all(65 <= s <= 99 for s in part_scores)
    print(f"65-99?  : {'YES' if all_in_range else 'NO -- OUT OF RANGE'}")
    results.append((data["overall"], data["scores"], part_scores))

print("\n" + "="*50)
if results[0] == results[1]:
    print("[V] 兩次分數完全一致 — 穩定性確認！")
else:
    print("[X] 分數不一致（注意：若 LLM 輸出不穩定會影響 parts note，但 scores/overall 應一致）")
    print(f"  Run1 overall={results[0][0]}  scores={results[0][1]}")
    print(f"  Run2 overall={results[1][0]}  scores={results[1][1]}")
    # 只驗 scores + overall（由 VL 決定）是否一致
    vl_stable = (results[0][0] == results[1][0]) and (results[0][1] == results[1][1])
    print(f"  VL-derived scores/overall 一致: {'[V] YES' if vl_stable else '[X] NO'}")
