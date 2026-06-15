# -*- coding: utf-8 -*-
"""
test_face_json.py — 顏值報告 第一步驗證（v5：強制 cons 非空 + 壓掉結尾散文）
======================================================================
v4 繁中已能一次吐乾淨 JSON，只差 cons（可改善）被模型留空。v5 在 prompt 硬性
要求 cons 一定要列 2 點，並要求輸出 } 後立即停止（不要再寫散文）。

放在 V2 資料夾，照片同層 face.jpg。
用法：
    python test_face_json.py face.jpg tw    繁體中文
    python test_face_json.py face.jpg cn    简体中文
    python test_face_json.py face.jpg en    English
"""

import sys
import re
import json

import cv2

from vlm_client import ask_about_image


def _make_cc(mode):
    try:
        from opencc import OpenCC
        cc = OpenCC(mode)
        return lambda s: cc.convert(str(s))
    except Exception:
        return lambda s: str(s)


def _skeleton(example):
    return (
        '{'
        '"overall":75,'
        '"scores":{"symmetry":70,"proportion":70,"features":70,"skin":70,"contour":70},'
        f'"pros":["{example[0]}","{example[1]}"],'
        f'"cons":["{example[2]}","{example[3]}"],'
        f'"tips":{{"hair":"{example[4]}","skincare":"{example[5]}","style":"{example[6]}"}}'
        '}'
    )


LANG = {
    "tw": {
        "name": "繁體中文",
        "rule": "pros、cons、tips 裡的文字全部用繁體中文（台灣用語），不要簡體字、不要英文。",
        "skeleton": _skeleton(["眼神有神", "笑容自然", "可多注意保濕", "髮型可更俐落",
                               "建議露出額頭", "加強防曬", "簡約休閒風"]),
        "labels": {"overall": "整體吸引力", "symmetry": "對稱度", "proportion": "比例",
                   "features": "五官", "skin": "皮膚", "contour": "輪廓",
                   "pros": "優點", "cons": "可改善", "tips": "個人化建議",
                   "hair": "髮型", "skincare": "保養", "style": "風格"},
        "purify": _make_cc("s2t"),
    },
    "cn": {
        "name": "简体中文",
        "rule": "pros、cons、tips 里的文字全部用简体中文，不要繁体字、不要英文。",
        "skeleton": _skeleton(["眼神有神", "笑容自然", "可多注意保湿", "发型可更利落",
                               "建议露出额头", "加强防晒", "简约休闲风"]),
        "labels": {"overall": "整体吸引力", "symmetry": "对称度", "proportion": "比例",
                   "features": "五官", "skin": "皮肤", "contour": "轮廓",
                   "pros": "优点", "cons": "可改善", "tips": "个性化建议",
                   "hair": "发型", "skincare": "保养", "style": "风格"},
        "purify": _make_cc("t2s"),
    },
    "en": {
        "name": "English",
        "rule": "pros、cons、tips 裡的文字全部用英文（English）填寫，不要出現中文。",
        "skeleton": _skeleton(["bright eyes", "natural smile", "improve hydration",
                               "neater hairstyle", "show forehead", "more sunscreen",
                               "smart casual"]),
        "labels": {"overall": "Overall", "symmetry": "Symmetry", "proportion": "Proportion",
                   "features": "Features", "skin": "Skin", "contour": "Contour",
                   "pros": "Pros", "cons": "To Improve", "tips": "Suggestions",
                   "hair": "Hair", "skincare": "Skincare", "style": "Style"},
        "purify": lambda s: str(s),
    },
}

SCORE_KEYS = ["symmetry", "proportion", "features", "skin", "contour"]
TIP_KEYS = ["hair", "skincare", "style"]


def _build_prompts(lang):
    cfg = LANG[lang]
    system = (
        "你是臉部美學分析師（娛樂用途），看照片做誠實、中肯的分析。\n"
        "【最重要】你只能輸出一個 JSON 物件：第一個字元必須是 { ，最後一個字元必須是 } 。\n"
        "輸出最後一個 } 之後立刻停止，不可以再寫任何說明、評論或段落。\n"
        "嚴禁 markdown、嚴禁出現 ### 或 ** 或 ``` 代碼框。\n"
        "JSON 的 key 必須完全照下面範例的英文，不可更改。\n"
        "所有分數欄位必須是 0 到 100 的整數，不可以是字母或文字。\n"
        "誠實評分，不要每一項都給高分。\n"
        "pros（優點）要列 2 點；cons（可改善）也一定要列 2 點——"
        "就算整體不錯，也要誠實指出可以再提升的小地方，cons 絕對不可以是空的 []。\n"
        f"文字語言要求：{cfg['rule']}\n"
        "現在直接輸出以下格式，每個欄位都要填滿（特別是 cons）：\n" + cfg["skeleton"]
    )
    question = "請分析照片中這個人的臉。現在只輸出那個 JSON，pros 和 cons 都要各列 2 點，不要寫任何別的文字。"
    return system, question


def _to_int(v, default=60):
    try:
        return max(0, min(100, int(round(float(v)))))
    except Exception:
        return default


def parse_face_json(raw, purify):
    txt = raw.replace("```json", "").replace("```", "")
    s, e = txt.find("{"), txt.rfind("}")
    blob = txt[s:e + 1] if (s >= 0 and e > s) else txt

    result = {"overall": 60, "scores": {k: 60 for k in SCORE_KEYS},
              "pros": [], "cons": [], "tips": {}}

    data = None
    try:
        data = json.loads(blob)
    except Exception:
        data = None

    if isinstance(data, dict):
        result["overall"] = _to_int(data.get("overall"))
        sc = data.get("scores") or {}
        for k in SCORE_KEYS:
            result["scores"][k] = _to_int(sc.get(k))
        result["pros"] = [purify(x) for x in (data.get("pros") or [])][:3]
        result["cons"] = [purify(x) for x in (data.get("cons") or [])][:3]
        tp = data.get("tips") or {}
        for k in TIP_KEYS:
            if tp.get(k):
                result["tips"][k] = purify(tp.get(k))
        return result, True

    def gi(name, d=60):
        m = re.search(rf'{name}\D*?(\d{{1,3}})', blob)
        return _to_int(m.group(1), d) if m else d

    def gl(name):
        m = re.search(rf'{name}["\s]*[:：]\s*\[([^\]]*)\]', blob)
        return [purify(x) for x in re.findall(r'"([^"]+)"', m.group(1))][:3] if m else []

    result["overall"] = gi("overall")
    for k in SCORE_KEYS:
        result["scores"][k] = gi(k)
    result["pros"] = gl("pros")
    result["cons"] = gl("cons")
    return result, False


def main():
    if len(sys.argv) < 2:
        print("用法：python test_face_json.py face.jpg [tw|cn|en]")
        return
    img_path = sys.argv[1]
    lang = sys.argv[2].lower() if len(sys.argv) >= 3 else "tw"
    if lang not in LANG:
        print(f"語言只能是 tw / cn / en，你給的是 {lang}")
        return

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[X] 讀不到照片：{img_path}")
        return

    cfg = LANG[lang]
    system, question = _build_prompts(lang)
    print(f"[OK] 照片 {img_path} {frame.shape[1]}x{frame.shape[0]}，語言＝{cfg['name']}")
    print("送進 Qwen2-VL 分析中（第一次載入約 10 秒）……\n")

    raw = ask_about_image(frame, question, system_prompt=system,
                          max_tokens=420, temperature=0.15)

    print("===== VLM 原始輸出 =====")
    print(raw)

    data, strict = parse_face_json(raw, cfg["purify"])
    L = cfg["labels"]
    print("\n===== 解析後（這就是報告畫面會顯示的內容）=====")
    print(f"（解析：{'JSON 一次過' if strict else 'regex 救援'}）")
    print(f"{L['overall']}: {data['overall']}")
    for k in SCORE_KEYS:
        print(f"  {L[k]}: {data['scores'][k]}")
    print(f"{L['pros']}: {data['pros']}")
    print(f"{L['cons']}: {data['cons']}")
    print(f"{L['tips']}: {data['tips']}")

    scores_ok = all(isinstance(v, int) for v in data["scores"].values())
    text_ok = bool(data["pros"]) and bool(data["cons"])
    print(f"\n[{'V' if scores_ok else 'X'}] 分數都是整數   "
          f"[{'V' if text_ok else 'X'}] 優點/可改善有內容")
    if scores_ok and text_ok:
        print(f"[V] {cfg['name']} 可用。")
    else:
        print(f"[!] {cfg['name']} 還有缺，把原始輸出貼回來我調。")


if __name__ == "__main__":
    main()
