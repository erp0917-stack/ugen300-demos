# -*- coding: utf-8 -*-
"""
face_helpers.py — prompt / 解析 / 三語標籤（雙模型流水線版）

VL 階段 : _build_vl_prompts()  / parse_vl_json()   → scores + obs
LLM 階段: _build_llm_prompts() / parse_llm_json()  → 完整報告 JSON
"""

import re
import json
import random
import hashlib


# ── OpenCC 工廠 ──────────────────────────────────────────────────────────
def _make_cc(mode):
    try:
        from opencc import OpenCC
        cc = OpenCC(mode)
        return lambda s: cc.convert(str(s))
    except Exception:
        return lambda s: str(s)


# ── 欄位常數 ──────────────────────────────────────────────────────────────
SCORE_KEYS  = ["symmetry", "proportion", "features", "skin", "contour"]
OBS_KEYS    = ["shape", "eyes", "nose", "mouth", "skin_state", "brow", "overall_impression"]
TIP_KEYS    = ["hair", "brow", "makeup", "skincare", "style"]
STRUCT_KEYS = ["shape", "forehead", "cheekbone", "jawline", "midface"]


# ════════════════════════════════════════════════════════════════
#  LANG — 三語標籤對照表
# ════════════════════════════════════════════════════════════════
LANG = {
    "tw": {
        "name": "繁體中文",
        "purify": _make_cc("s2t"),
        "labels": {
            # 評分維度
            "overall": "整體吸引力", "symmetry": "對稱度", "proportion": "五官比例",
            "features": "五官精緻度", "skin": "膚質", "contour": "輪廓",
            # 臉型結構
            "face_struct": "臉型與結構", "shape": "臉型",
            "forehead": "額頭", "cheekbone": "顴骨",
            "jawline": "下顎線", "midface": "中庭",
            # 各部位
            "parts": "各部位評估", "score_label": "評分", "note_label": "分析",
            # 優缺點建議
            "pros": "優勢", "cons": "可改善", "tips": "造型與保養建議",
            "hair": "髮型", "brow_tip": "眉型", "makeup": "妝容",
            "skincare": "保養", "style": "風格",
            # meta
            "gender": "性別", "age_range": "預估年齡", "summary": "整體短評",
            # 圖片
            "photo": "原照片", "sketch": "結構線稿",
            # 報告標題
            "report_title": "臉部美學分析報告",
            "report_sub":   "AI 顏值分析 · 娛樂用途",
            "scores_title": "五大維度評分",
        },
        "disclaimer": (
            "本報告由 AI 自動生成，僅供個人娛樂與參考，"
            "不代表任何專業美學、醫療或心理意見。"
        ),
    },
    "cn": {
        "name": "简体中文",
        "purify": _make_cc("t2s"),
        "labels": {
            "overall": "整体吸引力", "symmetry": "对称度", "proportion": "五官比例",
            "features": "五官精致度", "skin": "肤质", "contour": "轮廓",
            "face_struct": "脸型与结构", "shape": "脸型",
            "forehead": "额头", "cheekbone": "颧骨",
            "jawline": "下颌线", "midface": "中庭",
            "parts": "各部位评估", "score_label": "评分", "note_label": "分析",
            "pros": "优势", "cons": "可改善", "tips": "造型与保养建议",
            "hair": "发型", "brow_tip": "眉型", "makeup": "妆容",
            "skincare": "保养", "style": "风格",
            "gender": "性别", "age_range": "预估年龄", "summary": "整体短评",
            "photo": "原照片", "sketch": "结构线稿",
            "report_title": "脸部美学分析报告",
            "report_sub":   "AI 颜值分析 · 娱乐用途",
            "scores_title": "五大维度评分",
        },
        "disclaimer": (
            "本报告由 AI 自动生成，仅供个人娱乐与参考，"
            "不代表任何专业美学、医疗或心理意见。"
        ),
    },
    "en": {
        "name": "English",
        "purify": lambda s: str(s),
        "labels": {
            "overall": "Overall", "symmetry": "Symmetry", "proportion": "Proportion",
            "features": "Features", "skin": "Skin", "contour": "Contour",
            "face_struct": "Face Structure", "shape": "Shape",
            "forehead": "Forehead", "cheekbone": "Cheekbone",
            "jawline": "Jawline", "midface": "Mid-face",
            "parts": "Feature Analysis", "score_label": "Score", "note_label": "Note",
            "pros": "Strengths", "cons": "To Improve", "tips": "Style & Care Tips",
            "hair": "Hair", "brow_tip": "Brows", "makeup": "Makeup",
            "skincare": "Skincare", "style": "Style",
            "gender": "Gender", "age_range": "Est. Age", "summary": "Overview",
            "photo": "Photo", "sketch": "Structure Sketch",
            "report_title": "Facial Aesthetic Analysis",
            "report_sub":   "AI Beauty Analysis · For Entertainment",
            "scores_title": "5-Dimension Scores",
        },
        "disclaimer": (
            "This report is AI-generated for personal entertainment only and does not "
            "represent professional aesthetic, medical, or psychological advice."
        ),
    },
}


# ════════════════════════════════════════════════════════════════
#  工具函式
# ════════════════════════════════════════════════════════════════
def _to_int(v, default=60):
    try:
        return max(0, min(100, int(round(float(v)))))
    except Exception:
        return default


def _is_suspicious(vals):
    """五項分數全相同或呈完美等差遞減 → True。"""
    if len(vals) < 2:
        return False
    if len(set(vals)) == 1:
        return True
    sd = sorted(vals, reverse=True)
    if vals == sd:
        diffs = [sd[i] - sd[i+1] for i in range(len(sd)-1)]
        if len(set(diffs)) == 1:
            return True
    return False


def _perturb(vals):
    """每個值加 ±3~8 隨機擾動，確保結果在 0-100。"""
    return [max(0, min(100, v + random.choice([-1, 1]) * random.randint(3, 8)))
            for v in vals]


# ════════════════════════════════════════════════════════════════
#  VL 階段（Qwen2-VL-2B：只看圖吐事實）
# ════════════════════════════════════════════════════════════════
_VL_SKEL_TW = (
    '{"scores":{"symmetry":78,"proportion":72,"features":68,"skin":80,"contour":65},'
    '"obs":{"shape":"橢圓臉","eyes":"眼形修長、眼神有神",'
    '"nose":"鼻梁略低、鼻翼適中","mouth":"唇形飽滿、唇色健康",'
    '"skin_state":"膚色均勻、毛孔細緻","brow":"眉型自然、眉峰不明顯",'
    '"overall_impression":"整體氣質溫柔親切，五官精緻但輪廓感不強"}}'
)

_VL_SKEL_EN = (
    '{"scores":{"symmetry":78,"proportion":72,"features":68,"skin":80,"contour":65},'
    '"obs":{"shape":"oval","eyes":"almond-shaped, expressive gaze",'
    '"nose":"low bridge, moderate nostrils","mouth":"full lips, healthy color",'
    '"skin_state":"even tone, fine pores","brow":"natural arch, undefined peak",'
    '"overall_impression":"Warm and approachable; delicate features but lacks contour depth"}}'
)


def _build_vl_prompts(lang):
    """
    VL 模型只需要回答事實觀察，不要寫報告。
    回傳 (system_prompt, question)。
    cn 使用 tw 的 prompt（結果由 LLM 階段 OpenCC 轉換）。
    """
    skel = _VL_SKEL_EN if lang == "en" else _VL_SKEL_TW
    lang_rule = (
        "所有 obs 文字欄位用英文填寫，不要出現中文。"
        if lang == "en" else
        "所有 obs 文字欄位用繁體中文（台灣用語）填寫，不要簡體字。"
    )
    system = (
        "你是臉部觀察員，只描述照片中人臉的客觀特徵。\n"
        "【最重要】你只能輸出一個 JSON 物件：第一個字元必須是 { ，最後一個字元必須是 } 。\n"
        "輸出最後一個 } 之後立刻停止，不可以再寫任何說明。\n"
        "嚴禁 markdown、嚴禁 ``` 代碼框、嚴禁散文段落。\n"
        "JSON key 必須完全照範例的英文，不可更改。\n"
        "所有 scores 必須是 0~100 整數；依實際觀察給分，要有合理高低差異，"
        "禁止規律遞減（如 80/75/70/65/60）或全部相同。\n"
        f"文字語言：{lang_rule}\n"
        "現在直接輸出以下格式，obs 七個欄位都要填寫：\n"
        + skel
    )
    question = "請觀察照片中這個人的臉，直接輸出 JSON，不要任何其他文字。"
    return system, question


def _clean_vl_blob(txt):
    """
    清洗 VL 原始輸出裡的常見髒點：
    - 移除 markdown fence
    - 把字串值內的跳脫引號 \" 換成普通引號（或直接移除）
    - 移除字串值內的字面 \n \r \t
    - 把分數欄位裡混入的非數字字元（如 7?）清掉，只留數字
    """
    txt = txt.replace("```json", "").replace("```", "")

    # 把分數欄位的值清乾淨：只保留數字（處理 "proportion":7? 這類）
    def fix_score_val(m):
        digits = re.sub(r"[^\d]", "", m.group(1))
        return f':{digits}' if digits else ':60'
    for sk in SCORE_KEYS:
        txt = re.sub(rf'("{sk}"\s*:\s*)([0-9?]+)', fix_score_val, txt)

    # 移除字串值內的跳脫引號殘留（\" → 空）
    txt = txt.replace('\\"', '')
    # 移除字面的 \n \r \t（防止 json.loads 遇到 literal backslash-n）
    txt = re.sub(r'\\[nrt]', ' ', txt)

    return txt


def _rescue_scores(blob):
    """從髒 blob 逐一用 regex 抽分數，個位數（1-9）視為被截斷，×10 補回。"""
    result = {}
    for k in SCORE_KEYS:
        m = re.search(rf'"{k}"\s*:\s*(\d+)', blob)
        if m:
            v = int(m.group(1))
            if v < 10:          # 被截斷：70 → "7" 這類
                v = v * 10
            result[k] = max(0, min(100, v))
        else:
            result[k] = 60
    return result


def _rescue_obs(blob, purify):
    """
    對 obs 的每個 key 用寬鬆 regex 逐欄抽取，策略：
    抓到 key 後面第一個 " 開始，到下一個未跳脫的 " 或 , 或 } 前結束，
    取出中間的文字（含中文、標點）。
    """
    obs = {}
    for k in OBS_KEYS:
        # 先試標準 "key":"value" 格式
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', blob)
        if m and m.group(1).strip():
            obs[k] = purify(m.group(1).strip())
            continue
        # 寬鬆模式：key 後面抓到下一個雙引號 key 或 } 前的內容
        m2 = re.search(
            rf'"{k}"\s*:\s*"?([一-鿿\w\s、，。：；！？()（）\-–—\.]+)',
            blob
        )
        obs[k] = purify(m2.group(1).strip()) if m2 else ""
    return obs


def parse_vl_json(raw, purify):
    """
    解析 VL 輸出，回傳 (vl_data, strict)。
    vl_data = {"scores": {...}, "obs": {...}}
    strict=True 表示 json.loads 成功且 obs 有內容。
    """
    blob = _clean_vl_blob(raw)
    s, e = blob.find("{"), blob.rfind("}")
    blob = blob[s:e+1] if (s >= 0 and e > s) else blob

    result = {"scores": {k: 60 for k in SCORE_KEYS}, "obs": {k: "" for k in OBS_KEYS}}

    data = None
    try:
        data = json.loads(blob)
    except Exception:
        pass

    if isinstance(data, dict):
        sc = data.get("scores") or {}
        raw_scores = [_to_int(sc.get(k)) for k in SCORE_KEYS]
        # 個位數補回（被截斷的情況）
        raw_scores = [v * 10 if v < 10 else v for v in raw_scores]
        raw_scores = [max(0, min(100, v)) for v in raw_scores]
        if _is_suspicious(raw_scores):
            raw_scores = _perturb(raw_scores)
        for i, k in enumerate(SCORE_KEYS):
            result["scores"][k] = raw_scores[i]

        obs = data.get("obs") or {}
        obs_filled = any(obs.get(k, "").strip() for k in OBS_KEYS)
        if obs_filled:
            for k in OBS_KEYS:
                result["obs"][k] = purify(obs.get(k, ""))
            return result, True
        # json.loads 成功但 obs 全空 → 用 regex 從原始 blob 救援 obs
        result["obs"] = _rescue_obs(blob, purify)
        return result, False

    # json.loads 失敗：scores + obs 都用 regex 救
    raw_scores_d = _rescue_scores(blob)
    vals = [raw_scores_d[k] for k in SCORE_KEYS]
    if _is_suspicious(vals):
        vals = _perturb(vals)
    for i, k in enumerate(SCORE_KEYS):
        result["scores"][k] = vals[i]
    result["obs"] = _rescue_obs(blob, purify)
    return result, False


# ════════════════════════════════════════════════════════════════
#  LLM 輔助函式（normalize_report 依賴）
# ════════════════════════════════════════════════════════════════
_FS_KEYS    = ["shape", "forehead", "cheekbone", "jawline", "midface"]
_PART_NAMES_TW = ["眼睛", "眉毛", "鼻子", "嘴唇", "膚質", "輪廓"]
_PART_NAMES_EN = ["Eyes", "Brows", "Nose", "Lips", "Skin", "Contour"]

_FS_DEF_TW  = {"shape": "輪廓柔和", "forehead": "額部飽滿",
               "cheekbone": "顴骨適中", "jawline": "下顎線條柔和",
               "midface": "中庭比例均衡"}
_FS_DEF_EN  = {"shape": "Soft contour", "forehead": "Full forehead",
               "cheekbone": "Subtle cheekbones", "jawline": "Soft jawline",
               "midface": "Balanced mid-face"}
_TIP_DEF_TW = {"hair": "中長髮修飾臉型", "brow": "加強眉峰增立體",
               "makeup": "臥蠶與腮紅提氣色", "skincare": "維持保濕與防曬",
               "style": "柔和色系突顯親和力"}
_TIP_DEF_EN = {"hair": "Mid-length hair to frame face", "brow": "Define brow peak",
               "makeup": "Lash line and blush for brightness",
               "skincare": "Maintain moisturizer and SPF",
               "style": "Soft tones to highlight warmth"}


def _int_or(v, d):
    try:
        return int(re.search(r"\d+", str(v)).group())
    except Exception:
        return d


def _as_list(v, purify, limit=5):
    if isinstance(v, list):
        return [purify(str(x)) for x in v if str(x).strip()][:limit]
    return []


def _strip_punct(s):
    """去掉字串尾端的標點、空白，避免拼接時產生雙標點。"""
    return re.sub(r"[，。、,.\s]+$", "", str(s).strip())


# ── JhengHei 可渲染字元集（模組載入時初始化一次）───────────────────────
def _build_jheng_cmap():
    """
    回傳 Microsoft JhengHei 字型的 codepoint frozenset。
    優先用 fontTools 精確讀取 cmap；失敗則回傳 None（切換 regex 模式）。
    """
    try:
        from matplotlib import font_manager as _fm
        from fontTools.ttLib import TTFont as _TTFont
        prop = _fm.FontProperties(family="Microsoft JhengHei")
        path = _fm.findfont(prop)
        if not any(k in path.lower() for k in ("jhenghei", "msjh")):
            raise FileNotFoundError("JhengHei not found in system fonts")
        tt  = _TTFont(path, lazy=True)
        cps = set()
        for tbl in tt["cmap"].tables:
            cps.update(tbl.cmap.keys())
        tt.close()
        return frozenset(cps)
    except Exception:
        return None


_JHENG_CMAP = _build_jheng_cmap()

def _is_jheng_safe(cp):
    # 基本 CJK 統一表意文字（JhengHei 覆蓋完整）
    if 0x4e00 <= cp <= 0x9fff:
        return True
    # CJK 符號與標點（全型空格、〔〕《》等）
    if 0x3000 <= cp <= 0x303f:
        return True
    # 全形英數及標點（！？，。等）
    if 0xff00 <= cp <= 0xffef:
        return True
    # ASCII 可列印字元
    if 0x0020 <= cp <= 0x007e:
        return True
    # 常用 Unicode 標點：U+00B7 U+2014 U+2018 U+2019 U+201C U+201D
    if cp in (0x00b7, 0x2014, 0x2018, 0x2019, 0x201c, 0x201d):
        return True
    return False
    # CJK Extension A (U+3400-U+4DBF) 刻意排除：
    # JhengHei 對 Extension A 覆蓋不完整，
    # 排除後可過濾 U+3756 U+37C2 等生僻字，避免報告出現方塊。


def _sanitize_cjk(text):
    """
    移除 Microsoft JhengHei 無法渲染的字元，避免報告上出現方塊。
    優先 cmap 精確過濾；cmap 未載入則用 _is_jheng_safe 整數範圍判斷。
    不修改 VL 快取——只在 normalize_report 渲染前呼叫。
    """
    if not text:
        return text
    if _JHENG_CMAP is not None:
        return "".join(c for c in text if ord(c) in _JHENG_CMAP)
    return "".join(c for c in text if _is_jheng_safe(ord(c)))


def calibrate_score(raw):
    """把 VL 生分（0-100）線性映射到 65-99：raw=40→65，raw=75→99。"""
    try:
        raw = float(raw)
    except Exception:
        raw = 55
    cal = 65 + (raw - 40) * 34.0 / 35.0
    return int(max(65, min(99, round(cal))))


def _face_seed(vl_obs, vl_scores):
    """用臉的 obs+scores 內容生成穩定 seed，同一張臉永遠相同。"""
    s = str(sorted(vl_obs.items())) + str(sorted(vl_scores.items()))
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def _derive_part_scores(vl_scores, vl_obs):
    """
    依部位對應 VL score 衍生 ±擾動，校準到 65-99。
    使用臉內容 seed，同張臉結果穩定可複現。
    """
    rng = random.Random(_face_seed(vl_obs, vl_scores))
    base_map = [
        vl_scores.get("features", 65),  # 眼睛
        vl_scores.get("features", 65),  # 眉毛
        vl_scores.get("features", 65),  # 鼻子
        vl_scores.get("features", 65),  # 嘴唇
        vl_scores.get("skin",     65),  # 膚質
        vl_scores.get("contour",  65),  # 輪廓
    ]
    scores = [calibrate_score(b + rng.choice([-1, 1]) * rng.randint(2, 6))
              for b in base_map]
    # 確保不全相同、不完美遞減（用同一 rng 保持 reproducible）
    for _ in range(5):
        if not _is_suspicious(scores) and len(set(scores)) > 2:
            break
        scores = [calibrate_score(s - 40 + rng.choice([-1, 1]) * rng.randint(2, 6) + 40)
                  for s in scores]
    return scores


def normalize_report(data, vl_scores, vl_obs, purify, lang="tw"):
    """
    把任意形狀的 LLM 輸出 dict 正規化成固定 schema。
    永不崩、永不空欄：obs 衍生真實文字 > LLM 輸出 > 靜態罐頭。
    """
    use_en     = (lang == "en")
    part_names = _PART_NAMES_EN if use_en else _PART_NAMES_TW

    # ── obs 字元清洗（移除 JhengHei 畫不出的生僻/異體字）────────────
    # 只建立局部 clean_obs 副本，不修改傳入的 vl_obs（不影響 VL 快取）
    vl_obs = {k: _sanitize_cjk(str(v)) for k, v in vl_obs.items()}

    out = {}

    # ── meta ──────────────────────────────────────────────────────
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    out["meta"] = {
        "gender":    purify(str(meta.get("gender",  "Female" if use_en else "女性"))),
        "age_range": purify(str(meta.get("age_range", "25-35" if use_en else "25-35 歲"))),
    }

    # ── scores / overall（永遠來自 VL，每項過 calibrate_score）────
    cal_scores = {k: calibrate_score(v) for k, v in vl_scores.items()}
    out["scores"]  = cal_scores
    out["overall"] = round(sum(cal_scores.values()) / len(cal_scores))

    # ── summary（overall_impression 拼接，去尾端標點避免雙標點）──
    impression = _strip_punct(purify(str(vl_obs.get("overall_impression", ""))))
    if impression:
        out["summary"] = f"{impression}，五官協調。"
    else:
        llm_s = _sanitize_cjk(purify(str(data.get("summary", ""))).strip())
        out["summary"] = (llm_s if llm_s and llm_s not in ("...", "20-30字整體評價")
                          else ("Balanced features with a warm look."
                                if use_en else "整體五官協調、氣質溫和。"))

    # ── face_struct（shape 吃 obs，其餘 LLM → 文字預設）────────────
    fs = data.get("face_struct")
    fs_def = {
        "shape":     purify(_strip_punct(vl_obs.get("shape", ""))) or ("Soft contour" if use_en else "輪廓柔和"),
        "forehead":  "Full forehead"     if use_en else "額部飽滿",
        "cheekbone": "Subtle cheekbones" if use_en else "顴骨適中",
        "jawline":   "Soft jawline"      if use_en else "下顎線條柔和",
        "midface":   "Balanced mid-face" if use_en else "中庭比例均衡",
    }
    out["face_struct"] = {}
    for k in _FS_KEYS:
        val = _sanitize_cjk(purify(str(fs.get(k, ""))).strip()) if isinstance(fs, dict) else ""
        out["face_struct"][k] = (val if val and val not in ("...",) else "") or fs_def[k]

    # ── parts（分數衍生 + note 吃 obs）────────────────────────────
    parts_raw = data.get("parts")
    if not isinstance(parts_raw, list) or not parts_raw:
        parts_raw = fs if isinstance(fs, list) else []
    by_name = {purify(str(p.get("name", ""))): p
               for p in parts_raw if isinstance(p, dict) and p.get("name")}

    OBS_MAP_TW = {"眼睛": "eyes", "眉毛": "brow", "鼻子": "nose",
                  "嘴唇": "mouth", "膚質": "skin_state", "輪廓": "shape"}
    OBS_MAP_EN = {"Eyes": "eyes", "Brows": "brow", "Nose": "nose",
                  "Lips": "mouth", "Skin": "skin_state", "Contour": "shape"}
    obs_map    = OBS_MAP_EN if use_en else OBS_MAP_TW
    fallback   = "Natural features." if use_en else "表現自然、五官協調"

    derived_scores = _derive_part_scores(vl_scores, vl_obs)

    out["parts"] = []
    for i, nm in enumerate(part_names):
        p    = by_name.get(nm, {})
        # score：衍生分數已過校準（65-99），LLM 輸出直接校準後採用
        raw_sc = _int_or(p.get("score"), 0)
        score  = calibrate_score(raw_sc) if 50 <= raw_sc <= 100 else derived_scores[i]
        # note：LLM → obs → fallback（每層都過 _sanitize_cjk）
        note = _sanitize_cjk(purify(str(p.get("note", ""))).strip())
        if not note or len(note) < 2 or note in ("...", "短描述", "one-sentence analysis"):
            obs_key = obs_map.get(nm, "")
            # vl_obs 已在進入點整體清洗，直接取用
            note    = purify(str(vl_obs.get(obs_key, ""))).strip() or fallback
        out["parts"].append({"name": nm, "score": score, "note": note})

    # ── pros / cons（obs 衍生長句，不採 LLM 短輸出）──────────────
    def _tw_join(obs_key, default_obs, suffix):
        """obs 非空 → 「obs，suffix」；空 → 只回 suffix。"""
        part = _strip_punct(purify(str(vl_obs.get(obs_key) or default_obs)))
        return f"{part}，{suffix}" if part else suffix

    def _en_join(obs_key, default_obs, suffix):
        part = _strip_punct(str(vl_obs.get(obs_key) or default_obs))
        return f"{part} — {suffix}" if part else suffix

    if not use_en:
        _sh = _strip_punct(purify(str(vl_obs.get('shape') or '輪廓柔和')))
        out["pros"] = [
            _tw_join('skin_state', '膚質均勻', '膚質是整體最大亮點'),
            _tw_join('eyes',       '眼神有神', '目光自帶親和力'),
            f"臉型{_sh}、輪廓柔順，顯得溫婉耐看" if _sh else "輪廓柔順，顯得溫婉耐看",
            _tw_join('overall_impression', '氣質溫和', '五官分佈協調'),
        ]
        out["cons"] = [
            _tw_join('brow', '眉峰不明顯', '可透過修眉強化立體輪廓'),
            _tw_join('nose', '鼻樑略低',   '打亮或修容能增加挺度與深邃感'),
            "中庭比例稍長，瀏海或妝容可平衡視覺重心",
            "表情偏靜態，微笑時更能帶出親和力",
        ]
    else:
        _sh_en = _strip_punct(str(vl_obs.get('shape') or 'oval'))
        out["pros"] = [
            _en_join('skin_state',         'Even skin',      'skin is the standout strength'),
            _en_join('eyes',               'Expressive eyes','gaze carries natural warmth'),
            f"Face shape {_sh_en} with soft contours, timelessly attractive" if _sh_en else "Soft contours, timelessly attractive",
            _en_join('overall_impression', 'Balanced',       'features are well-distributed'),
        ]
        out["cons"] = [
            _en_join('brow', 'Brow peak undefined', 'defining arch adds depth'),
            _en_join('nose', 'Low nose bridge',     'contouring can enhance definition'),
            "Mid-face ratio slightly long; fringe or makeup can balance visual weight",
            "Expression tends static; a natural smile greatly boosts approachability",
        ]

    # ── tips（A/B 情境對照模板）───────────────────────────────────
    if not use_en:
        out["tips"] = {
            "hair":     "柔和取向：中長微捲修飾臉型；俐落取向：齊肩直髮顯精神",
            "brow":     "自然感：順毛眉維持柔和；立體感：加強眉峰增輪廓",
            "makeup":   "日常：臥蠶與裸色提氣色；正式：腮紅與眼線增神采",
            "skincare": "基礎：保濕鎖水為主；進階：加強防曬與抗老",
            "style":    "親和路線：柔和粉色系；知性路線：低彩度中性色",
        }
    else:
        out["tips"] = {
            "hair":     "Soft: mid-length wavy to frame face; Sharp: shoulder-length straight for polish",
            "brow":     "Natural: follow hair grain for softness; Defined: arch peak for structure",
            "makeup":   "Daily: lower lash highlight + nude tone; Formal: blush + liner for impact",
            "skincare": "Basic: prioritize hydration; Advanced: add SPF and anti-aging routine",
            "style":    "Approachable: soft pastel tones; Intellectual: low-saturation neutrals",
        }

    return out


# ════════════════════════════════════════════════════════════════
#  LLM 階段（Qwen2.5-1.5B：把觀察擴寫成完整報告）
# ════════════════════════════════════════════════════════════════
_LLM_SKEL_TW = (
    '{"meta":{"gender":"女性","age_range":"25-35 歲"},'
    '"summary":"輪廓柔和、五官精緻，皮膚狀態是最大亮點，比例稍有不足。",'
    '"face_struct":{"shape":"鵝蛋臉","forehead":"額頭適中、弧度自然",'
    '"cheekbone":"顴骨不明顯、臉頰圓潤","jawline":"下顎線偏圓潤、缺立體感",'
    '"midface":"中庭略長、使五官比例稍失衡"},'
    '"parts":['
    '{"name":"眼睛","score":76,"note":"眼形修長、眼神有神，眼距稍寬顯親切"},'
    '{"name":"眉毛","score":68,"note":"眉型自然但眉峰不明顯，略顯平淡"},'
    '{"name":"鼻子","score":65,"note":"鼻梁略低、鼻翼適中，整體偏平"},'
    '{"name":"嘴唇","score":72,"note":"唇形飽滿自然，唇色健康"},'
    '{"name":"膚質","score":82,"note":"膚色均勻細緻，是整體最大亮點"},'
    '{"name":"輪廓","score":66,"note":"下顎線柔和，缺乏雕塑感"}],'
    '"pros":["皮膚狀態極佳、膚色均勻細緻","眼神有神、眼形修長","唇形自然飽滿",'
    '"整體氣質溫柔親切，親和力強"],'
    '"cons":["鼻梁略低影響立體感","眉峰不夠明顯使五官顯平","下顎線條缺乏輪廓感",'
    '"髮型遮住額頭使臉部比例顯短"],'
    '"tips":{"hair":"建議露出額頭拉長臉部比例","brow":"修飾眉峰增加立體感",'
    '"makeup":"淡妝重點放在眼線與腮紅","skincare":"加強保濕維持現有膚況",'
    '"style":"簡約休閒或知性風格最適合"}}'
)

_LLM_SKEL_EN = (
    '{"meta":{"gender":"Female","age_range":"25-35"},'
    '"summary":"Soft contours with delicate features; skin is the standout strength.",'
    '"face_struct":{"shape":"Oval","forehead":"Medium, natural arc",'
    '"cheekbone":"Subtle, rounded cheeks","jawline":"Rounded, lacks definition",'
    '"midface":"Slightly long, affecting proportion"},'
    '"parts":['
    '{"name":"Eyes","score":76,"note":"Almond-shaped with expressive gaze, slightly wide-set"},'
    '{"name":"Brows","score":68,"note":"Natural but undefined arch, looks flat"},'
    '{"name":"Nose","score":65,"note":"Low bridge, moderate nostrils, overall flat"},'
    '{"name":"Lips","score":72,"note":"Full and natural lip shape, healthy color"},'
    '{"name":"Skin","score":82,"note":"Even tone and fine texture, standout feature"},'
    '{"name":"Contour","score":66,"note":"Soft jawline, lacks sculpted definition"}],'
    '"pros":["Excellent skin condition and even tone","Expressive almond-shaped eyes",'
    '"Naturally full lips","Warm and approachable impression"],'
    '"cons":["Low nose bridge reduces depth","Undefined brow peak flattens features",'
    '"Jawline lacks sculpted definition","Hairstyle covering forehead shortens face ratio"],'
    '"tips":{"hair":"Show forehead to elongate face ratio","brow":"Define brow peak for dimension",'
    '"makeup":"Focus on liner and blush","skincare":"Maintain with moisturizer and SPF",'
    '"style":"Smart casual or intellectual styles suit best"}}'
)


def _build_llm_prompts(lang, vl_data):
    """
    使用填空骨架，只要求 LLM 填 summary / face_struct / parts / pros / cons / tips。
    scores / overall / meta 由 normalize_report 從 VL 數據補入，不要求 LLM 輸出。
    cn 使用 tw prompt，後續 OpenCC t2s 轉換。
    """
    obs    = vl_data["obs"]
    scores = vl_data["scores"]
    use_en = (lang == "en")

    if use_en:
        lang_rule = "All text fields must be written in English only. No Chinese."
        obs_block = (
            f"Face observations from image analysis:\n"
            f"- Shape: {obs['shape']}\n"
            f"- Eyes: {obs['eyes']}\n"
            f"- Nose: {obs['nose']}\n"
            f"- Mouth: {obs['mouth']}\n"
            f"- Skin: {obs['skin_state']}\n"
            f"- Brows: {obs['brow']}\n"
            f"- Overall: {obs['overall_impression']}\n"
            f"Reference scores: symmetry={scores['symmetry']}, "
            f"proportion={scores['proportion']}, features={scores['features']}, "
            f"skin={scores['skin']}, contour={scores['contour']}\n\n"
        )
        skeleton = (
            'Fill in every field and output ONLY this JSON:\n'
            '{\n'
            '  "summary": "20-30 word overall impression",\n'
            '  "face_struct": {"shape":"describe face shape","forehead":"...","cheekbone":"...","jawline":"...","midface":"..."},\n'
            '  "parts": [\n'
            '    {"name":"Eyes",   "score":76,"note":"one-sentence analysis"},\n'
            '    {"name":"Brows",  "score":68,"note":"..."},\n'
            '    {"name":"Nose",   "score":65,"note":"..."},\n'
            '    {"name":"Lips",   "score":72,"note":"..."},\n'
            '    {"name":"Skin",   "score":80,"note":"..."},\n'
            '    {"name":"Contour","score":70,"note":"..."}\n'
            '  ],\n'
            '  "pros": ["...","...","...","..."],\n'
            '  "cons": ["...","...","...","..."],\n'
            '  "tips": {"hair":"...","brow":"...","makeup":"...","skincare":"...","style":"..."}\n'
            '}\n'
            'Note: face_struct is text descriptions only (no scores); '
            'parts has scores. Both must be filled.'
        )
    else:
        lang_rule = "所有文字欄位用繁體中文（台灣用語），不要簡體字、不要英文。"
        obs_block = (
            f"臉部觀察資料（由看圖模型提供）：\n"
            f"- 臉型：{obs['shape']}\n"
            f"- 眼睛：{obs['eyes']}\n"
            f"- 鼻子：{obs['nose']}\n"
            f"- 嘴唇：{obs['mouth']}\n"
            f"- 膚質：{obs['skin_state']}\n"
            f"- 眉毛：{obs['brow']}\n"
            f"- 整體印象：{obs['overall_impression']}\n"
            f"評分參考：symmetry={scores['symmetry']}, proportion={scores['proportion']}, "
            f"features={scores['features']}, skin={scores['skin']}, "
            f"contour={scores['contour']}\n\n"
        )
        skeleton = (
            '請嚴格輸出以下 JSON，所有欄位都要填、不可省略：\n'
            '{\n'
            '  "summary": "20-30字整體評價",\n'
            '  "face_struct": {"shape":"臉型描述","forehead":"...","cheekbone":"...","jawline":"...","midface":"..."},\n'
            '  "parts": [\n'
            '    {"name":"眼睛","score":76,"note":"短描述"},\n'
            '    {"name":"眉毛","score":68,"note":"..."},\n'
            '    {"name":"鼻子","score":65,"note":"..."},\n'
            '    {"name":"嘴唇","score":72,"note":"..."},\n'
            '    {"name":"膚質","score":80,"note":"..."},\n'
            '    {"name":"輪廓","score":70,"note":"..."}\n'
            '  ],\n'
            '  "pros": ["...","...","...","..."],\n'
            '  "cons": ["...","...","...","..."],\n'
            '  "tips": {"hair":"...","brow":"...","makeup":"...","skincare":"...","style":"..."}\n'
            '}\n'
            '注意：face_struct 是臉型結構純文字描述、不打分；'
            'parts 是六個部位要打分。兩者不同，都要填。'
        )

    system = (
        "你是專業臉部美學分析師（娛樂用途）。\n"
        "【最重要】你只能輸出一個 JSON 物件：第一個字元必須是 { ，最後一個字元必須是 } 。\n"
        "輸出最後一個 } 之後立刻停止，不可以再寫任何說明或段落。\n"
        "嚴禁 markdown、嚴禁 ``` 代碼框。\n"
        "JSON key 必須完全照骨架的英文，不可更改。所有 score 必須是 0~100 整數。\n"
        "pros、cons 各要 4 點，cons 不可為空。語氣專業而親和（美容顧問口吻）。\n"
        f"語言要求：{lang_rule}\n"
    )
    user_msg = obs_block + skeleton
    return system, user_msg


def parse_llm_json(raw, vl_scores, vl_obs, purify, lang="tw"):
    """
    解析 LLM 輸出，回傳 (report_data, strict)。
    永不崩、永不空欄：透過 normalize_report 兜底。
    vl_scores / vl_obs 用於填補 scores / obs 來源的預設值。
    """
    txt = raw.replace("```json", "").replace("```", "")
    s, e = txt.find("{"), txt.rfind("}")
    blob = txt[s:e+1] if (s >= 0 and e > s) else txt

    strict = False
    data   = {}
    try:
        data   = json.loads(blob)
        strict = isinstance(data, dict)
    except Exception:
        pass

    result = normalize_report(data if isinstance(data, dict) else {},
                               vl_scores, vl_obs, purify, lang=lang)
    return result, strict


def merge_vl_llm(vl_data, llm_data):
    """
    normalize_report 已把 scores/overall 從 vl_scores 填入 llm_data，
    直接回傳 llm_data 即為最終 data dict。
    """
    return llm_data
