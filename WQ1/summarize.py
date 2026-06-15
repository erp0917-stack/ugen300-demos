# -*- coding: utf-8 -*-
"""
summarize.py
============
把逐字稿丟給 UGen300 / Hailo-10H 上的 Qwen2.5-1.5B-Instruct，
回傳「會議重點 + 待辦事項 + 決議事項」三段式繁體中文摘要。

直接走 HailoRT GenAI Python API（hailo_platform.genai.LLM），
不需要 Hailo-Ollama 伺服器。模型在第一次呼叫時載入一次（約 8 秒），
之後同一個 process 內重複呼叫共用同一個 VDevice 與 LLM。

對 meeting_summary.py 而言，公開介面仍是同一支：
    summarize(transcript) -> str

公開的釋放函式：
    close()  # 主程式跑完 transcribe 後，必須在叫 summarize 之前確保 transcribe.close()
             # 把 Whisper 的 VDevice 釋放掉（Hailo-10H 一次只給一個 VDevice）
"""
import atexit
import os
import re
import threading
import time
import traceback
from pathlib import Path

try:
    from hailo_platform import VDevice
    from hailo_platform.genai import LLM
    _HAS_HAILO = True
except Exception:
    _HAS_HAILO = False


# ── 設定 ────────────────────────────────────────────────────────────────
_DEFAULT_HEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen2.5-1.5B-Instruct.hef")
HEF_PATH = os.environ.get("HAILO_LLM_HEF", _DEFAULT_HEF)

# Round 1 已驗證的 system 規則（五條硬限制，壓掉錯字/亂詞）
SYSTEM_PROMPT = (
    "你是專業的會議記錄助理。\n"
    "\n"
    "嚴格規則：\n"
    "(1) 只用標準繁體中文，禁止使用簡體字（例如禁止『区』『决』『发』『钟』『话』『听』『说』等）。\n"
    "(2) 禁止使用非標準詞彙或自創縮寫；用詞必須完整正確。"
    "範例：要寫『負責人』不是『責人』、『決議事項』不是『暗決』、"
    "『工作內容』不是『發動內容』、『期限』不是『頻期』也不是『長度』。\n"
    "(3) 一律照範例的固定格式輸出，章節順序為「會議重點 → 待辦事項 → 決議事項」，"
    "不可更動章節名與項目名。\n"
    "(4) 沒提到的資訊在該欄位寫『未提及』，禁止編造資訊或人名。\n"
    "(5) 不要加開場白、結語、客套話、或自我評論。直接輸出三段式摘要。"
)

# 1-shot 範例（刻意跟產品發表會 demo domain 不同，避免事實洩漏到實際答案）
_EXAMPLE_TRANSCRIPT = (
    "好今天主要講Q4行銷預算。第一，請大華在十月底前把 KPI 報告交給我。"
    "第二，社群這邊我們決定先暫停 LinkedIn 廣告，集中火力在 Instagram。"
    "第三，總預算這季控制在五十萬以內。OK 散會。"
)
_EXAMPLE_OUTPUT = (
    "會議重點：\n"
    "- 討論 Q4 行銷預算配置。\n"
    "- 確認 KPI 報告交付時程。\n"
    "- 決定社群廣告平台策略。\n"
    "\n"
    "待辦事項：\n"
    "- 負責人：大華　工作內容：交付 KPI 報告　期限：十月底前\n"
    "\n"
    "決議事項：\n"
    "- 暫停 LinkedIn 廣告，社群預算集中於 Instagram。\n"
    "- Q4 總預算上限五十萬。"
)

# 取樣參數（round 1 已驗證組合）
_TEMPERATURE = 0.1
_TOP_P = 0.85
_TOP_K = 30
_FREQ_PENALTY = 1.15
_SEED = 42
_MAX_TOKENS = 400

# ── 英文版 prompt / 範例（逐字稿是英文時用）──────────────────────────
SYSTEM_PROMPT_EN = (
    "You are a professional meeting-notes assistant.\n"
    "\n"
    "Strict rules:\n"
    "(1) Write ONLY in English. Use clear, correct words.\n"
    "(2) Always follow the example's fixed format. Section order must be "
    "\"Key Points -> Action Items -> Decisions\". Do not change the section names.\n"
    "(3) If a field has no information, write \"Not mentioned\". Never invent "
    "facts or names.\n"
    "(4) No greetings, no closing remarks, no self-comments. Output the "
    "three sections directly."
)
_EXAMPLE_TRANSCRIPT_EN = (
    "Okay, today we mainly discuss the Q4 marketing budget. First, please "
    "have John submit the KPI report by the end of October. Second, for social, "
    "we decided to pause LinkedIn ads and focus on Instagram. Third, keep the "
    "total budget under five hundred thousand this quarter. OK, meeting adjourned."
)
_EXAMPLE_OUTPUT_EN = (
    "Key Points:\n"
    "- Discussed Q4 marketing budget allocation.\n"
    "- Confirmed KPI report delivery schedule.\n"
    "- Decided social ad platform strategy.\n"
    "\n"
    "Action Items:\n"
    "- Owner: John  Task: Submit KPI report  Deadline: end of October\n"
    "\n"
    "Decisions:\n"
    "- Pause LinkedIn ads, focus social budget on Instagram.\n"
    "- Total Q4 budget capped at five hundred thousand."
)


def _is_chinese(text):
    """判斷逐字稿主要是中文還是英文：數中日韓字元(CJK)的比例。"""
    if not text:
        return True  # 空字串預設走中文（不影響結果）
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha() and ch.isascii())
    # CJK 字數 >= 英文字母數 → 視為中文
    return cjk >= letters


# 對外公開的語言判斷（meeting_summary.py 用來決定要翻成哪個語言）
def is_chinese(text):
    return _is_chinese(text)


def _looks_valid(summary, want_chinese):
    """
    防呆：判斷摘要是不是「正常的三段式」，用來決定要不要重跑。
    脫序樣態：太短、或缺少三個段落標題（像只吐出「簡體字」）。
    """
    if not summary or not isinstance(summary, str):
        return False
    s = summary.strip()
    if len(s) < 20:                      # 太短 → 一定不是完整三段式
        return False
    if want_chinese:
        heads = ("會議重點", "待辦事項", "決議事項")
    else:
        heads = ("Key Points", "Action Items", "Decisions")
    hit = sum(1 for h in heads if h in s)
    return hit >= 2                      # 至少出現 2 個段落標題才算正常


# ── singleton ──────────────────────────────────────────────────────────
_VDEVICE = None
_LLM = None
_LOCK = threading.Lock()
_LOADED = False


def _ensure_loaded():
    global _VDEVICE, _LLM, _LOADED
    if _LOADED and _LLM is not None:
        return _LLM, None
    with _LOCK:
        if _LOADED and _LLM is not None:
            return _LLM, None
        if not _HAS_HAILO:
            return None, "[summarize] 無法 import hailo_platform.genai，請確認 HailoRT 5.3.x 已安裝"
        hef = Path(HEF_PATH)
        if not hef.exists():
            return None, (f"[summarize] 找不到 .hef：{hef}\n"
                          f"請下載 Qwen2.5-1.5B-Instruct.hef 放到該位置，"
                          f"或設環境變數 HAILO_LLM_HEF 指向其他路徑。")
        try:
            print(f"[summarize] 第一次使用：載入 LLM 模型（約 8 秒）……", flush=True)
            t0 = time.time()
            _VDEVICE = VDevice()
            _LLM = LLM(_VDEVICE, str(hef))
            _LOADED = True
            print(f"[summarize] LLM 載入完成（{time.time()-t0:.1f}s）", flush=True)
            return _LLM, None
        except Exception as e:
            traceback.print_exc()
            try:
                if _LLM is not None: _LLM.release()
            except Exception: pass
            try:
                if _VDEVICE is not None: _VDEVICE.release()
            except Exception: pass
            _LLM = None; _VDEVICE = None; _LOADED = False
            return None, f"[summarize] 載入 LLM 失敗：{type(e).__name__}: {e}"


def close():
    """釋放 VDevice + LLM。給 orchestrator 在跑完摘要後叫，方便其他 Hailo 程式接手。"""
    global _VDEVICE, _LLM, _LOADED
    if _LLM is not None:
        try: _LLM.clear_context()
        except Exception: pass
        try: _LLM.release()
        except Exception: pass
        _LLM = None
    if _VDEVICE is not None:
        try: _VDEVICE.release()
        except Exception: pass
        _VDEVICE = None
    _LOADED = False


atexit.register(close)


# ── 輸出後處理 ─────────────────────────────────────────────────────────
def _to_text(resp):
    """
    把 generate_all 的回傳正規化成純文字。
    有些 HailoRT 版本回傳的是「內容區塊」結構 [{'type':'text','text':'...'}]
    （或其字面字串），不是純 str。這裡一律拆出裡面的 text，避免把 [{'type'...}]
    外殼直接印出來。
    """
    if isinstance(resp, (list, tuple)):
        parts = []
        for it in resp:
            if isinstance(it, dict):
                parts.append(str(it.get("text", it.get("content", ""))))
            else:
                parts.append(_to_text(it))
        return "".join(parts)
    if isinstance(resp, dict):
        return str(resp.get("text", resp.get("content", "")))
    if isinstance(resp, str):
        s = resp.strip()
        # 模型/API 偶爾把回應包成 [{'type':'text','text':'...'}] 的「字面字串」
        if s.startswith("[{") or s.startswith("{'") or s.startswith('{"'):
            try:
                import ast
                return _to_text(ast.literal_eval(s))
            except Exception:
                pass
        return resp
    return str(resp)


# 1.5B 量化模型偶爾會吐字形相近的錯字 / 多包一層 markdown fence。
# 純字串替換修掉這些已知 pattern，不影響其他內容。
_TYPO_FIXES = {
    "朁定": "決定",
    "淉定": "決定",      # 筆電麥實測出現的錯字
    "貞責人": "負責人",
    "汁謝":   "負責人",   # 實測：模型把第二條待辦的「負責人」吐成「汁謝」
    "客護":   "客戶",     # 實測：Whisper/模型把「客戶」聽/吐成「客護」
    "暗決":   "決議",
    "頻期":   "期限",
    "發動內容": "工作內容",
}

# 範例洩漏防護：這些是 1-shot 範例「專屬」的字詞。若摘要某行出現這些字、
# 但逐字稿根本沒提到 → 判定為「抄範例的捏造內容」，整行刪掉。
# 註：英文範例的結構字「Owner」「Task」「Deadline」是格式骨架，不能單獨列為洩漏詞
#     （會把合法的待辦事項一起殺掉），所以只列特定的人名 / 數字 / 片語組合。
_EXAMPLE_LEAK_TERMS = (
    # 中文範例專屬
    "Q4", "五十萬", "五十 萬", "LinkedIn", "Instagram", "KPI", "社群廣告",
    "行銷預算",
    # 英文範例專屬（只放具體內容詞；Owner/Task/Deadline 是格式欄位不能列）
    "John", "social ad platform",
    "end of October", "five hundred thousand",
    "Submit KPI report", "Pause LinkedIn",
)


def _strip_example_leak(summary, transcript):
    """逐行刪掉「含範例專屬字詞、但逐字稿沒提到」的捏造行（防 1-shot 範例洩漏）。"""
    if not summary:
        return summary
    tr = transcript or ""
    kept = []
    for ln in summary.splitlines():
        leaked = any((term in ln) and (term not in tr) for term in _EXAMPLE_LEAK_TERMS)
        if leaked:
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


# 模型偶爾會把 prompt 裡的「鷹架」（分隔線、範例/逐字稿小標）整段回吐到摘要裡。
# 跟 _strip_example_leak 一樣是「整行刪除」式防護，但條件改成「行本身是鷹架」，
# 不依賴逐字稿比對。中英文 header 都涵蓋。
_SCAFFOLD_LINE_RES = (
    # 純分隔線：----- 或 ===== 或 ──── 等三個以上重複符號
    re.compile(r"^\s*[-=─━]{3,}\s*$"),
    # ----- 中/英文 header -----（前後可有空白、分隔符可重複）
    re.compile(
        r"^\s*[-=─━]+\s*"
        r"(範例(?:逐字稿|輸出)?|逐字稿|你的輸出|"
        r"Example(?:\s+(?:transcript|output))?|Transcript|Your\s+output)"
        r"\s*[-=─━]+\s*$",
        re.IGNORECASE,
    ),
    # 單獨一行「範例:」「Example:」
    re.compile(r"^\s*範例\s*[:：]?\s*$"),
    re.compile(r"^\s*Example\s*:?\s*$", re.IGNORECASE),
)


def _strip_scaffold_lines(summary):
    """逐行刪掉模型回吐的 prompt 鷹架（分隔線、範例/逐字稿小標）。"""
    if not summary:
        return summary
    kept = []
    for ln in summary.splitlines():
        if any(p.match(ln) for p in _SCAFFOLD_LINE_RES):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _clean_response(text):
    if not isinstance(text, str):
        text = str(text)

    # (1) 砍尾標記 + 結構化內容尾
    for cut in ("<|im_end|>", "<|endoftext|>", ". [{'type'"):
        idx = text.find(cut)
        if idx >= 0:
            text = text[:idx]

    # (1b) 模型偶爾把換行吐成「字面的 \n / \r\n / \t」兩個字元 → 轉回真正的換行
    text = (text.replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\r", "\n")
                .replace("\\t", " "))

    text = text.strip()

    # (2) 去掉開頭/結尾的 markdown code fence
    import re
    text = re.sub(r"^```\w*\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)

    # (3) 已知錯字替換
    for bad, good in _TYPO_FIXES.items():
        if bad in text:
            text = text.replace(bad, good)

    # (3b) 防呆：避免出現「負負責人」「負負負責人」之類的疊字
    text = re.sub(r"負+責人", "負責人", text)

    return text.strip()


# ── 公開介面 ───────────────────────────────────────────────────────────
def summarize(transcript):
    """
    輸入：transcript = 會議逐字稿字串
    回傳：三段式摘要。自動依逐字稿語言：中文→中文摘要、英文→英文摘要。
    失敗時回友善訊息字串，不丟例外。
    """
    llm, err = _ensure_loaded()
    if err:
        return err

    if _is_chinese(transcript):
        sys_prompt = SYSTEM_PROMPT
        inline_user = (
            "範例：\n"
            "----- 範例逐字稿 -----\n"
            f"{_EXAMPLE_TRANSCRIPT}\n"
            "----- 範例輸出 -----\n"
            f"{_EXAMPLE_OUTPUT}\n"
            "\n"
            "現在請用「完全相同」的格式整理下面這份逐字稿，"
            "用詞必須完整正確、只用繁體中文：\n"
            "----- 逐字稿 -----\n"
            f"{transcript.strip()}\n"
            "----- 你的輸出 -----"
        )
    else:
        sys_prompt = SYSTEM_PROMPT_EN
        inline_user = (
            "Example:\n"
            "----- Example transcript -----\n"
            f"{_EXAMPLE_TRANSCRIPT_EN}\n"
            "----- Example output -----\n"
            f"{_EXAMPLE_OUTPUT_EN}\n"
            "\n"
            "Now summarize the following transcript using the EXACT same "
            "format, in English only:\n"
            "----- Transcript -----\n"
            f"{transcript.strip()}\n"
            "----- Your output -----"
        )

    prompt = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": inline_user}]},
    ]

    try:
        want_zh = _is_chinese(transcript)
        # 防呆：最多試 2 次，脫序就自動重跑；第 2 次加強硬格式指令
        seeds = [_SEED, _SEED + 7]
        cleaned = ""
        for attempt, seed in enumerate(seeds, 1):
            try: llm.clear_context()
            except Exception: pass

            # 第 2 次（重跑）：在使用者訊息尾端追加強硬的三段式要求
            this_user = inline_user
            if attempt >= 2:
                if want_zh:
                    this_user += (
                        "\n\n【重要】你必須輸出三個段落，段落標題一字不差："
                        "「會議重點：」「待辦事項：」「決議事項：」，缺一不可。"
                        "每段至少一條。沒有的資訊寫『未提及』。不要寫成一段話。"
                    )
                else:
                    this_user += (
                        "\n\n[IMPORTANT] You MUST output three sections with these "
                        "exact headers: \"Key Points:\", \"Action Items:\", "
                        "\"Decisions:\". None may be omitted. Write \"Not mentioned\" "
                        "if a section has no info. Do NOT write a single paragraph."
                    )
            this_prompt = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user",   "content": [{"type": "text", "text": this_user}]},
            ]

            t0 = time.time()
            resp = llm.generate_all(
                prompt=this_prompt,
                temperature=_TEMPERATURE,
                seed=seed,
                max_generated_tokens=_MAX_TOKENS,
                frequency_penalty=_FREQ_PENALTY,
                top_p=_TOP_P,
                top_k=_TOP_K,
                do_sample=True,
            )
            dt = time.time() - t0
            cleaned = _clean_response(_to_text(resp))
            cleaned = _strip_example_leak(cleaned, transcript)
            cleaned = _strip_scaffold_lines(cleaned)
            ok = _looks_valid(cleaned, want_zh)
            print(f"[summarize] 第{attempt}次推論完成（{dt:.1f}s，{len(cleaned)} chars，"
                  f"{'正常' if ok else '疑似脫序'}）", flush=True)
            if ok:
                return cleaned
        return cleaned
    except Exception as e:
        traceback.print_exc()
        return f"[summarize] 推論失敗：{type(e).__name__}: {e}"


# ── 翻譯（共用同一顆 LLM，給雙語輸出用） ───────────────────────────────
def translate(text, target):
    """
    把 text 翻成 target 語言（'en' 英文 / 'zh' 繁體中文），共用 summarize 已載入的 LLM。
    翻譯是單純任務（不像摘要要生格式），所以單次推論、不重跑。
    保留原文的換行 / 條列 / 段落標題結構。失敗回友善訊息字串，不丟例外。
    """
    if not text or not str(text).strip():
        return ""
    # 若上游傳進來的是錯誤訊息（[transcribe]/[summarize] 開頭），不翻、原樣回傳
    if str(text).lstrip().startswith("["):
        return text

    llm, err = _ensure_loaded()
    if err:
        return err

    if str(target).lower().startswith("en"):
        sys_prompt = (
            "You are a professional translator. Translate the user's text into "
            "natural, fluent English. Keep the original structure exactly: line "
            "breaks, bullet points, and section headers. Translate Chinese section "
            "headers as: 會議重點->Key Points, 待辦事項->Action Items, 決議事項->Decisions, "
            "負責人->Owner, 工作內容->Task, 期限->Deadline, 未提及->Not mentioned. "
            "Output ONLY the translation. No notes, no extra text, no original text."
        )
    else:
        sys_prompt = (
            "你是專業翻譯。請把使用者的文字翻成自然、流暢的『繁體中文』（台灣用語）。"
            "完整保留原本的結構：換行、條列、段落標題。"
            "英文段落標題請譯為：Key Points→會議重點、Action Items→待辦事項、"
            "Decisions→決議事項、Owner→負責人、Task→工作內容、Deadline→期限、"
            "Not mentioned→未提及。只輸出譯文，不要加任何說明、不要附原文、禁止簡體字。"
        )

    prompt = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": str(text).strip()}]},
    ]

    try:
        try:
            llm.clear_context()
        except Exception:
            pass
        t0 = time.time()
        resp = llm.generate_all(
            prompt=prompt,
            temperature=_TEMPERATURE,
            seed=_SEED,
            max_generated_tokens=_MAX_TOKENS,
            frequency_penalty=_FREQ_PENALTY,
            top_p=_TOP_P,
            top_k=_TOP_K,
            do_sample=True,
        )
        dt = time.time() - t0
        cleaned = _clean_response(_to_text(resp))
        print(f"[translate→{target}] 完成（{dt:.1f}s，{len(cleaned)} chars）", flush=True)
        return cleaned
    except Exception as e:
        traceback.print_exc()
        return f"[translate] 失敗：{type(e).__name__}: {e}"
