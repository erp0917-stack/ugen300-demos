# -*- coding: utf-8 -*-
"""
llm_smoke_test.py
=================
Qwen2.5-1.5B-Instruct 透過 hailo_platform.genai.LLM 跑摘要：
  - 餵一段假的會議逐字稿（中英夾雜，模擬 WQ1 真實場景）
  - 要求輸出「會議重點 + 待辦事項」三段式繁體中文摘要
  - 每步獨立 try/except、印推論時間
"""
import os
import sys
import time
import traceback
from pathlib import Path

HEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen2.5-1.5B-Instruct.hef")

# WQ1 transcribe.py 用過的示範逐字稿（已含中英夾雜、人名、預算）
FAKE_TRANSCRIPT = (
    "好我們今天開會主要討論下週的產品發表會。"
    "第一個 marketing 的部分，請 Amy 在週三前把新聞稿初稿給我。"
    "RD 這邊，阿志你負責把 demo 機在週四前準備好，要兩台備援。"
    "然後通路的部分我們決定先鎖定北部三家經銷商，這個是確定的。"
    "預算的話，這次發表會控制在二十萬以內，超過要再簽核。"
    "OK 那就這樣，散會。"
)

SYSTEM = (
    "你是專業的會議記錄助理。\n"
    "\n"
    "嚴格規則：\n"
    "(1) 只用標準繁體中文，禁止使用簡體字（例如禁止『区』『决』『发』『钟』『话』『听』『说』等）。\n"
    "(2) 禁止使用非標準詞彙或自創縮寫；用詞必須完整正確。"
    "範例：要寫『負責人』不是『責人』、『決議事項』不是『暗決』、"
    "『工作內容』不是『發動內容』、『期限』不是『頻期』也不是『長度』。\n"
    "(3) 一律照範例的固定格式輸出，章節順序為「會議重點 → 待辦事項 → 決議事項」，不可更動章節名與項目名。\n"
    "(4) 沒提到的資訊在該欄位寫『未提及』，禁止編造資訊或人名。\n"
    "(5) 不要加開場白、結語、客套話、或自我評論。直接輸出三段式摘要。"
)

# 1-shot 範例：round 1 已驗證版本（Q4 行銷預算，刻意跟 demo 主題不同 domain，避免事實洩漏）
EXAMPLE_TRANSCRIPT = (
    "好今天主要講Q4行銷預算。第一，請大華在十月底前把 KPI 報告交給我。"
    "第二，社群這邊我們決定先暫停 LinkedIn 廣告，集中火力在 Instagram。"
    "第三，總預算這季控制在五十萬以內。OK 散會。"
)

EXAMPLE_OUTPUT = (
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

EXAMPLE_USER_TEXT = (
    "以下是會議逐字稿，請整理成標準格式：\n\n"
    f"{EXAMPLE_TRANSCRIPT}"
)
REAL_USER_TEXT = (
    "以下是會議逐字稿，請整理成標準格式：\n\n"
    f"{FAKE_TRANSCRIPT}"
)


def step(t):
    print(f"\n========== {t} ==========", flush=True)


def _robust_clean(text):
    """
    Hailo Qwen2.5 偶爾會把回答包成 [{'type': 'text', 'text': '...'}] 結構
    （參考官方 simple_vlm_chat 的 split('. [{\\'type\\'')）。
    這個函式把這層結構剝掉、再去掉 <|im_end|> 等尾標記，回傳純文字。
    """
    import re, ast
    if not isinstance(text, str):
        text = str(text)

    # 1. 去 markdown code fence（``` 或 ```python...```）
    text = re.sub(r"^```\w*\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)

    # 2. 若包成 list-of-dict ([{'type':'text','text':'...'}]) → 萃取內層 text
    if text.startswith("[") and "type" in text and "text" in text:
        # 找 ]} 收尾，截到那為止再 literal_eval
        for end_idx in range(len(text), 0, -1):
            try:
                parsed = ast.literal_eval(text[:end_idx])
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    inner = parsed[0].get("text") or parsed[0].get("content") or ""
                    if inner:
                        text = inner
                        break
            except Exception:
                continue

    # 3. 砍尾標記
    for cut in ("<|im_end|>", "<|endoftext|>", ". [{'type'"):
        idx = text.find(cut)
        if idx >= 0:
            text = text[:idx]

    return text.strip()


# ---- imports ----
step("import LLM")
try:
    from hailo_platform import VDevice
    from hailo_platform.genai import LLM
    print("  ✓ imports OK")
except Exception as e:
    print(f"  ✗ FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)


# ---- hef present ----
step(".hef present & size sanity")
hef = Path(HEF_PATH)
if not hef.exists():
    print(f"  ✗ 找不到 {hef}")
    sys.exit(1)
print(f"  ✓ {hef.name} = {hef.stat().st_size/1024/1024/1024:.2f} GB")


# ---- open VDevice + load LLM ----
step("open VDevice + load LLM(Qwen2.5-1.5B-Instruct)")
vdevice = None
llm = None
try:
    t0 = time.time()
    vdevice = VDevice()
    print(f"  ✓ VDevice opened in {time.time()-t0:.2f}s")
    t0 = time.time()
    llm = LLM(vdevice, str(hef))
    print(f"  ✓ LLM loaded in {time.time()-t0:.2f}s")
except Exception as e:
    print(f"  ✗ FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    if vdevice: vdevice.release()
    sys.exit(1)


# ---- summarise (single-user-inline: smoke test round 1 已驗證格式 + 新範例) ----
step("summarise the fake transcript  [single user message with inline example]")
inline_user = (
    "範例：\n"
    "----- 範例逐字稿 -----\n"
    f"{EXAMPLE_TRANSCRIPT}\n"
    "----- 範例輸出 -----\n"
    f"{EXAMPLE_OUTPUT}\n"
    "\n"
    "現在請用「完全相同」的格式整理下面這份逐字稿，"
    "用詞必須完整正確、只用繁體中文：\n"
    "----- 逐字稿 -----\n"
    f"{FAKE_TRANSCRIPT}\n"
    "----- 你的輸出 -----"
)
prompt = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
    {"role": "user",   "content": [{"type": "text", "text": inline_user}]},
]

answer = None
try:
    # 跟 VLM 同樣的習慣：跑前 clear_context 避免上次狀態殘留
    try:
        llm.clear_context()
    except Exception as e:
        print(f"  clear_context 略過：{e}")

    t0 = time.time()
    resp = llm.generate_all(
        prompt=prompt,
        temperature=0.1,    # 進一步壓低，更貼近 greedy
        seed=42,
        max_generated_tokens=400,
        frequency_penalty=1.15,
        top_p=0.85,         # 砍長尾
        top_k=30,           # 每步只看前 30 個 token
        do_sample=True,
    )
    dt = time.time() - t0
    raw = resp if isinstance(resp, str) else str(resp)
    cleaned = _robust_clean(raw)

    print(f"  ✓ generate_all returned in {dt:.2f}s "
          f"(raw={len(raw)} chars, clean={len(cleaned)} chars)")
    print("  ---- 摘要結果 ----")
    print(cleaned)
    print("  ---- end ----")
    answer = cleaned
except Exception as e:
    print(f"  ✗ FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()


# ---- cleanup ----
step("cleanup")
try:
    if llm:
        try: llm.clear_context()
        except Exception: pass
        llm.release()
        print("  ✓ LLM released")
    if vdevice:
        vdevice.release()
        print("  ✓ VDevice released")
except Exception as e:
    print(f"  ! cleanup warning: {e}")


# ---- summary ----
step("驗證摘要")
if not answer:
    print("❌ LLM 沒回東西")
    sys.exit(1)
print(f"✅ LLM 給出 {len(answer)} 字的摘要")

# example leak 檢查：範例獨有的人/事/物不應該出現在實際答案
LEAK_TOKENS = ["大華", "KPI", "LinkedIn", "Instagram", "五十萬", "Q4"]
leaks = [tok for tok in LEAK_TOKENS if tok in answer]
if leaks:
    print(f"\n⚠️ 偵測到範例洩漏（範例事實出現在實際答案）：{leaks}")
    print("   多輪 + 同主題範例可能造成 leak，建議退回單 user 內嵌版（smoke test round 1）")
else:
    print("\n✅ 沒偵測到範例洩漏，多輪格式安全")
