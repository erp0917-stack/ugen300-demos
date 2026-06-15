# -*- coding: utf-8 -*-
"""
llm_client.py — Qwen2.5-1.5B-Instruct 純文字 LLM 客戶端
照搬 WQ1/summarize.py 的 VDevice 生命週期模式。

公開介面：
    generate(user_msg, system_prompt=None, max_tokens=600, temperature=0.15) -> str
    close()   # 釋放 VDevice + LLM，必須在 VL close 之後才呼叫
"""

import atexit
import os
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

# ── 設定 ─────────────────────────────────────────────────────────────────
_DEFAULT_HEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen2.5-1.5B-Instruct.hef")
HEF_PATH = os.environ.get("HAILO_LLM_HEF", _DEFAULT_HEF)

_TEMPERATURE   = 0.15
_TOP_P         = 0.85
_TOP_K         = 30
_FREQ_PENALTY  = 1.15
_SEED          = 42
_MAX_TOKENS    = 600

# ── singleton ─────────────────────────────────────────────────────────────
_VDEVICE = None
_LLM     = None
_LOCK    = threading.Lock()
_LOADED  = False


def _ensure_loaded():
    global _VDEVICE, _LLM, _LOADED
    if _LOADED and _LLM is not None:
        return _LLM, None
    with _LOCK:
        if _LOADED and _LLM is not None:
            return _LLM, None
        if not _HAS_HAILO:
            return None, "[llm_client] 無法 import hailo_platform.genai，請確認 HailoRT 5.3.x 已安裝"
        hef = Path(HEF_PATH)
        if not hef.exists():
            return None, (f"[llm_client] 找不到 .hef：{hef}\n"
                          f"請下載 Qwen2.5-1.5B-Instruct.hef 放到該位置，"
                          f"或設環境變數 HAILO_LLM_HEF 指向其他路徑。")
        try:
            print("[llm_client] 第一次使用：載入 LLM 模型（約 8 秒）……", flush=True)
            t0 = time.time()
            _VDEVICE = VDevice()
            _LLM = LLM(_VDEVICE, str(hef))
            _LOADED = True
            print(f"[llm_client] LLM 載入完成（{time.time()-t0:.1f}s）", flush=True)
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
            return None, f"[llm_client] 載入 LLM 失敗：{type(e).__name__}: {e}"


def close():
    """釋放 VDevice + LLM。VL close 之後、LLM 推論完畢後呼叫。"""
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


def _to_text(resp):
    """把 generate_all 回傳正規化成純字串（照搬 WQ1/summarize.py 的 _to_text）。"""
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
        if s.startswith("[{") or s.startswith("{'") or s.startswith('{"'):
            try:
                import ast
                return _to_text(ast.literal_eval(s))
            except Exception:
                pass
        return resp
    return str(resp)


def _clean(text):
    """去掉模型常見的結尾標記。"""
    for cut in ("<|im_end|>", "<|endoftext|>", ". [{'type'"):
        idx = text.find(cut)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


# ── 公開介面 ──────────────────────────────────────────────────────────────
def generate(user_msg, system_prompt=None,
             max_tokens=_MAX_TOKENS, temperature=_TEMPERATURE):
    """
    輸入純文字 user_msg，回傳 LLM 文字回應（str）。
    失敗回友善訊息字串，不丟例外。
    """
    llm, err = _ensure_loaded()
    if err:
        return err

    sys_txt = system_prompt or "你是專業助理，只輸出被要求的內容，不加任何額外說明。"

    prompt = [
        {"role": "system",  "content": [{"type": "text", "text": sys_txt}]},
        {"role": "user",    "content": [{"type": "text", "text": user_msg}]},
    ]

    try:
        llm.clear_context()
    except Exception as e:
        print(f"[llm_client] clear_context 略過：{e}", flush=True)

    try:
        t0 = time.time()
        resp = llm.generate_all(
            prompt=prompt,
            temperature=temperature,
            seed=_SEED,
            max_generated_tokens=max_tokens,
            frequency_penalty=_FREQ_PENALTY,
            top_p=_TOP_P,
            top_k=_TOP_K,
            do_sample=True,
        )
        dt = time.time() - t0
        raw  = _to_text(resp)
        clean = _clean(raw)
        print(f"[llm_client] 推論完成（{dt:.1f}s，{len(clean)} chars）", flush=True)
        return clean
    except Exception as e:
        traceback.print_exc()
        return f"[llm_client] 推論失敗：{type(e).__name__}: {e}"
