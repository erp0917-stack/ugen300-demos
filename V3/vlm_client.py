# -*- coding: utf-8 -*-
"""
vlm_client.py
=============
「會看圖說話的大腦」：一張畫面 + 一個問題 → UGen300 / Hailo-10H 上的
Qwen2-VL-2B 給出文字回答。

直接走 HailoRT GenAI Python API（hailo_platform.genai.VLM），不需要 Hailo-Ollama
伺服器。模型在第一次呼叫時載入一次（約 10 秒），之後重複呼叫共用同一個 VDevice
與 VLM 實例。

對 V1/V2/V3 三支主程式而言，公開介面仍是同一支函式：
    ask_about_image(frame_bgr, question) -> str
所以主程式完全不用改。

預設 .hef 路徑：
    ../models/Qwen2-VL-2B-Instruct.hef（相對於本檔所在的 app 資料夾）
可用環境變數 HAILO_VLM_HEF 覆寫。
"""

import atexit
import os
import time
import threading
import traceback
from pathlib import Path

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from hailo_platform import VDevice
    from hailo_platform.genai import VLM
    _HAS_HAILO = True
except Exception:
    _HAS_HAILO = False


# ── 設定 ────────────────────────────────────────────────────────────────
_DEFAULT_HEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen2-VL-2B-Instruct.hef")
HEF_PATH = os.environ.get("HAILO_VLM_HEF", _DEFAULT_HEF)

# Qwen2-VL-2B Hailo .hef 的輸入是 336x336 RGB uint8（NHWC）
INPUT_SIZE = 336

DEFAULT_SYSTEM_PROMPT = (
    "你是看圖回答的中文助理。"
    "嚴格規則："
    "(1) 必須用『繁體中文』作答，禁止使用簡體中文。"
    "(2) 不可夾雜任何英文字母、亂碼或無意義字元；除非是專有名詞，否則一律用中文。"
    "(3) 看不清楚或不確定時，直接用中文說「畫面看不清楚」「無法判斷」，不要拼湊不存在的字詞。"
    "(4) 不要說「希望這對你有幫助」「如果還有問題請告訴我」之類的客套話。"
    "(5) 回答只給結論，控制在 80 字以內，除非使用者明確要求詳細說明。"
)

# 生成參數（看圖回答要穩定、不要亂編，但要防止 Qwen2-VL 偶發跳針 / 吐亂碼 token）
DEFAULT_TEMPERATURE = 0.2
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 120
DEFAULT_FREQ_PENALTY = 1.15  # >1 抑制重複 token，0~2 之間，1.1~1.3 常用
DEFAULT_TOP_P = 0.9          # nucleus sampling，砍掉機率分布長尾（亂碼/英文碎片多在那）
DEFAULT_TOP_K = 40           # 每步只從機率最高的 K 個 token 取樣


# ── singleton VLM ──────────────────────────────────────────────────────
_VDEVICE = None
_VLM = None
_LOAD_LOCK = threading.Lock()
_LOADED = False


def _ensure_loaded():
    """第一次呼叫時載入模型；之後 no-op。回傳 (vlm, error_msg)，error 為 None 代表成功。"""
    global _VDEVICE, _VLM, _LOADED

    if _LOADED and _VLM is not None:
        return _VLM, None

    with _LOAD_LOCK:
        if _LOADED and _VLM is not None:
            return _VLM, None

        if not _HAS_HAILO:
            return None, ("[vlm_client] 無法 import hailo_platform.genai。"
                          "請確認 HailoRT 5.3.x Python 套件已安裝。")
        if not _HAS_CV2:
            return None, "[vlm_client] 無法 import cv2，請 pip install opencv-python。"

        hef = Path(HEF_PATH)
        if not hef.exists():
            return None, (f"[vlm_client] 找不到 .hef：{hef}\n"
                          f"請下載 Qwen2-VL-2B-Instruct.hef 放到該位置，"
                          f"或設環境變數 HAILO_VLM_HEF 指向其他路徑。")

        try:
            print(f"[vlm_client] 第一次使用：載入 VLM 模型（約 10 秒）……", flush=True)
            t0 = time.time()
            _VDEVICE = VDevice()
            _VLM = VLM(_VDEVICE, str(hef))
            _LOADED = True
            print(f"[vlm_client] VLM 載入完成（{time.time()-t0:.1f}s）。"
                  f"輸入 shape={_VLM.input_frame_shape()}", flush=True)
            return _VLM, None
        except Exception as e:
            traceback.print_exc()
            # 清乾淨，下次還能重試
            try:
                if _VLM is not None: _VLM.release()
            except Exception:
                pass
            try:
                if _VDEVICE is not None: _VDEVICE.release()
            except Exception:
                pass
            _VLM = None
            _VDEVICE = None
            _LOADED = False
            return None, f"[vlm_client] 載入 VLM 失敗：{type(e).__name__}: {e}"


def _release():
    """程式結束時釋放 VDevice / VLM。"""
    global _VDEVICE, _VLM, _LOADED
    if _VLM is not None:
        try: _VLM.clear_context()
        except Exception: pass
        try: _VLM.release()
        except Exception: pass
        _VLM = None
    if _VDEVICE is not None:
        try: _VDEVICE.release()
        except Exception: pass
        _VDEVICE = None
    _LOADED = False


atexit.register(_release)


# ── 影像前處理 ──────────────────────────────────────────────────────────
def _preprocess_bgr(frame_bgr):
    """OpenCV BGR → Qwen2-VL Hailo 期待的 (336, 336, 3) uint8 RGB。"""
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _clean_response(text):
    """去掉模型常見的尾標記（<|im_end|>、tool 之類）。"""
    if not isinstance(text, str):
        text = str(text)
    # 官方 simple_vlm_chat 用過的兩個 cut 點
    for cut in ("<|im_end|>", "<|endoftext|>", ". [{'type'"):
        idx = text.find(cut)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


# ── 公開介面（V1/V2/V3 主程式呼叫的就是這支） ──────────────────────────
def ask_about_image(frame_bgr, question,
                    system_prompt=None,
                    temperature=DEFAULT_TEMPERATURE,
                    seed=DEFAULT_SEED,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    frequency_penalty=DEFAULT_FREQ_PENALTY,
                    top_p=DEFAULT_TOP_P,
                    top_k=DEFAULT_TOP_K,
                    timeout=None):
    """
    輸入：
      frame_bgr         = OpenCV 影像（webcam 拍到的當前畫面，BGR）
      question          = 要問的問題（中文或英文皆可）
      system_prompt     = 想覆寫系統提示時可以給；不給就用內建中文預設
      temperature       = 取樣溫度（看圖回答建議 0.1~0.3）
      seed              = 取樣 seed（同 seed + temperature 結果穩定）
      max_tokens        = 最長回答 token 數
      frequency_penalty = 抑制 token 重複（防 Qwen2-VL 跳針），1.1~1.3 常用
      timeout           = 為相容舊介面保留，目前忽略
    回傳：
      VLM 文字回答（str）。失敗時回友善訊息（不丟例外）。

    註：HailoRT VLM 預設會「保留對話 context」，意思是第二次呼叫就不能再帶 system 訊息。
        我們把每次 ask_about_image 視為「一個獨立的問題」，所以在每次推論前
        clear_context()，等於每次都是全新對話。
    """
    vlm, err = _ensure_loaded()
    if err:
        return err

    try:
        img = _preprocess_bgr(frame_bgr)
    except Exception as e:
        return f"[vlm_client] 影像前處理失敗：{e}"

    sys_txt = system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt = [
        {"role": "system",
         "content": [{"type": "text", "text": sys_txt}]},
        {"role": "user",
         "content": [
             {"type": "image"},
             {"type": "text", "text": question},
         ]},
    ]

    # 重置對話狀態 → 每次呼叫都當成新對話（V1/V2/V3 demo 每次按鍵都是獨立 Q）
    try:
        vlm.clear_context()
    except Exception as e:
        print(f"[vlm_client] clear_context 略過：{e}", flush=True)

    try:
        t0 = time.time()
        resp = vlm.generate_all(
            prompt=prompt,
            frames=[img],
            temperature=temperature,
            seed=seed,
            max_generated_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,  # 開取樣，否則 greedy 模式遇抽象圖會直接吐 <|im_end|>
        )
        dt = time.time() - t0
        raw = resp if isinstance(resp, str) else str(resp)
        clean = _clean_response(raw)
        print(f"[vlm_client] 推論完成（{dt:.1f}s，raw={len(raw)} chars, clean={len(clean)} chars）",
              flush=True)
        if len(clean) == 0:
            print(f"[vlm_client] WARN clean 後為空，raw={raw!r}", flush=True)
        return clean
    except Exception as e:
        traceback.print_exc()
        return (f"[vlm_client] 推論失敗：{type(e).__name__}: {e}\n"
                f"請確認 .hef 與 HailoRT 5.3.x 相容，以及 UGen300 沒有被其他程式佔用。")
