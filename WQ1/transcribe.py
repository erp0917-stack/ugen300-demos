# -*- coding: utf-8 -*-
"""
transcribe.py
=============
把 wav 音檔或現場錄音餵給 UGen300 / Hailo-10H 上的 Whisper-Base，
回傳逐字稿（中文/英文/中英夾雜）。

走 HailoRT GenAI 直接 API（hailo_platform.genai.Speech2Text），
模型在第一次呼叫時載入一次（約 1.5 秒），同一 process 內重複使用。

公開介面：
    transcribe(wav_path, language='zh') -> str
    close()  # 釋放 VDevice + Speech2Text，給 summarize 等其他 Hailo 程式接手

Whisper-Base 期待的音訊：16kHz / mono / float32 normalised (-1.0~1.0)。
這裡會做 sanity check，輸入規格不對直接拒絕（避免轉錯）。
"""
import atexit
import os
import sys
import threading
import time
import traceback
import wave
from pathlib import Path

import numpy as np

try:
    from hailo_platform import VDevice
    from hailo_platform.genai import Speech2Text, Speech2TextTask
    _HAS_HAILO = True
except Exception:
    _HAS_HAILO = False


# ── 設定 ────────────────────────────────────────────────────────────────
_DEFAULT_HEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Whisper-Base.hef")
HEF_PATH = os.environ.get("HAILO_WHISPER_HEF", _DEFAULT_HEF)

REQUIRED_SR = 16000
REQUIRED_CH = 1
REQUIRED_SW = 2  # 16-bit


# ── singleton ──────────────────────────────────────────────────────────
_VDEVICE = None
_STT = None
_LOCK = threading.Lock()
_LOADED = False


def _ensure_loaded():
    global _VDEVICE, _STT, _LOADED
    if _LOADED and _STT is not None:
        return _STT, None
    with _LOCK:
        if _LOADED and _STT is not None:
            return _STT, None
        if not _HAS_HAILO:
            return None, "[transcribe] 無法 import hailo_platform.genai，請確認 HailoRT 5.3.x 已安裝"
        hef = Path(HEF_PATH)
        if not hef.exists():
            return None, (f"[transcribe] 找不到 .hef：{hef}\n"
                          f"請下載 Whisper-Base.hef 放到該位置，"
                          f"或設環境變數 HAILO_WHISPER_HEF 指向其他路徑。")
        try:
            print(f"[transcribe] 第一次使用：載入 Whisper 模型（約 2 秒）……", flush=True)
            t0 = time.time()
            _VDEVICE = VDevice()
            _STT = Speech2Text(_VDEVICE, str(hef))
            _LOADED = True
            print(f"[transcribe] Whisper 載入完成（{time.time()-t0:.1f}s）", flush=True)
            return _STT, None
        except Exception as e:
            traceback.print_exc()
            try:
                if _STT is not None: _STT.release()
            except Exception: pass
            try:
                if _VDEVICE is not None: _VDEVICE.release()
            except Exception: pass
            _STT = None; _VDEVICE = None; _LOADED = False
            return None, f"[transcribe] 載入 Whisper 失敗：{type(e).__name__}: {e}"


def close():
    """釋放 VDevice + Speech2Text，給 summarize 接手用。"""
    global _VDEVICE, _STT, _LOADED
    if _STT is not None:
        try: _STT.release()
        except Exception: pass
        _STT = None
    if _VDEVICE is not None:
        try: _VDEVICE.release()
        except Exception: pass
        _VDEVICE = None
    _LOADED = False


atexit.register(close)


# ── 音檔讀取 ───────────────────────────────────────────────────────────
def _load_wav_as_float32(path):
    """讀 wav 檔，sanity check 規格，轉成 Whisper 要的 float32。"""
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n  = w.getnframes()
        raw = w.readframes(n)
    duration = n / float(sr) if sr else 0.0
    print(f"[transcribe] 音檔 {path}：{duration:.1f}s, {sr}Hz, {ch}ch, "
          f"{sw*8}-bit", flush=True)

    if ch != REQUIRED_CH:
        raise ValueError(f"音檔聲道={ch}，Whisper 需要 mono (1 channel)")
    if sr != REQUIRED_SR:
        raise ValueError(f"音檔取樣率={sr}Hz，Whisper 需要 {REQUIRED_SR}Hz")
    if sw != REQUIRED_SW:
        raise ValueError(f"音檔 sample width={sw}，Whisper 需要 16-bit (sw=2)")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.astype("<f4")


# ── 公開介面 ───────────────────────────────────────────────────────────
def transcribe(wav_path, language="zh", translate_to_english=False):
    """
    輸入：
      wav_path             WAV 檔路徑（必須 16kHz / mono / 16-bit）
      language             'zh' 中文、'en' 英文。None 自動偵測（中英夾雜可試 None）
      translate_to_english 為 True 時用 Whisper 的「翻譯」任務：不管講什麼語言，
                           直接輸出英文（這比叫 LLM 翻譯可靠太多）。若該 Whisper
                           不支援 TRANSLATE 任務，會自動退回一般轉錄並印出提示。
    回傳：
      逐字稿字串。失敗回友善訊息字串，不丟例外。
    """
    stt, err = _ensure_loaded()
    if err:
        return err

    try:
        audio = _load_wav_as_float32(wav_path)
    except Exception as e:
        return f"[transcribe] 讀音檔失敗：{e}"

    # Whisper-Base 一次最多吃 30 秒；超過要切段。Hailo 的 generate_all_segments
    # 自己會處理長音檔，不過超長的話 timeout 要拉長。
    duration = len(audio) / REQUIRED_SR
    timeout_ms = max(15000, int(duration * 3000))  # 3x real-time 寬限

    # 選任務：要英文輸出 → 用 TRANSLATE（不支援就退回 TRANSCRIBE）
    task = Speech2TextTask.TRANSCRIBE
    if translate_to_english:
        task_translate = getattr(Speech2TextTask, "TRANSLATE", None)
        if task_translate is not None:
            task = task_translate
            print("[transcribe] 使用 Whisper TRANSLATE 任務（輸出英文）。", flush=True)
        else:
            print("[transcribe] 此 Whisper 不支援 TRANSLATE 任務，改用一般轉錄"
                  "（英文輸出需直接講英文）。", flush=True)

    try:
        t0 = time.time()
        kwargs = dict(
            audio_data=audio,
            task=task,
            timeout_ms=timeout_ms,
        )
        if language:
            kwargs["language"] = language
        segments = stt.generate_all_segments(**kwargs)
        dt = time.time() - t0
        n_seg = len(segments) if segments else 0
        print(f"[transcribe] 推論完成（{dt:.1f}s，{n_seg} segment(s)）", flush=True)
        if not segments:
            return ""
        full = "".join(seg.text for seg in segments).strip()
        return full
    except Exception as e:
        traceback.print_exc()
        return f"[transcribe] 推論失敗：{type(e).__name__}: {e}"


# ── CLI（給 debug 用，不需要透過 meeting_summary.py 也能單獨跑）─────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Whisper-Base 轉錄 wav")
    ap.add_argument("wav", help="要轉錄的 wav 路徑（必須 16kHz / mono / 16-bit）")
    ap.add_argument("--language", default="zh", help="zh / en / 留空 = 自動")
    args = ap.parse_args()
    lang = None if args.language.lower() in ("auto", "none", "") else args.language
    text = transcribe(args.wav, language=lang)
    print("\n----- 逐字稿 -----")
    print(text)
    print("------------------")
