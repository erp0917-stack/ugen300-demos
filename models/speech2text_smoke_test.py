# -*- coding: utf-8 -*-
"""
speech2text_smoke_test.py
=========================
Whisper-Base 透過 hailo_platform.genai.Speech2Text 跑兩次轉錄：
  Round 1：英文 Hailo 官方 sample (sample_audio_en.wav, 1.3s)
  Round 2：繁體中文 SAPI Hanhan 合成 (sample_audio_zh.wav, ~8.9s)

每輪獨立 try/except；整個 process 共用一個 VDevice + Speech2Text 實例。
"""
import os
import sys
import time
import traceback
import wave
from pathlib import Path

import numpy as np

HEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Whisper-Base.hef")
EN_WAV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_audio_en.wav")
ZH_WAV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_audio_zh.wav")


def step(title):
    print(f"\n========== {title} ==========", flush=True)


def load_wav_as_float32(path):
    """Whisper 要 16kHz mono float32 normalised (-1.0~1.0)。"""
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    assert ch == 1, f"{path}: 預期 mono，實際 {ch} 聲道"
    assert sw == 2, f"{path}: 預期 16-bit (sw=2)，實際 sw={sw}"
    assert sr == 16000, f"{path}: 預期 16kHz，實際 {sr}Hz"
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.astype("<f4")


# ---- imports ----
step("import Speech2Text / Speech2TextTask")
try:
    from hailo_platform import VDevice
    from hailo_platform.genai import Speech2Text, Speech2TextTask
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
print(f"  ✓ {hef.name} = {hef.stat().st_size/1024/1024:.1f} MB")


# ---- open VDevice + load Whisper ----
step("open VDevice + load Speech2Text(Whisper-Base)")
vdevice = None
stt = None
try:
    t0 = time.time()
    vdevice = VDevice()
    print(f"  ✓ VDevice opened in {time.time()-t0:.2f}s")
    t0 = time.time()
    stt = Speech2Text(vdevice, str(hef))
    print(f"  ✓ Speech2Text loaded in {time.time()-t0:.2f}s")
except Exception as e:
    print(f"  ✗ FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    if vdevice: vdevice.release()
    sys.exit(1)


def run_round(label, wav_path, language):
    step(f"transcribe: {label}  ({wav_path})  lang={language}")
    try:
        audio = load_wav_as_float32(wav_path)
        print(f"  audio: {audio.shape} dtype={audio.dtype} "
              f"duration={len(audio)/16000:.2f}s")
        t0 = time.time()
        segments = stt.generate_all_segments(
            audio_data=audio,
            task=Speech2TextTask.TRANSCRIBE,
            language=language,
            timeout_ms=30000,
        )
        dt = time.time() - t0
        print(f"  ✓ generate_all_segments returned in {dt:.2f}s, "
              f"{len(segments) if segments else 0} segment(s)")
        if not segments:
            print("  ⚠️ no segments")
            return ""
        for i, seg in enumerate(segments):
            print(f"    [seg {i}] {seg.text!r}")
        full = "".join(seg.text for seg in segments).strip()
        print(f"  ---- combined ----")
        print(f"  {full}")
        return full
    except Exception as e:
        print(f"  ✗ FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ---- Round 1: English sample ----
en_text = run_round("英文 sample", EN_WAV, "en")

# ---- Round 2: Traditional Chinese SAPI sample ----
zh_text = run_round("繁體中文 SAPI 合成", ZH_WAV, "zh")


# ---- cleanup ----
step("cleanup")
try:
    if stt:
        stt.release()
        print("  ✓ Speech2Text released")
    if vdevice:
        vdevice.release()
        print("  ✓ VDevice released")
except Exception as e:
    print(f"  ! cleanup warning: {e}")


# ---- summary ----
step("驗證摘要")
print(f"EN 轉錄：{en_text!r}")
print(f"ZH 轉錄：{zh_text!r}")
bad = []
if not en_text: bad.append("EN")
if not zh_text: bad.append("ZH")
if bad:
    print(f"\n❌ 失敗：{bad}")
    sys.exit(1)
print("\n✅ 兩輪都拿到非空轉錄文字")
