# -*- coding: utf-8 -*-
"""
record_audio.py
===============
負責「拿到聲音」：用麥克風錄音，或讀預錄 wav 檔。
本檔不依賴 UGen300，是純錄音工具。

預設用系統預設輸入裝置（通常是筆電內建麥克風或 webcam 內建麥克風）。
透過 meeting_summary.py 的 --mic 參數可指定其他裝置：
   --mic default      → 系統預設
   --mic BRIO         → 名字含 BRIO 的裝置（外接 webcam）
   --mic 2            → 直接用編號 2 的裝置
   --mic-list         → 列出所有可用麥克風

輸出規格固定為 Whisper 要的 16kHz / mono / 16-bit PCM WAV。
（Whisper 推論時會由 transcribe.py 把 int16 normalize 成 float32。）
"""

import wave

try:
    import sounddevice as sd
    import numpy as np
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False

SAMPLE_RATE = 16000  # Whisper 標準取樣率
CHANNELS = 1


def find_input_device(name_hint=None):
    """
    依名稱找麥克風裝置編號。
      name_hint=None → 回傳 None（代表用系統預設）
      name_hint="BRIO" → 找名字含 BRIO 的輸入裝置，回傳它的編號
      name_hint="2" → 直接當編號用
    找不到就回 None（退回系統預設），並印出提示。
    """
    if not _HAS_AUDIO or name_hint is None:
        return None
    # 純數字 → 直接當編號
    if str(name_hint).isdigit():
        return int(name_hint)
    # 否則當名稱關鍵字找
    hint = str(name_hint).lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and hint in dev["name"].lower():
            print(f"[record] 找到麥克風：#{idx} {dev['name']}")
            return idx
    print(f"[record] 找不到名稱含「{name_hint}」的麥克風，改用系統預設。")
    return None


def record_until_enter(out_path="meeting.wav", device=None):
    """現場錄音：按 Enter 開始，講完再按 Enter 停止。device 可指定麥克風。"""
    if not _HAS_AUDIO:
        raise RuntimeError("尚未安裝 sounddevice，請先 pip install sounddevice numpy")

    input("\n>>> 準備好就按 Enter 開始錄音…")
    print(">>> 錄音中…講完再按一次 Enter 停止。")

    frames = []

    def callback(indata, n, t, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="int16", device=device, callback=callback)
    with stream:
        input()
    print(">>> 錄音結束。")

    audio = np.concatenate(frames, axis=0)
    _save_wav(out_path, audio)
    print(f">>> 已存檔：{out_path}")
    return out_path


def record_fixed_seconds(seconds=30, out_path="meeting.wav", device=None):
    """錄固定秒數。device 可指定麥克風。"""
    if not _HAS_AUDIO:
        raise RuntimeError("尚未安裝 sounddevice，請先 pip install sounddevice numpy")
    print(f">>> 開始錄音 {seconds} 秒…")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="int16", device=device)
    sd.wait()
    _save_wav(out_path, audio)
    print(f">>> 已存檔：{out_path}")
    return out_path


def _save_wav(path, audio):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def list_microphones():
    """列出所有麥克風（含編號），demo 前確認 BRIO 300 用。"""
    if not _HAS_AUDIO:
        print("尚未安裝 sounddevice。")
        return
    print("可用的輸入裝置（麥克風）：")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  #{idx}  {dev['name']}")
