# -*- coding: utf-8 -*-
"""
make_sample_audio_zh.py
=======================
用 Windows SAPI（pywin32）合成一段繁體中文語音，輸出 Whisper 期待的
16kHz、mono、16-bit PCM WAV。Python 端會在跑 Whisper 時把 int16
正規化成 float32，所以這裡只負責產生規格正確的 WAV。
"""
import os
import win32com.client
import pywintypes
import wave
import sys

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_audio_zh.wav")
TEXT = "今天會議結束後，請小張在週三前把報告交出來。預算控制在二十萬以內。"

# SpeechAudioFormatType.SAFT16kHz16BitMono = 18 (符合 Whisper)
SAFT_16K_16BIT_MONO = 18
SSFM_CREATE_FOR_WRITE = 3

def main():
    voice = win32com.client.Dispatch("SAPI.SpVoice")

    # 找 Hanhan（zh-TW）；沒有就 fallback 第一個 zh 開頭的
    found = None
    fallback = None
    for token in voice.GetVoices():
        desc = token.GetDescription()
        if "Hanhan" in desc:
            found = token
            break
        if fallback is None and "Chinese" in desc:
            fallback = token
    chosen = found or fallback
    if chosen is None:
        print("[tts] 找不到任何中文 voice", file=sys.stderr)
        return 1
    voice.Voice = chosen
    print(f"[tts] voice: {chosen.GetDescription()}")

    fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = SAFT_16K_16BIT_MONO

    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Format = fmt
    stream.Open(DEST, SSFM_CREATE_FOR_WRITE, False)
    voice.AudioOutputStream = stream
    voice.Speak(TEXT, 0)
    stream.Close()
    print(f"[tts] wrote: {DEST}")

    # 驗證寫出的 WAV 規格
    with wave.open(DEST, "rb") as w:
        print(f"[tts] channels={w.getnchannels()} sample_width={w.getsampwidth()}B "
              f"framerate={w.getframerate()}Hz nframes={w.getnframes()} "
              f"duration={w.getnframes()/w.getframerate():.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
