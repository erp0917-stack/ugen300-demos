# -*- coding: utf-8 -*-
"""
make_sample_audio_zh_long.py
============================
SAPI Hanhan 合成「正常長度」會議音檔（~30s），結構同 smoke test FAKE_TRANSCRIPT：
多人、多個待辦、有期限、有預算決議。輸出 16kHz / mono / 16-bit WAV。
"""
import os
import sys
import wave
import win32com.client

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_audio_zh_long.wav")

# 純中文 + SSML 在每句之間插 600ms 停頓
# （SAPI 連續快速念會讓 Whisper-Base 分段失準、漏字；句間停頓讓 Whisper
#  正確切 segment，這也比較接近真實會議講話節奏）
SENTENCES = [
    "好，我們今天開會，主要討論下週的產品發表會。",
    "第一個，行銷部分，請小美在週三前，把新聞稿初稿交給我。",
    "研發這邊，阿志負責把展示機在週四前準備好，要兩台備援。",
    "然後通路的部分，我們決定先鎖定北部三家經銷商，這個是確定的。",
    "預算的話，這次發表會控制在二十萬以內，超過要再簽核。",
    "好，那就這樣，散會。",
]
# 用 SAPI SSML 的 silence 標籤把停頓塞進每句之間
SILENCE = '<silence msec="600"/>'
SSML = SILENCE.join(SENTENCES)
TEXT = SSML

SAFT_16K_16BIT_MONO = 18
SSFM_CREATE_FOR_WRITE = 3


def main():
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    chosen = None
    for token in voice.GetVoices():
        if "Hanhan" in token.GetDescription():
            chosen = token
            break
    if chosen is None:
        print("[tts] Hanhan voice not found", file=sys.stderr)
        return 1
    voice.Voice = chosen
    print(f"[tts] voice: {chosen.GetDescription()}")

    # 正常語速（rate=0 是預設），太慢反而會引入合成噪音
    voice.Rate = 0

    fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = SAFT_16K_16BIT_MONO

    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Format = fmt
    stream.Open(DEST, SSFM_CREATE_FOR_WRITE, False)
    voice.AudioOutputStream = stream
    # flag 8 = SVSFIsXML，啟用 SSML（讓 <silence msec="600"/> 生效）
    voice.Speak(TEXT, 8)
    stream.Close()

    with wave.open(DEST, "rb") as w:
        dur = w.getnframes() / w.getframerate()
        print(f"[tts] wrote: {DEST}")
        print(f"[tts] channels={w.getnchannels()} sample_width={w.getsampwidth()}B "
              f"framerate={w.getframerate()}Hz duration={dur:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
