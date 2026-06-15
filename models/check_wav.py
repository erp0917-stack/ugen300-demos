# -*- coding: utf-8 -*-
"""
check_wav.py <wav_path>
檢查一個 WAV 是否符合 Whisper 期待的格式：16kHz / mono / 16-bit。
順便印出長度（秒）與 RMS 振幅，確認真有音訊內容。
"""
import sys
import wave
import numpy as np

WHISPER_SR = 16000
WHISPER_CHANNELS = 1
WHISPER_SAMPLEWIDTH = 2  # int16

def main(path):
    print(f"check: {path}")
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n  = w.getnframes()
        raw = w.readframes(n)

    duration = n / float(sr) if sr else 0.0
    audio = np.frombuffer(raw, dtype=np.int16) if sw == 2 else np.frombuffer(raw, dtype=np.uint8)
    if audio.size == 0:
        rms = 0.0
    else:
        x = audio.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x)))

    print(f"  channels    : {ch}     (whisper wants {WHISPER_CHANNELS})")
    print(f"  sample_width: {sw} byte(s) (whisper wants {WHISPER_SAMPLEWIDTH})")
    print(f"  framerate   : {sr} Hz   (whisper wants {WHISPER_SR})")
    print(f"  n_samples   : {n}")
    print(f"  duration    : {duration:.2f} sec")
    print(f"  rms (peak ~1.0): {rms:.4f}")

    ok = (ch == WHISPER_CHANNELS and sw == WHISPER_SAMPLEWIDTH and sr == WHISPER_SR)
    print(f"  whisper-ready: {'YES' if ok else 'NO'}")
    return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python check_wav.py <wav_path>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
