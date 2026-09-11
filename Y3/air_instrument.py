# -*- coding: utf-8 -*-
"""
air_instrument.py  ——  空氣樂器（S3 / 對應發想 W-05）
====================================================
對著筆電鏡頭揮手，手揮到畫面不同位置就發出不同音（像空中的鍵盤/木琴）。
純娛樂互動、暖場神器。

操作：站到鏡頭前揮動雙手，畫面分成幾個彩色音區，手揮進去就出聲。按 q 離開。
跑法：python air_instrument.py --hef yolov8s_pose.hef --source 1

⚠️ demo 需求：要有喇叭/音響，音效才出得來。
⚠️ 音效檔：預設用 Windows 內建 winsound 發出不同音高的嗶聲（最簡單、零素材）。
   想用真實樂器音色，可放 do.wav re.wav... 在 sounds/ 資料夾，程式會自動改用（見下方）。
"""

import argparse
import os
import sys
import cv2

from hailo_pose import PoseEstimator
from air_music_logic import AirInstrument

# 音區顏色（畫面上的彩色直條）
ZONE_COLORS = [
    (216, 212, 0), (0, 200, 120), (0, 212, 216),
    (80, 80, 220), (200, 120, 255), (120, 200, 255), (0, 165, 255),
]

# 各音區對應的音高（Hz），給 winsound 用（Do Re Mi Fa Sol La Si）
NOTE_FREQS = [262, 294, 330, 349, 392, 440, 494]


def make_player(num_zones):
    """
    回傳一個 play(zone) 函式。
    優先順序：
      ① sounds/*.wav  → winsound 播 wav（保留原本素材用法）
      ② numpy 合成 sine + sounddevice  ← 最穩，預設走這條
      ③ numpy 合成 sine + winsound.PlaySound(SND_MEMORY)（沒裝 sounddevice 用）
      ④ 都不行：只印
    """
    sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")
    wavs = []
    if os.path.isdir(sounds_dir):
        wavs = sorted(f for f in os.listdir(sounds_dir) if f.lower().endswith(".wav"))

    # ① 有 wav 檔 → 用 winsound 播 wav（Windows）
    if wavs and sys.platform.startswith("win"):
        import winsound
        def play(zone):
            f = NOTE_FREQS[zone % len(NOTE_FREQS)]
            print(f"♪ 音區 {zone} → 播放 {wavs[zone] if zone < len(wavs) else '?'}（{f}Hz）", flush=True)
            if zone < len(wavs):
                winsound.PlaySound(os.path.join(sounds_dir, wavs[zone]),
                                   winsound.SND_FILENAME | winsound.SND_ASYNC)
        return play

    # ② numpy sine + sounddevice ── 最穩，每次都會出聲
    try:
        import numpy as np
        import sounddevice as sd
        SR = 44100
        DURATION = 0.22  # 一個音的長度（秒）
        t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False, dtype=np.float32)
        tones = []
        for f in NOTE_FREQS:
            sine = np.sin(2.0 * np.pi * f * t).astype(np.float32)
            # tanh 軟削波：sine 灌 4 倍進 tanh，輸出帶有奇數諧波（接近方波），
            # 同樣 peak amplitude 下感知響度提升 ~6~10 dB。
            warm = np.tanh(4.0 * sine).astype(np.float32)
            # 包絡：短 attack（無 click）、短 release，中間 sustain 盡量長
            env = np.ones_like(warm)
            attack  = int(SR * 0.005)
            release = int(SR * 0.025)
            env[:attack]   = np.linspace(0.0, 1.0, attack,  dtype=np.float32)
            env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
            tone = warm * env
            # 正規化到 0.95 peak（盡量逼近滿幅、留一點 headroom 防破音）
            peak = float(np.max(np.abs(tone)))
            if peak > 0:
                tone = (tone * (0.95 / peak)).astype(np.float32)
            tones.append(tone)
        # 探測 + 印出實際輸出裝置（方便 debug 音量去哪了）
        sd.play(np.zeros(8, dtype=np.float32), SR)
        sd.wait()
        try:
            dev_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            dev_info = sd.query_devices(dev_idx)
            print(f"[air] 輸出裝置：{dev_info['name']}  （{int(dev_info['default_samplerate'])}Hz）")
        except Exception:
            pass
        def play(zone):
            f = NOTE_FREQS[zone % len(NOTE_FREQS)]
            print(f"♪ 音區 {zone} → 播放音調 {f}Hz（sounddevice / tanh-warm）", flush=True)
            try:
                sd.stop()
                sd.play(tones[zone % len(tones)], SR, blocking=False)
            except Exception as e:
                print(f"[air] sd.play 例外：{e}", flush=True)
        print(f"[air] 音效後端：sounddevice + tanh 軟削波（峰值 0.95）")
        return play
    except Exception as e:
        print(f"[air] sounddevice 不可用：{e}；改用 winsound in-memory WAV")

    # ③ numpy sine 寫進記憶體 WAV，用 winsound.PlaySound(SND_MEMORY) 播
    if sys.platform.startswith("win"):
        try:
            import winsound, wave, io
            import numpy as np
            SR = 22050
            DURATION = 0.22
            wav_bytes_per_note = []
            for f in NOTE_FREQS:
                n = int(SR * DURATION)
                t = np.arange(n, dtype=np.float32) / SR
                env = np.ones(n, dtype=np.float32)
                attack = int(SR * 0.005); release = int(SR * 0.025)
                env[:attack]  = np.linspace(0.0, 1.0, attack)
                env[-release:] = np.linspace(1.0, 0.0, release)
                sine = np.sin(2.0 * np.pi * f * t)
                warm = np.tanh(4.0 * sine).astype(np.float32)  # 加諧波，大幅提升感知響度
                tone = warm * env
                peak = float(np.max(np.abs(tone)))
                if peak > 0:
                    tone = tone * (0.95 / peak)
                samples = (tone * 32767).astype('<i2')
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
                    wf.writeframes(samples.tobytes())
                wav_bytes_per_note.append(buf.getvalue())
            def play(zone):
                f = NOTE_FREQS[zone % len(NOTE_FREQS)]
                print(f"♪ 音區 {zone} → 播放音調 {f}Hz（winsound mem）", flush=True)
                winsound.PlaySound(wav_bytes_per_note[zone % len(wav_bytes_per_note)],
                                   winsound.SND_MEMORY | winsound.SND_ASYNC)
            print("[air] 音效後端：winsound.PlaySound (SND_MEMORY)")
            return play
        except Exception as e:
            print(f"[air] in-memory WAV 也失敗：{e}")

    # ④ 都不行：只印
    def play(zone):
        f = NOTE_FREQS[zone % len(NOTE_FREQS)]
        print(f"♪ 音區 {zone} → {f}Hz（無音效後端，只印）", flush=True)
    return play


def draw_zones(frame, num_zones, fired_zones):
    """畫出彩色音區，被觸發的區塊高亮"""
    h, w = frame.shape[:2]
    zw = w // num_zones
    overlay = frame.copy()
    for i in range(num_zones):
        x1, x2 = i * zw, (i + 1) * zw
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        alpha = 0.45 if i in fired_zones else 0.12
        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.line(frame, (x2, 0), (x2, h), (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="yolov8s_pose.hef")
    ap.add_argument("--source", default="0")
    ap.add_argument("--zones", type=int, default=5, help="幾個音區（音）")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[air] 無法開啟影像來源：{args.source}（外接 webcam 試 --source 1）")
        return

    ok, frame = cap.read()
    width = frame.shape[1] if ok else 640

    pose = PoseEstimator(args.hef)
    inst = AirInstrument(num_zones=args.zones, width=width)
    play = make_player(args.zones)

    print(f"[air] 空氣樂器啟動，{args.zones} 個音區。揮手玩玩看，按 q 離開。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            keypoints = pose.infer(frame)
            fired = inst.update(keypoints) if keypoints else []
            for zone in fired:
                play(zone)
            frame = draw_zones(frame, args.zones, set(fired))
            # 畫手腕點
            if keypoints:
                for idx in (9, 10):
                    x, y, c = keypoints[idx]
                    if c >= 0.5:
                        cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 3)
            cv2.imshow("UGen300 Air Instrument (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("[air] 已結束。")


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
