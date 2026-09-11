# -*- coding: utf-8 -*-
"""
meeting_summary.py  ——  WQ1 離線會議記錄主程式（三語介面版）
==========================================================
開頭選「版本」三選一：English / 繁體中文 / 简体中文。選了哪個版本，整支程式
的介面文字、逐字稿、摘要、存檔，全部都用那個語言；對麥克風就講那個語言。

  English  → 講英文 → 英文介面 + 英文轉錄 + 英文摘要
  繁體中文 → 講中文 → 繁體介面 + 中文轉錄 + 摘要（OpenCC 確保繁體）
  简体中文 → 講中文 → 簡體介面 + 中文轉錄 + 摘要（OpenCC 轉簡體）

簡繁轉換優先用 OpenCC；沒裝就用內建常用字對照表頂著（英文模式不受影響）。

麥克風：預設用「系統預設裝置」（插什麼用什麼，不綁 BRIO）。
  --mic pick   → 列出所有收音裝置讓使用者挑（分享給別人用很方便）
  --mic BRIO   → 指定名稱關鍵字　|　--mic 2 → 指定編號

⚠️ 只動這支外層主程式。transcribe.py / summarize.py / record_audio.py 都不動。

操作：空白鍵 = 開始/停止錄音；做完一段：空白鍵 = 再錄一段，Q = 結束
跑法：
   python meeting_summary.py                 # 開頭互動選語言 + 系統預設麥克風
   python meeting_summary.py --lang en       # 直接英文版（跳過語言選單）
   python meeting_summary.py --mic pick      # 開頭讓你挑麥克風
"""

import argparse
import io
import os
import sys
import time
import threading
from contextlib import redirect_stdout

from transcribe import transcribe
from transcribe import close as transcribe_close
from summarize import summarize
from summarize import close as summarize_close

# ── 終端機顏色（Windows 10+ 開 ANSI）──────────────────────────────────
os.system("")
R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
CY = "\033[96m"; GR = "\033[92m"; YE = "\033[93m"; WH = "\033[97m"; GREY = "\033[90m"
LINE = "═" * 60

# ── 介面語言（選單選完後設定）────────────────────────────────────────
_LANG = "zh"   # 'en' / 'zh'(繁體) / 'cn'(簡體)


# ====================================================================
#  介面文字總表：每句列三版（en / zh 繁 / cn 簡）。要改字、加語言都改這裡。
# ====================================================================
STRINGS = {
    "banner": {
        "en": "WQ1 Offline Meeting Notes  |  UGen300 fully on-device · offline",
        "zh": "WQ1 離線會議記錄　|　UGen300 全程本機運算 · 不連網",
        "cn": "WQ1 离线会议记录　|　UGen300 全程本机运算 · 不联网",
    },
    "sec_prerecorded": {"en": "Using pre-recorded audio", "zh": "使用預錄音檔", "cn": "使用预录音档"},
    "sec_live": {"en": "Live recording", "zh": "現場錄音", "cn": "现场录音"},
    "sec_step1": {
        "en": "Step 1/2  Speech to Text (UGen300 · Whisper)",
        "zh": "步驟 1／2　語音轉文字（UGen300 · Whisper）",
        "cn": "步骤 1／2　语音转文字（UGen300 · Whisper）",
    },
    "sec_step2": {
        "en": "Step 2/2  Summarize (UGen300 · Qwen2.5)",
        "zh": "步驟 2／2　整理會議摘要（UGen300 · Qwen2.5）",
        "cn": "步骤 2／2　整理会议摘要（UGen300 · Qwen2.5）",
    },
    "sec_transcript": {"en": "Transcript", "zh": "逐字稿", "cn": "逐字稿"},
    "sec_summary": {"en": "Summary", "zh": "會議摘要", "cn": "会议摘要"},

    "rec_press_start": {
        "en": ">>> Press [SPACE] to start recording…",
        "zh": ">>> 按【空白鍵】開始錄音…",
        "cn": ">>> 按【空格键】开始录音…",
    },
    "rec_press_start_enter": {
        "en": ">>> Press Enter to start recording…",
        "zh": ">>> 按 Enter 開始錄音…",
        "cn": ">>> 按 Enter 开始录音…",
    },
    "rec_recording_space": {
        "en": ">>> ● Recording… press [SPACE] again to stop.",
        "zh": ">>> ● 錄音中…講完再按一次【空白鍵】停止。",
        "cn": ">>> ● 录音中…讲完再按一次【空格键】停止。",
    },
    "rec_recording_enter": {
        "en": ">>> ● Recording… press Enter again to stop.",
        "zh": ">>> ● 錄音中…講完再按一次 Enter 停止。",
        "cn": ">>> ● 录音中…讲完再按一次 Enter 停止。",
    },
    "rec_stopped": {"en": ">>> Recording stopped.", "zh": ">>> 錄音結束。", "cn": ">>> 录音结束。"},
    "rec_no_audio": {
        "en": ">>> No audio captured, please try again.",
        "zh": ">>> 沒有收到聲音，請再試一次。",
        "cn": ">>> 没有收到声音，请再试一次。",
    },

    "mic_using": {"en": ">>> Using microphone: {name}", "zh": ">>> 使用麥克風：{name}", "cn": ">>> 使用麦克风：{name}"},
    "mic_default": {"en": ">>> Using system default microphone.", "zh": ">>> 使用系統預設麥克風。", "cn": ">>> 使用系统默认麦克风。"},
    "mic_pick_title": {"en": "Select a microphone:", "zh": "選擇麥克風：", "cn": "选择麦克风："},
    "mic_pick_prompt": {
        "en": "  Enter number (Enter = system default): ",
        "zh": "  輸入編號（直接 Enter = 系統預設）：",
        "cn": "  输入编号（直接 Enter = 系统默认）：",
    },
    "mic_pick_invalid": {"en": "  Invalid choice, try again.", "zh": "  輸入無效，請重試。", "cn": "  输入无效，请重试。"},

    "ask_again": {
        "en": "  Record another? [SPACE] = record again  |  Q = quit",
        "zh": "  再錄一段？　【空白鍵】= 再錄一段　｜　Q = 結束",
        "cn": "  再录一段？　【空格键】= 再录一段　｜　Q = 结束",
    },
    "ask_again_enter": {
        "en": "  Type y to record again, any other key to quit: ",
        "zh": "  輸入 y 再錄一段、其他鍵結束：",
        "cn": "  输入 y 再录一段、其他键结束：",
    },

    "spin_transcribe": {"en": "Transcribing…", "zh": "正在轉文字…", "cn": "正在转文字…"},
    "spin_summarize": {"en": "Summarizing…", "zh": "正在整理摘要…", "cn": "正在整理摘要…"},
    "spin_elapsed": {
        "en": "elapsed {n}s · UGen300 on-device · offline",
        "zh": "已 {n} 秒 · UGen300 本機運算中 · 不連網",
        "cn": "已 {n} 秒 · UGen300 本机运算中 · 不联网",
    },
    "spin_done": {"en": "done ({sec}s)", "zh": "完成（{sec} 秒）", "cn": "完成（{sec} 秒）"},

    "all_local": {
        "en": "  ✓ Everything ran on-device / UGen300. Nothing sent to the cloud.",
        "zh": "  ✓ 全程在本機 / UGen300 完成，資料未連網外傳。",
        "cn": "  ✓ 全程在本机 / UGen300 完成，数据未联网外传。",
    },
    "saved": {"en": "  Saved: {path}", "zh": "  已存檔：{path}", "cn": "  已存档：{path}"},
    "error": {"en": "  Something went wrong: {e}", "zh": "  處理時發生問題：{e}", "cn": "  处理时发生问题：{e}"},
    "bye": {"en": "  Thanks for using it. Goodbye!", "zh": "  感謝使用，再見！", "cn": "  感谢使用，再见！"},
    "audio_not_found": {"en": "Audio file not found: {path}", "zh": "找不到音檔：{path}", "cn": "找不到音档：{path}"},

    "save_transcript_head": {"en": "[Transcript]", "zh": "【逐字稿】", "cn": "【逐字稿】"},
    "save_summary_head": {"en": "[Summary]", "zh": "【摘要】", "cn": "【摘要】"},
}


def T(key, **kw):
    """查表取得目前語言的介面字串。"""
    s = STRINGS.get(key, {}).get(_LANG) or STRINGS.get(key, {}).get("en") or key
    return s.format(**kw) if kw else s


# ====================================================================
#  簡繁轉換（優先 OpenCC，沒裝就用內建常用字對照表）
# ====================================================================
def _build_s2t_pairs():
    return {
        "这": "這", "来": "來", "们": "們", "个": "個", "东": "東", "车": "車",
        "长": "長", "时": "時", "现": "現", "场": "場", "风": "風", "协": "協",
        "调": "調", "体": "體", "质": "質", "审": "審", "简": "簡", "约": "約",
        "单": "單", "为": "為", "较": "較", "会": "會", "宽": "寬", "适": "適",
        "应": "應", "让": "讓", "过": "過", "还": "還", "没": "沒", "说": "說",
        "话": "話", "问": "問", "题": "題", "张": "張", "开": "開", "关": "關",
        "门": "門", "间": "間", "实": "實", "际": "際", "业": "業", "产": "產",
        "务": "務", "动": "動", "团": "團", "国": "國", "园": "園", "图": "圖",
        "学": "學", "觉": "覺", "网": "網", "线": "線", "红": "紅", "绿": "綠",
        "蓝": "藍", "颜": "顏", "设": "設", "计": "計", "师": "師", "样": "樣",
        "总": "總", "结": "結", "议": "議", "评": "評", "价": "價", "优": "優",
        "点": "點", "选": "選", "择": "擇", "标": "標", "类": "類", "别": "別",
        "显": "顯", "当": "當", "装": "裝", "数": "數", "据": "據", "处": "處",
        "确": "確", "认": "認", "与": "與", "给": "給", "职": "職", "气": "氣",
        "经": "經", "继": "繼", "续": "續", "传": "傳", "统": "統", "细": "細",
        "节": "節", "买": "買", "卖": "賣", "贵": "貴", "钱": "錢", "进": "進",
        "运": "運", "负": "負", "责": "責", "决": "決", "议": "議", "钟": "鐘",
        "听": "聽", "区": "區", "录": "錄", "项": "項", "办": "辦", "务": "務",
        "员": "員", "确": "確", "记": "記", "讨": "討", "论": "論", "划": "劃",
        # 常見補強（讓沒裝 OpenCC 時簡體也別太難看）
        "请": "請", "户": "戶", "训": "訓", "练": "練", "导": "導", "帮": "幫",
        "报": "報", "则": "則", "极": "極", "构": "構", "层": "層", "织": "織",
        "纯": "純", "维": "維", "丰": "豐", "满": "滿", "种": "種", "档": "檔",
        "镜": "鏡", "头": "頭", "脸": "臉", "营": "營", "养": "養", "态": "態",
        "众": "眾", "严": "嚴", "齐": "齊", "宁": "寧", "财": "財", "购": "購",
        "费": "費", "资": "資", "馆": "館", "饭": "飯", "纸": "紙", "组": "組",
        "纳": "納", "绍": "紹", "缩": "縮",
    }


_S2T_PAIRS = _build_s2t_pairs()
_T2S_PAIRS = {v: k for k, v in _S2T_PAIRS.items()}
_FALLBACK_S2T = str.maketrans(_S2T_PAIRS)
_FALLBACK_T2S = str.maketrans(_T2S_PAIRS)

try:
    from opencc import OpenCC
    _CC_S2T = OpenCC("s2t")
    _CC_T2S = OpenCC("t2s")
    _HAS_OPENCC = True
except Exception:
    _HAS_OPENCC = False


def _to_traditional(text):
    if not text:
        return text
    if _HAS_OPENCC:
        try:
            return _CC_S2T.convert(text)
        except Exception:
            pass
    return text.translate(_FALLBACK_S2T)


def _to_simplified(text):
    if not text:
        return text
    if _HAS_OPENCC:
        try:
            return _CC_T2S.convert(text)
        except Exception:
            pass
    return text.translate(_FALLBACK_T2S)


def _localize_cjk(text):
    """依目前介面語言把中文內容轉成對應字體；英文模式原樣回傳。"""
    if _LANG == "zh":
        return _to_traditional(text)
    if _LANG == "cn":
        return _to_simplified(text)
    return text


# ====================================================================
#  畫面排版
# ====================================================================
def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _banner():
    print(CY + B + LINE)
    print("   " + T("banner"))
    print(LINE + R)


def _section(title, color=CY):
    print()
    print(color + B + "── " + title + " " + "─" * max(2, 54 - len(title)) + R)


def _block(text, color=WH):
    for ln in (text or "").splitlines() or [""]:
        print(color + " │ " + R + ln)


# ====================================================================
#  語言選單（在語言選定前，固定用三語並列，誰都看得懂）
# ====================================================================
def _choose_language():
    print()
    print(CY + B + "  Select language / 語言 / 语言" + R)
    print(CY + "    [1] English（default, press Enter）")
    print(CY + "    [2] 繁體中文")
    print(CY + "    [3] 简体中文" + R)
    try:
        import msvcrt
        while True:
            ch = msvcrt.getch()
            if ch in (b"1", b"\r", b"\n"):
                return "en"
            if ch == b"2":
                return "zh"
            if ch == b"3":
                return "cn"
    except Exception:
        ans = input("  1=English / 2=繁體中文 / 3=简体中文 (Enter=English): ").strip()
        return {"2": "zh", "3": "cn"}.get(ans, "en")


# ====================================================================
#  麥克風：預設系統預設、可互動挑選、可指定
# ====================================================================
def _device_name(device):
    try:
        import sounddevice as sd
        d = sd.query_devices(kind="input") if device is None else sd.query_devices(device)
        return d.get("name")
    except Exception:
        return None


def _pick_mic_interactive():
    """列出所有收音裝置讓使用者挑；回傳裝置編號或 None（=系統預設）。"""
    try:
        import sounddevice as sd
    except Exception:
        return None
    devs = [(i, d) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]
    if not devs:
        return None
    print()
    print(CY + B + "  " + T("mic_pick_title") + R)
    for i, d in devs:
        print(CY + f"    [{i}] {d['name']}" + R)
    valid = {i for i, _ in devs}
    while True:
        ans = input(T("mic_pick_prompt")).strip()
        if ans == "":
            return None
        if ans.isdigit() and int(ans) in valid:
            return int(ans)
        print(YE + T("mic_pick_invalid") + R)


def _resolve_mic(mic_arg):
    """把 --mic 參數解析成 sounddevice 的 device 值，並印出實際用哪支。"""
    a = (mic_arg or "auto").lower()
    if a == "pick":
        device = _pick_mic_interactive()
    elif a in ("auto", "default", ""):
        device = None  # 系統預設裝置（插什麼用什麼）
    else:
        from record_audio import find_input_device
        device = find_input_device(mic_arg)
    name = _device_name(device)
    if name:
        print(GR + T("mic_using", name=name) + R)
    else:
        print(GR + T("mic_default") + R)
    return device


# ====================================================================
#  空白鍵控制的錄音
# ====================================================================
def record_with_space(device=None, out_path="meeting.wav"):
    import sounddevice as sd
    import numpy as np
    from record_audio import SAMPLE_RATE, CHANNELS, _save_wav

    try:
        import msvcrt
        def wait_key():
            while True:
                ch = msvcrt.getch()
                if ch == b" ":
                    return "space"
                if ch.lower() == b"q":
                    return "quit"
        have_key = True
    except Exception:
        have_key = False

    if have_key:
        print(YE + T("rec_press_start") + R)
        if wait_key() == "quit":
            return None
    else:
        input(YE + T("rec_press_start_enter") + R)

    print(GR + B + T("rec_recording_space" if have_key else "rec_recording_enter") + R)
    frames = []

    def callback(indata, n, t, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="int16", device=device, callback=callback)
    with stream:
        if have_key:
            wait_key()
        else:
            input()
    print(GREY + T("rec_stopped") + R)

    if not frames:
        print(YE + T("rec_no_audio") + R)
        return None
    audio = np.concatenate(frames, axis=0)
    _save_wav(out_path, audio)
    return out_path


def _ask_again():
    print()
    print(CY + B + T("ask_again") + R)
    try:
        import msvcrt
        while True:
            ch = msvcrt.getch().lower()
            if ch == b" ":
                return True
            if ch == b"q":
                return False
    except Exception:
        ans = input(T("ask_again_enter")).strip().lower()
        return ans == "y"


def _run_with_spinner(label, work_fn):
    spin = "◐◓◑◒"
    result = {}
    real_out = sys.stdout

    def worker():
        with redirect_stdout(io.StringIO()):
            result["value"] = work_fn()

    th = threading.Thread(target=worker, daemon=True)
    t0 = time.time()
    th.start()
    i = 0
    last = -1
    while th.is_alive():
        el = int(time.time() - t0)
        if el != last:
            print(f"   {CY}{spin[i % len(spin)]}{R} {label}  {GREY}{T('spin_elapsed', n=el)}{R}",
                  file=real_out)
            last = el
            i += 1
        time.sleep(0.1)
    th.join()
    el = time.time() - t0
    print(f"{GR}✓{R} {DIM}{T('spin_done', sec=f'{el:.1f}')}{R}", file=real_out)
    return result.get("value", "")


# ====================================================================
#  主流程
# ====================================================================
def process_once(wav_path):
    """轉文字 + 摘要 + 顯示，全部用目前介面語言。回傳 (顯示用逐字稿, 顯示用摘要)。"""
    whisper_lang = "en" if _LANG == "en" else "zh"

    _section(T("sec_step1"))
    transcript = _run_with_spinner(T("spin_transcribe"),
                                   lambda: transcribe(wav_path, language=whisper_lang))
    transcribe_close()
    disp_transcript = _localize_cjk(transcript)
    print()
    _block(disp_transcript, WH)

    # 摘要用「原始」逐字稿去摘（summarize 內部會處理繁體/清洗）；之後再轉字體顯示
    _section(T("sec_step2"))
    summary = _run_with_spinner(T("spin_summarize"), lambda: summarize(transcript))
    summarize_close()
    disp_summary = _localize_cjk(summary)

    _section(T("sec_transcript"), GR); _block(disp_transcript, WH)
    _section(T("sec_summary"), GR);    _block(disp_summary, GR)
    print()
    print(GREY + T("all_local") + R)
    return disp_transcript, disp_summary


def main():
    global _LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="讀預錄 wav 檔（保底用）")
    ap.add_argument("--mic", default="auto",
                    help="auto 系統預設（預設）/ pick 互動挑選 / 名稱關鍵字 / 編號")
    ap.add_argument("--mic-list", action="store_true", help="列出麥克風後結束")
    ap.add_argument("--lang", default=None, help="介面與輸出語言 en / zh / cn；不給則開頭互動選")
    ap.add_argument("--save", default="meeting_summary.txt", help="存檔路徑")
    args = ap.parse_args()

    if args.mic_list:
        from record_audio import list_microphones
        list_microphones()
        return

    # 1) 選語言（之後所有介面文字都跟著它）
    _LANG = (args.lang or "").lower()
    if _LANG not in ("en", "zh", "cn"):
        _LANG = _choose_language()

    # 2) 決定麥克風（預設系統預設；--mic pick 可互動挑）
    device = _resolve_mic(args.mic)

    round_no = 0
    first_audio = args.audio

    while True:
        round_no += 1
        _clear()
        _banner()

        if first_audio and round_no == 1:
            wav_path = first_audio
            if not os.path.exists(wav_path):
                print(YE + T("audio_not_found", path=wav_path) + R)
                return
            _section(T("sec_prerecorded"))
            print("   " + wav_path)
        else:
            _section(T("sec_live"))
            wav_path = record_with_space(device=device)
            if wav_path is None:
                if _ask_again():
                    continue
                break

        try:
            transcript, summary = process_once(wav_path)
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(T("save_transcript_head") + "\n" + (transcript or "") + "\n\n")
                f.write(T("save_summary_head") + "\n" + (summary or "") + "\n")
            print(GREY + T("saved", path=args.save) + R)
        except Exception as e:
            print(YE + T("error", e=e) + R)

        if not _ask_again():
            break

    print()
    print(CY + T("bye") + R)


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
