# -*- coding: utf-8 -*-
"""
face_report.py — AI 顏值報告主程式（雙模型流水線）

用法：
    python face_report.py <圖片路徑> [--lang tw|cn|en] [--out filename.png]
    python face_report.py --camera    [--lang tw|cn|en] [--out filename.png]
"""

import argparse
import sys
import os
import io
import json
import time
import hashlib
import threading
from contextlib import redirect_stdout

import cv2
import numpy as np

import vlm_client
from vlm_client import ask_about_image
from face_helpers import (
    LANG, SCORE_KEYS,
    _build_vl_prompts, parse_vl_json,
    _build_llm_prompts, parse_llm_json,
    merge_vl_llm,
)
from sketch import make_dodge_sketch
from report_card import make_report_png

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_PAD     = 0.20

# ── 終端機編碼容錯（cp950 無法顯示稀有 Unicode 字元時用 ? 代替）────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# ── ANSI 顏色（Windows 10+ 啟用虛擬終端機後生效）────────────────────────
os.system("")   # 開啟 Windows ANSI 支援
R    = "\033[0m"
B    = "\033[1m"
DIM  = "\033[2m"
CY   = "\033[96m"
GR   = "\033[92m"
YE   = "\033[93m"
WH   = "\033[97m"
GREY = "\033[90m"
LINE = "═" * 60

# ── VL 快取 ───────────────────────────────────────────────────────────────
_VL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vl_cache")
_CAPTURE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".capture")


def _crop_hash(crop_bgr):
    return hashlib.md5(np.ascontiguousarray(crop_bgr).tobytes()).hexdigest()[:20]


def _vl_cache_load(key, lang):
    path = os.path.join(_VL_CACHE_DIR, f"{key}_{lang}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[VL] 快取命中 {key[:8]}…（跳過推論）")
            return data
        except Exception:
            pass
    return None


def _vl_cache_save(key, lang, vl_data):
    os.makedirs(_VL_CACHE_DIR, exist_ok=True)
    path = os.path.join(_VL_CACHE_DIR, f"{key}_{lang}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vl_data, f, ensure_ascii=False, indent=2)
    print(f"[VL] 結果已快取 → {os.path.basename(path)}")


# ── 自旋動畫 ─────────────────────────────────────────────────────────────
def _run_with_spinner(label, work_fn):
    """背景 thread 跑 work_fn，前台顯示 ◐◓◑◒ 自旋動畫。回傳 work_fn 的結果。"""
    real_out = sys.stdout   # 在 redirect 前先捕捉真實 stdout
    result   = {}
    done     = threading.Event()

    def _worker():
        with redirect_stdout(io.StringIO()):   # 吃掉推論期間的所有 print
            result["v"] = work_fn()
        done.set()

    th = threading.Thread(target=_worker, daemon=True)
    t0 = time.time()
    th.start()

    frames = "◐◓◑◒"
    i = 0
    while not done.wait(0.15):
        el = int(time.time() - t0)
        real_out.write(f"\r   {CY}{frames[i % 4]}{R} {label}  {GREY}({el}s){R}           ")
        real_out.flush()
        i += 1

    th.join()
    real_out.write("\r" + " " * 72 + "\r")
    real_out.flush()
    return result.get("v")


# ── 鏡頭拍照 ─────────────────────────────────────────────────────────────
def take_selfie(camera_index=0, camera_name=""):
    """開鏡頭→倒數 10 秒→拍照→白閃→存暫存→關視窗。回傳暫存路徑，取消回傳 None。"""
    from selfie_timer import countdown_selfie, _open_camera

    os.makedirs(_CAPTURE_DIR, exist_ok=True)
    out_path = os.path.join(_CAPTURE_DIR, "selfie.jpg")

    # 統一用 _open_camera 開，後端與 detect_camera() 完全一致（CAP_DSHOW）
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        print(f"{YE}[警告] 無法開啟攝影機（device={camera_index}）{R}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    label = f"{camera_name}  " if camera_name else ""
    print(f"[鏡頭] take_selfie 實際開啟 index={camera_index}  {label}{w}x{h}", flush=True)

    WIN = "UGen300 AI Beauty  |  Press Q to cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    frame, cancelled = countdown_selfie(cap, WIN, seconds=10)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)   # flush 視窗訊息佇列

    if cancelled or frame is None:
        print(f"\n{YE}已取消拍照。{R}")
        return None

    cv2.imwrite(out_path, frame)
    return out_path


# ── 步驟 1：偵臉裁圖 ─────────────────────────────────────────────────────
def detect_and_crop(frame):
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    faces    = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        print("[偵臉] 未偵測到人臉，使用整張圖。")
        return frame, False

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    H, W = frame.shape[:2]
    px, py = int(w * FACE_PAD), int(h * FACE_PAD)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(W, x + w + px), min(H, y + h + py)
    print(f"[偵臉] 偵測到 {len(faces)} 張臉，裁最大臉 ({x},{y},{w},{h}) "
          f"→ crop ({x1},{y1},{x2},{y2})")
    return frame[y1:y2, x1:x2], True


# ── 步驟 2：VL 看圖 ──────────────────────────────────────────────────────
def run_vl(crop, lang):
    """呼叫 Qwen2-VL-2B，回傳 vl_data。快取命中時跳過推論。結束後呼叫端負責 close。"""
    vl_lang = "tw" if lang == "cn" else lang
    key     = _crop_hash(crop)

    cached = _vl_cache_load(key, vl_lang)
    if cached is not None:
        return cached

    system, question = _build_vl_prompts(vl_lang)
    print(f"[VL] 送進 Qwen2-VL-2B（lang={vl_lang}）……")
    raw = ask_about_image(crop, question, system_prompt=system,
                          max_tokens=400, temperature=0.10)
    print("\n===== VL 原始輸出 =====")
    print(raw)
    cfg     = LANG["tw" if lang == "cn" else lang]
    vl_data, strict = parse_vl_json(raw, cfg["purify"])
    print(f"\n===== VL 解析結果（{'JSON 一次過' if strict else 'regex 救援'}）=====")
    print(f"scores: {vl_data['scores']}")
    print(f"obs:    {vl_data['obs']}")
    _vl_cache_save(key, vl_lang, vl_data)
    return vl_data


# ── 步驟 3：LLM 擴寫 ─────────────────────────────────────────────────────
def run_llm(vl_data, lang):
    """呼叫 Qwen2.5-1.5B，回傳 llm_data。結束後呼叫端負責 close。"""
    import llm_client
    llm_lang         = "tw" if lang == "cn" else lang
    system, user_msg = _build_llm_prompts(llm_lang, vl_data)
    print(f"\n[LLM] 送進 Qwen2.5-1.5B（lang={llm_lang}）……")
    raw = llm_client.generate(user_msg, system_prompt=system,
                               max_tokens=800, temperature=0.15)
    print("\n===== LLM 原始輸出 =====")
    print(raw)
    purify           = LANG[lang]["purify"]
    llm_data, strict = parse_llm_json(raw, vl_data["scores"], vl_data["obs"],
                                       purify, lang=lang)
    print(f"\n===== LLM 解析結果（{'JSON 一次過' if strict else 'regex 救援'}）=====")
    print(f"meta:    {llm_data['meta']}")
    print(f"summary: {llm_data['summary']}")
    print(f"parts:   {len(llm_data['parts'])} 筆")
    print(f"pros:    {llm_data['pros']}")
    print(f"cons:    {llm_data['cons']}")
    print(f"tips:    {llm_data['tips']}")
    return llm_data


# ── 主流程 ────────────────────────────────────────────────────────────────
def run(img_path, lang, out_path, use_spinner=False):
    """
    共用 pipeline 後半段（偵臉→VL→LLM→素描→PNG）。
    use_spinner=True：過場動畫模式（camera 模式用）；False：詳細 print 模式。
    """
    import llm_client as _llmc

    cfg   = LANG[lang]
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[X] 讀不到圖片：{img_path}")
        sys.exit(1)

    if not use_spinner:
        print(f"[OK] 圖片 {img_path}  {frame.shape[1]}x{frame.shape[0]}  語言={cfg['name']}")

    # 步驟 1：偵臉
    crop, _ = detect_and_crop(frame)
    if not use_spinner:
        print(f"[偵臉] 裁圖尺寸：{crop.shape[1]}x{crop.shape[0]}")

    # 步驟 2 + 3：推論（spinner 模式包自旋動畫，一般模式直接印 log）
    if use_spinner:
        # ── [1/2] 視覺模型 ──
        print(f"\n{B}{CY}[1/2]{R} {WH}視覺模型 Qwen2-VL-2B 正在觀察五官…{R}")

        def _vl_job():
            d = run_vl(crop, lang)
            vlm_client.close()
            return d

        vl_data = _run_with_spinner("觀察五官中", _vl_job)
        print(f"   {GR}✓{R} {DIM}觀察完成，釋放 VDevice{R}")

        # ── [2/2] 語言模型 ──
        print(f"\n{B}{CY}[2/2]{R} {WH}語言模型 Qwen2.5-1.5B 正在撰寫報告…{R}")

        def _llm_job():
            d = run_llm(vl_data, lang)
            _llmc.close()
            return d

        llm_data = _run_with_spinner("撰寫報告中", _llm_job)
        print(f"   {GR}✓{R} {DIM}報告完成{R}")

    else:
        # 一般模式：完整 debug log
        vl_data = run_vl(crop, lang)
        print("\n[VL] 推論完成，釋放 VDevice……")
        vlm_client.close()
        print("[VL] VDevice 已釋放。")

        llm_data = run_llm(vl_data, lang)
        print("\n[LLM] 推論完成，釋放 VDevice……")
        _llmc.close()
        print("[LLM] VDevice 已釋放。")

    # 合併 + 素描 + PNG
    data = merge_vl_llm(vl_data, llm_data)
    if not use_spinner:
        print(f"\n[合併] overall={data['overall']}  parts={len(data['parts'])} 筆")
        print("[素描] 產生 dodge 線稿……")

    sketch = make_dodge_sketch(crop)

    if not use_spinner:
        print("[報告] 產生 PNG……")

    make_report_png(crop, sketch, data, lang, cfg, out_path)

    abs_path = os.path.abspath(out_path)
    if use_spinner:
        print(f"\n{LINE}")
        print(f"   {GR}✓{R} {B}報告已生成，開啟中…{R}")
        print(f"   {GREY}→ {abs_path}{R}")
        print(LINE)
        os.startfile(abs_path)
    else:
        print(f"\n完成！請開啟：{abs_path}")

    return data


# ── CLI 入口 ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UGen300 AI 顏值報告")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("photo",   nargs="?", default=None,
                       help="圖片路徑，例如 ..\\V2\\face.jpg")
    group.add_argument("--camera", action="store_true",
                       help="開啟筆電鏡頭倒數自拍")
    parser.add_argument("--lang", choices=["tw", "cn", "en"], default="tw")
    parser.add_argument("--out",  default=None,
                       help="輸出 PNG（預設 face_report_<lang>.png）")
    args = parser.parse_args()

    out_path = args.out or f"face_report_{args.lang}.png"

    if args.camera:
        # ── 鏡頭模式：過場 banner → 偵測鏡頭 → 拍照 → spinner pipeline ──
        print(f"\n{LINE}")
        print(f"  {B}{WH}UGen300 離線 AI 顏值分析{R}")
        print(f"  {DIM}一支 USB 棒，兩顆 AI 接力{R}")
        print(LINE)

        from selfie_timer import detect_camera
        print(f"\n{GREY}   正在偵測鏡頭…{R}")
        result = detect_camera()
        if result is None:
            print(f"{YE}找不到可用鏡頭，請確認鏡頭連接。{R}")
            sys.exit(0)
        cam_idx, cam_name = result
        print(f"   {GR}✓{R} 使用鏡頭 index={cam_idx}：{cam_name}")

        print(f"\n{YE}▶ 即將開啟鏡頭，請站好，10 秒後自動拍照。{R}")
        print(f"{GREY}  按 Q 可取消。{R}\n")

        img_path = take_selfie(cam_idx, cam_name)
        if img_path is None:
            sys.exit(0)

        print(f"\n{LINE}")
        print(f"  {B}{CY}AI 分析開始…{R}")
        print(LINE)

        run(img_path, args.lang, out_path, use_spinner=True)

    else:
        # ── 讀檔模式：原有詳細 log ──
        if args.photo is None:
            parser.error("請提供圖片路徑，或加上 --camera 使用鏡頭。")
        run(args.photo, args.lang, out_path, use_spinner=False)


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
