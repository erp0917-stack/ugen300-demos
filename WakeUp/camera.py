"""
camera.py —— 共用鏡頭選擇:有外接鏡頭就用外接,沒有就用筆電內建。

做法(沿用 FACE1/selfie_timer.py 的驗證過邏輯):
  1. pygrabber 枚舉 DirectShow 裝置名稱(index 與 OpenCV CAP_DSHOW 一致)。
  2. 名稱含內建關鍵字(integrated / built-in / HD camera / IR camera…)視為筆電內建;含 obs / virtual 等視為虛擬鏡頭,跳過。
  3. 逐台試開並連續讀 3 幀確認可用;外接優先,沒外接就用第一台可用的。
用法:cap, idx, name = open_camera("auto")      # 或 "0" / "1" 指定編號
"""
import time

import cv2

_INTERNAL_HINTS = ["integrated", "built-in", "hd camera", "hd webcam", "internal", "facing",
                   "ir camera", "windows hello", "內建", "内建", "asus fhd", "asus ir"]
# 名稱含這些的一定是外接 USB 鏡頭(優先於內建關鍵字;例如 "Logitech HD Webcam C270" 含 hd webcam 但其實是外接)
_EXTERNAL_HINTS = ["logitech", "logi ", "brio", "c920", "c922", "c930", "c270", "c310", "streamcam", "razer", "elgato",
                   "insta360", "obsbot", "anker", "aukey", "外接"]
_VIRTUAL_HINTS = ["obs", "virtual", "droidcam", "manycam", "snap camera"]


def _names():
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def _is_external(name): return any(k in name.lower() for k in _EXTERNAL_HINTS)
def _is_internal(name): return (not _is_external(name)) and any(k in name.lower() for k in _INTERNAL_HINTS)
def _is_virtual(name): return any(k in name.lower() for k in _VIRTUAL_HINTS)


def _open(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def detect_camera(width=1280, height=720, verbose=True):
    """回傳 (index, name) 或 None。外接 > 內建 > None。"""
    names = _names()
    if verbose:
        for i, nm in enumerate(names):
            print(f"[鏡頭] index={i}: {nm}{'(內建)' if _is_internal(nm) else '(虛擬,略過)' if _is_virtual(nm) else '(外接)'}", flush=True)
    n_total = len(names) if names else 3   # 枚舉成功就只掃有名字的,避免探不存在的 index 噴警告
    usable = []
    for idx in range(n_total):
        nm = names[idx] if idx < len(names) else f"Camera {idx}"
        if _is_virtual(nm):
            continue
        cap = _open(idx, width, height)
        ok = cap.isOpened() and sum(1 for _ in range(3) if cap.read()[0]) == 3
        cap.release(); time.sleep(0.2)
        if ok:
            usable.append((idx, nm, _is_internal(nm)))
    if not usable:
        return None
    external = [(i, nm) for i, nm, internal in usable if not internal]
    if external:
        return external[0]
    i, nm, _ = usable[0]
    return (i, nm)


def open_camera(source="auto", width=1280, height=720):
    """依 source 開鏡頭。回傳 (cap, index, name);找不到時回傳 (cap 未開啟, -1, "")。"""
    if source is None or str(source).lower() == "auto":
        found = detect_camera(width, height)
        if found is None:
            print("[鏡頭] 找不到任何可用鏡頭", flush=True)
            return cv2.VideoCapture(), -1, ""
        idx, nm = found
        print(f"[鏡頭] 使用 index={idx}:{nm}", flush=True)
        return _open(idx, width, height), idx, nm
    try:
        idx = int(source)
    except (TypeError, ValueError):
        print(f"[鏡頭] --source 只能是 auto 或編號,收到 {source!r}", flush=True)
        return cv2.VideoCapture(), -1, ""
    names = _names(); nm = names[idx] if idx < len(names) else f"Camera {idx}"
    print(f"[鏡頭] 指定 index={idx}:{nm}", flush=True)
    return _open(idx, width, height), idx, nm
