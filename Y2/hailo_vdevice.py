"""
hailo_vdevice.py —— 同一程序共用一個 VDevice(單例)+ HailoRT 物件保活 + 統一逾時。

2026-09-10:Windows 9 月更新後,HailoRT 5.3.2 在 USB 上呼叫 VDevice.release() 或任何 HailoRT 物件的
C++ 解構子(會送 hRPC shutdown + USB close)都會失敗(LIBUSB_ERROR_IO)並讓 UGen300 從系統消失,必須重新插拔。
對策:
  1. 同一程序只建立一個 VDevice(get()),各模型共用。
  2. 每個 HailoRT 物件(InferModel / ConfiguredInferModel / Bindings)建好就 keep() 起來,
     保證程序存活期間永不被回收 —— 即使模型類別的 __init__ 半途丟例外也不會觸發解構。
  3. 程序結束一律 exit_now()(os._exit)跳過 Python 收尾,由作業系統直接收回 USB handle。
  4. 前一個程序剛硬退出時,裝置端舊 session 還沒收掉,下一個程序的 create_infer_model 會撞 hRPC 逾時
     (預設 45 秒);把 HAILO_REQUEST_TIMEOUT_SECONDS 縮短並重試。

需要舊行為(結束時 release)時,設環境變數 HAILO_RELEASE_VDEVICE=1;HAILO_HARD_EXIT=0 可切回一般結束。
"""
import os
import time

# hRPC 單一請求逾時(秒)。必須在建立 VDevice 之前設定;configure() 大模型也走這個逾時,不要設太短。
os.environ.setdefault("HAILO_REQUEST_TIMEOUT_SECONDS", "20")
# 單次同步推論逾時(毫秒)。穩態 30~60 ms,但第一幀含排程器啟用、兩模型交錯時可能數百 ms;逾時=不可恢復,寧可寬。
RUN_TIMEOUT_MS = 10000

_VDEVICE = None
_KEEP = []
_TRANSIENT = ("OUT_OF_PHYSICAL_DEVICES", "not enough free devices", "HAILO_TIMEOUT", "Received a timeout")


def keep(*objs):
    """把 HailoRT 物件掛在模組層,讓它們活到 os._exit,永不觸發會讓裝置掉線的解構子。"""
    _KEEP.extend(objs)


def get(retries=6, delay=1.5):
    """回傳程序內唯一的 VDevice;第一次呼叫才建立。

    上一個程式剛結束時,UGen300 會重新列舉 2~4 秒,這段期間建立 VDevice 會失敗
    (HAILO_OUT_OF_PHYSICAL_DEVICES),所以只對這類暫時性錯誤重試;驅動沒裝、DLL 不合等錯誤直接丟出。
    """
    global _VDEVICE
    if _VDEVICE is None:
        from hailo_platform import VDevice
        last = None
        for i in range(retries):
            try:
                _VDEVICE = VDevice(); keep(_VDEVICE); break
            except Exception as e:  # noqa: BLE001
                last = e
                if not any(k in str(e) for k in _TRANSIENT):
                    raise
                print(f"[hailo_vdevice] 裝置尚未就緒({i + 1}/{retries}),{delay}s 後重試…", flush=True)
                time.sleep(delay)
        if _VDEVICE is None:
            raise last
    return _VDEVICE


def create_infer_model(hef_path, retries=3, delay=2.0):
    """在單例 VDevice 上建立 InferModel 並保活;前一個程序剛硬退出時可能撞 hRPC 逾時,重試幾次。"""
    vd = get(); last = None
    for i in range(retries):
        try:
            m = vd.create_infer_model(hef_path); keep(m); return m
        except Exception as e:  # noqa: BLE001
            last = e
            if not any(k in str(e) for k in _TRANSIENT) and type(e).__name__ != "HailoRTTimeout":
                raise
            print(f"[hailo_vdevice] create_infer_model 逾時({i + 1}/{retries}),{delay}s 後重試…", flush=True)
            time.sleep(delay)
    raise last


def is_timeout(e):
    """推論逾時 / 串流中止 = 舊 job 可能還在寫緩衝,不可重試,上層應直接 fatal。"""
    return type(e).__name__ in ("HailoRTTimeout", "HailoRTStreamAborted") or "timeout" in str(e).lower()


def release():
    """預設不做事;HAILO_RELEASE_VDEVICE=1 時才真的 release。"""
    global _VDEVICE
    if _VDEVICE is None:
        return
    if os.environ.get("HAILO_RELEASE_VDEVICE") != "1":
        return  # 保留單例給同一程序的下一階段使用
    try:
        _VDEVICE.release()
    except Exception:
        pass
    _VDEVICE = None


def exit_now(code=0):
    """程序結束時的最後一步。

    HAILO_HARD_EXIT=1 時用 os._exit 跳過 Python 收尾:HailoRT 的解構子不會送出那個在 Windows 2026-09
    更新後必定失敗的 USB close 訊息(LIBUSB_ERROR_IO),由作業系統直接收回 handle。
    2026-09-11 以 scripts/exit_stress.py 實測 8 輪:位址不變、全數存活 → 預設 1。設 HAILO_HARD_EXIT=0 可切回一般結束。
    """
    import sys
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    if os.environ.get("HAILO_HARD_EXIT", "1") == "1":
        os._exit(code)
    sys.exit(code)
