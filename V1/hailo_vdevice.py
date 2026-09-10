"""
hailo_vdevice.py —— 同一程序共用一個 VDevice(單例)。

2026-09-10:Windows 9 月更新後,HailoRT 5.3.2 在 USB 上呼叫 VDevice.release() 會失敗
(LIBUSB_ERROR_IO)並讓 UGen300 從系統消失,必須重新插拔。
實測:同一程序只建立一個 VDevice、各階段(VLM→LLM、Whisper→LLM)只釋放模型層物件、
程序結束時不呼叫 VDevice.release(),裝置可正常存活。

需要舊行為(結束時 release)時,設環境變數 HAILO_RELEASE_VDEVICE=1。
"""
import os

_VDEVICE = None


def get():
    """回傳程序內唯一的 VDevice;第一次呼叫才建立。"""
    global _VDEVICE
    if _VDEVICE is None:
        from hailo_platform import VDevice
        _VDEVICE = VDevice()
    return _VDEVICE


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
