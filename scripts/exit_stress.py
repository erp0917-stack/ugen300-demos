"""
exit_stress.py —— 驗證「程序結束方式」對 UGen300 存活的影響。

用法(需插著 UGen300):
  python scripts/exit_stress.py --mode normal --n 8     # 一般結束(HailoRT 解構子會送 USB close)
  python scripts/exit_stress.py --mode hard   --n 8     # HAILO_HARD_EXIT=1:os._exit 跳過收尾

每一輪:子程序載入 yolov8s、跑一幀、結束 → 記錄結束後 USB 位址(hailortcli scan)是否改變、裝置是否還在。
判定:hard 模式若「位址不變且 8 輪都在」即可把 hailo_vdevice.exit_now 的預設切成 1。
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "..", "EdgeVsCloud")
CHILD = r'''
import numpy as np, hailo_vdevice
from hailo_detect import ObjectDetector
d = ObjectDetector("yolov8s.hef", conf_threshold=0.4)
d.infer(np.zeros((480, 640, 3), np.uint8))
print("child ok", flush=True)
hailo_vdevice.exit_now(0)
'''


def scan():
    out = subprocess.run(["hailortcli", "scan"], capture_output=True, text=True, timeout=30).stdout
    m = re.search(r"usb/(\d+:\d+)", out)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["normal", "hard"], default="hard"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    env = dict(os.environ, PYTHONUTF8="1", HAILO_HARD_EXIT="1" if a.mode == "hard" else "0")
    before = scan(); print(f"起始位址 {before}")
    if not before:
        print("裝置不在,請插上"); return 1
    changes = 0
    for i in range(1, a.n + 1):
        r = subprocess.run([sys.executable, "-c", CHILD], cwd=APP_DIR, env=env, capture_output=True, text=True, timeout=120)
        ok = "child ok" in r.stdout
        time.sleep(1.0)
        addr = scan()
        for _ in range(4):
            if addr: break
            time.sleep(2); addr = scan()
        print(f"第 {i} 輪:子程序 {'OK' if ok else 'FAIL'} | 結束後位址 {addr} {'(不變)' if addr == before else '(改變)' if addr else '(裝置消失!)'}")
        if not addr:
            print("=> 裝置掉線,測試中止"); return 2
        if addr != before: changes += 1; before = addr
    print(f"完成 {a.n} 輪,位址改變 {changes} 次,裝置存活。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
