"""離線單元測試:python test_gestures.py(合成 21 點,不需 UGen300)"""
import math
from gestures import classify, finger_states, GestureSmoother


def synth(ext, thumb_open=False, thumb_dir="up", thumb_touch_index=False):
    """合成手掌:手腕 (100,200),手指朝上。ext: 四指是否伸直 [index, middle, ring, pinky]。"""
    pts = [(0.0, 0.0)] * 21
    pts[0] = (100.0, 200.0)
    xs = {"index": 80.0, "middle": 95.0, "ring": 110.0, "pinky": 125.0}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for f, e in zip(("index", "middle", "ring", "pinky"), ext):
        b = base[f]; x = xs[f]
        pts[b] = (x, 150.0)                       # MCP
        if e:
            pts[b + 1] = (x, 125.0); pts[b + 2] = (x, 105.0); pts[b + 3] = (x, 85.0)
        else:                                     # 彎曲:指尖折回手掌
            pts[b + 1] = (x, 140.0); pts[b + 2] = (x + 5, 155.0); pts[b + 3] = (x + 8, 165.0)
    # 拇指
    pts[1] = (75.0, 185.0); pts[2] = (62.0, 170.0)
    if thumb_touch_index:
        pts[3] = (70.0, 160.0); pts[4] = pts[8]   # 拇指尖碰食指尖
    elif thumb_open:
        pts[3] = (48.0, 155.0); pts[4] = (35.0, 140.0) if thumb_dir == "up" else (35.0, 200.0)
    else:
        pts[3] = (66.0, 172.0); pts[4] = (64.0, 182.0)   # 折在掌側,遠離食指尖
    return pts


def test_numbers():
    assert classify(synth([False, False, False, False]))[0] == "拳頭"
    assert classify(synth([True, False, False, False]))[0] == "1"
    assert classify(synth([True, True, False, False]))[0] == "2"
    assert classify(synth([True, True, True, False]))[0] == "3"
    assert classify(synth([True, True, True, True]))[0] == "4"
    assert classify(synth([True, True, True, True], thumb_open=True))[0] == "5"


def test_special():
    assert classify(synth([False, False, False, False], thumb_open=True, thumb_dir="up"))[0] == "讚"
    assert classify(synth([False, False, False, False], thumb_open=True, thumb_dir="down"))[0] == "倒讚"
    assert classify(synth([True, False, False, True]))[0] == "搖滾"
    assert classify(synth([False, True, True, True], thumb_touch_index=True))[0] == "OK"
    assert classify(synth([False, False, False, False], thumb_touch_index=True))[0] == "比心"
    assert classify(synth([False, False, False, True]))[0] == "小指"


def test_rotation_invariant():
    pts = synth([True, True, False, False])
    # 整隻手轉 90 度
    cx, cy = 100.0, 150.0
    rot = [((x - cx) * 0 - (y - cy) * 1 + cx, (x - cx) * 1 + (y - cy) * 0 + cy) for x, y in pts]
    assert classify(rot)[0] == "2"


def test_smoother():
    s = GestureSmoother(3)
    assert s.update("1") is None and s.update("1") is None and s.update("1") == "1"
    assert s.update("2") == "1"      # 尚未穩定,維持舊值
    s.update("2"); assert s.update("2") == "2"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("✅", name)
    print("全部通過")
