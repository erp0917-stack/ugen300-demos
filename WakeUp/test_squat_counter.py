"""離線單元測試:python test_squat_counter.py(不需 UGen300)"""
import math
from squat_counter import SquatCounter, knee_angle


def _kps(knee_deg, conf=0.9):
    """合成 17 點:只放髖/膝/踝,讓兩邊膝角等於 knee_deg。"""
    kps = [(0.0, 0.0, 0.0)] * 17
    for hip, knee, ank, x in ((11, 13, 15, 100.0), (12, 14, 16, 140.0)):
        # 髖在膝正上方 100px;踝依角度擺放
        kps[knee] = (x, 300.0, conf)
        kps[hip] = (x, 200.0, conf)
        rad = math.radians(knee_deg)
        kps[ank] = (x + 100.0 * math.sin(rad), 300.0 - 100.0 * math.cos(rad), conf)
    return kps


def test_angle():
    assert abs(knee_angle(_kps(170)) - 170) < 1e-6
    assert abs(knee_angle(_kps(90)) - 90) < 1e-6
    assert knee_angle(_kps(90, conf=0.1)) is None


def test_one_rep_needs_confirm_frames():
    c = SquatCounter(target=3, confirm_frames=3)
    for _ in range(5): c.update(_kps(175))
    assert c.count == 0 and c.state == "stand"
    for _ in range(2): c.update(_kps(90))
    assert c.state == "stand", "兩幀不夠確認"
    c.update(_kps(90)); assert c.state == "down"
    for _ in range(3): r = c.update(_kps(175))
    assert c.count == 1 and r["just_counted"] and c.state == "stand"


def test_jitter_in_middle_zone_does_not_count():
    c = SquatCounter(target=3, confirm_frames=2)
    for a in (175, 175, 130, 140, 130, 175, 175):
        c.update(_kps(a))
    assert c.count == 0


def test_reaches_target_and_stops():
    c = SquatCounter(target=2, confirm_frames=1)
    for _ in range(4):
        c.update(_kps(175)); c.update(_kps(90))
    r = c.update(_kps(175))
    assert c.count == 2 and c.done and r["done"]


def test_invisible_keeps_state():
    c = SquatCounter(target=2, confirm_frames=1)
    c.update(_kps(175)); c.update(_kps(90))
    r = c.update(None)
    assert r["visible"] is False and c.state == "down"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("✅", name)
    print("全部通過")
