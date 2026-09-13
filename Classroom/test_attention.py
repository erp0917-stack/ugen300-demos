"""離線測試(不需要 UGen300):姿態規則、持續時間、九宮格、手機歸屬。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attention as A  # noqa: E402


def person(nose=(100, 100), eyes=((90, 95), (110, 95)), ears=((80, 100), (120, 100)), sho=((60, 160), (140, 160)),
           wrists=((50, 260), (150, 260)), conf=0.9, hide=()):
    k = [(0, 0, 0.0)] * 17
    k[0] = (*nose, conf); k[1] = (*eyes[0], conf); k[2] = (*eyes[1], conf); k[3] = (*ears[0], conf); k[4] = (*ears[1], conf)
    k[5] = (*sho[0], conf); k[6] = (*sho[1], conf); k[9] = (*wrists[0], conf); k[10] = (*wrists[1], conf)
    k[11] = (70, 300, conf); k[12] = (130, 300, conf)
    for i in hide: k[i] = (k[i][0], k[i][1], 0.05)
    return k


def test_attentive_default():
    f = A.instant_flags(person()); assert not any(f.values()), f


def test_raise_hand():
    assert A.instant_flags(person(wrists=((50, 40), (150, 260))))["raise_"]
    assert A.instant_flags(person(wrists=((50, 40), (150, 260)), hide=(0,)))["raise_"]   # 沒鼻子改用肩線


def test_head_down_by_eyes_below_ears():
    assert A.instant_flags(person(eyes=((90, 110), (110, 110))))["head_down"]


def test_head_down_by_nose_near_shoulder():
    assert A.instant_flags(person(nose=(100, 145)))["head_down"]


def test_sleeping_nose_below_shoulder():
    f = A.instant_flags(person(nose=(100, 170))); assert f["sleeping"]


def test_sleeping_head_missing():
    assert A.instant_flags(person(hide=(0, 1, 2)))["sleeping"]


def test_turned_one_eye():
    assert A.instant_flags(person(hide=(2,)))["turned"]
    assert not A.instant_flags(person(hide=(1, 2)))["turned"]


def test_phone_in_box():
    box = (0, 0, 200, 300)
    assert A.phone_in_box([(90, 150, 110, 170)], box)
    assert not A.phone_in_box([(90, 10, 110, 30)], box)      # 太靠上(舉高)
    assert not A.phone_in_box([(400, 150, 420, 170)], box)


def test_hold_timing_and_priority():
    s = A.PersonState(); en = {k: True for k in A.RULES}
    k = person(wrists=((50, 40), (150, 260)), eyes=((90, 110), (110, 110)))
    assert s.update(k, [(90, 200, 110, 220)], en, now=0.0) == "ok" and not s.raised
    assert s.update(k, [(90, 200, 110, 220)], en, now=0.6) == "ok" and s.raised
    assert s.update(k, [(90, 200, 110, 220)], en, now=2.1) == "head_down"
    assert s.update(k, [(90, 200, 110, 220)], en, now=3.1) == "phone"     # 手機優先
    en["phone"] = False
    assert s.update(k, [(90, 200, 110, 220)], en, now=3.2) == "head_down"
    assert s.update(person(), [], en, now=3.3) == "ok" and not s.raised


def test_grid_and_summary():
    W, H = 1280, 720
    assert A.grid_cell((0, 0, 100, 100), W, H) == (0, 0)
    assert A.grid_cell((1200, 650, 1270, 700), W, H) == (2, 2)
    a = A.PersonState(); a.raised = True; a.box = (1200, 650, 1270, 700); a.state = "phone"
    b = A.PersonState(); b.state = "ok"
    out = A.summarize({1: a, 2: b}, W, H)
    assert out["raised"] == 1 and out["grid"][2][2] == 1 and out["inattentive"] == 1 and out["by"]["phone"] == 1


def test_bbox_from_kps():
    assert A.bbox_from_kps([(0, 0, 0.0)] * 17) is None
    x1, y1, x2, y2 = A.bbox_from_kps(person()); assert x1 < 50 and x2 > 150 and y1 < 95 and y2 > 300


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
