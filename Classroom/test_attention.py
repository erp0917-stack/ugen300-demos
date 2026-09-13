"""離線測試(不需要 UGen300):姿態規則、持續時間、凍結補償、九宮格、手機歸屬。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attention as A  # noqa: E402



def person(nose=(100, 100), eyes=((90, 95), (110, 95)), ears=((80, 100), (120, 100)), sho=((60, 160), (140, 160)),
           elbows=((45, 220), (155, 220)), wrists=((50, 260), (150, 260)), conf=0.9, hide=(), dy=0):
    """肩寬 80。dy 把整個人往下移(讓肩膀落在畫面下半)。"""
    k = [(0, 0, 0.0)] * 17
    k[0] = (*nose, conf); k[1] = (*eyes[0], conf); k[2] = (*eyes[1], conf); k[3] = (*ears[0], conf); k[4] = (*ears[1], conf)
    k[5] = (*sho[0], conf); k[6] = (*sho[1], conf); k[7] = (*elbows[0], conf); k[8] = (*elbows[1], conf)
    k[9] = (*wrists[0], conf); k[10] = (*wrists[1], conf); k[11] = (70, 300, conf); k[12] = (130, 300, conf)
    for i in hide: k[i] = (k[i][0], k[i][1], 0.05)
    return [(x, y + dy, c) for x, y, c in k]


def test_attentive_default():
    f = A.instant_flags(person()); assert not any(f.values()), f


def test_laptop_viewer_is_not_head_down():
    """筆電鏡頭俯視、頭低 10° 看螢幕:眼睛比耳朵低約 0.08 肩寬、鼻子距肩線 0.27 肩寬 → 不算低頭。"""
    f = A.instant_flags(person(eyes=((90, 106), (110, 106)), nose=(100, 138)))
    assert not f["head_down"] and not f["sleeping"], f


def test_real_head_down():
    assert A.instant_flags(person(eyes=((90, 112), (110, 112))))["head_down"]        # 眼低於耳 0.15 肩寬
    assert A.instant_flags(person(nose=(100, 148)))["head_down"]                     # 鼻距肩線 0.15 肩寬


def test_raise_hand_needs_elbow_up_and_wrist_well_above_eyes():
    assert A.instant_flags(person(elbows=((45, 120), (155, 220)), wrists=((50, 20), (150, 260))))["raise_"]   # 眼線 95,手腕 20:高 75 > 0.6*80
    assert not A.instant_flags(person(wrists=((50, 90), (150, 260))))["raise_"]      # 摸頭:手肘還在肩下
    assert not A.instant_flags(person(elbows=((45, 120), (155, 220)), wrists=((50, 60), (150, 260))))["raise_"]   # 撥瀏海:手腕只比眼線高 35
    assert A.instant_flags(person(elbows=((45, 60), (155, 220)), hide=(9,)))["raise_"]    # 手腕出框、手肘高於眼線
    assert not A.instant_flags(person(elbows=((45, 120), (155, 220)), hide=(9,)))["raise_"]   # 手腕出框但手肘只到肩上


def test_sleeping_nose_below_shoulder():
    assert A.instant_flags(person(nose=(100, 170)))["sleeping"]


def test_head_missing_is_never_sleeping():
    for k in (person(hide=(0, 1, 2)), person(hide=(0, 1, 2), dy=400), person(hide=(0, 1, 2, 3, 4))):   # 出框、背對都不判
        f = A.instant_flags(k); assert not f["sleeping"] and not f["head_down"] and not f["turned"]


def test_far_small_person_skips_face_rules():
    small = person(nose=(100, 100), eyes=((97, 97), (103, 97)), ears=((95, 100), (105, 100)), sho=((85, 120), (115, 120)), hide=(2,))
    f = A.instant_flags(small); assert not f["turned"] and not f["head_down"]


def test_turned_one_eye():
    assert A.instant_flags(person(hide=(2,)))["turned"]
    k = person(); k[2] = (k[2][0], k[2][1], 0.35); assert not A.instant_flags(k)["turned"]   # 遠眼 0.35 不算
    assert not A.instant_flags(person(hide=(1, 2)))["turned"]


def test_assign_phones():
    persons = [(1, (0, 0, 200, 300), 160, 80, [(100, 210)]), (2, (150, 0, 400, 300), 160, 80, [(240, 210)])]
    out = A.assign_phones([(90, 200, 110, 220), (230, 200, 250, 220), (90, 50, 110, 70), (500, 200, 520, 220)], persons)
    assert out == {1: [(90, 200, 110, 220)], 2: [(230, 200, 250, 220)]}      # 重疊區歸最近者;肩線以上與框外不算
    desk = [(1, (0, 0, 400, 600), 160, 80, [(100, 250)])]                    # 手腕在 (100,250),手機在桌角 (350,550):距離 > 1.2 肩寬 → 不算
    assert A.assign_phones([(340, 540, 360, 560)], desk) == {}
    no_wrist = [(1, (0, 0, 400, 600), 160, 80, [])]                          # 手腕看不到 → 退回框規則
    assert A.assign_phones([(340, 540, 360, 560)], no_wrist) == {1: [(340, 540, 360, 560)]}
    assert A.wrists_of(person()) == [(50, 260), (150, 260)] and A.wrists_of(person(hide=(9, 10))) == []


def test_hold_timing_priority_and_freeze_shift():
    s = A.PersonState(); en = {k: True for k in A.RULES}
    k = person(elbows=((45, 120), (155, 220)), wrists=((50, 20), (150, 260)), eyes=((90, 112), (110, 112)))
    ph = [(90, 200, 110, 220)]
    assert s.update(k, ph, en, now=0.0) == "ok" and not s.raised
    assert s.update(k, ph, en, now=1.3) == "ok" and s.raised
    assert s.update(k, ph, en, now=2.1) == "head_down"
    s.shift(10.0)                                                   # 凍結 10 秒
    assert s.update(k, ph, en, now=12.5) == "head_down"   # 手機 2.5 秒有效時間,還不到 3
    assert s.update(k, ph, en, now=13.1) == "phone"
    en["phone"] = False
    assert s.update(k, ph, en, now=13.2) == "head_down"
    assert s.update(person(), [], en, now=13.3) == "ok" and not s.raised


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
    hx1, hy1, hx2, hy2 = A.bbox_from_kps(person(), idx=A.HEAD_SHOULDER); assert hy2 < 200   # 頭肩框不含手


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
