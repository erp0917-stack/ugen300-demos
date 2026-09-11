"""離線單元測試:python test_traffic_logic.py(不需 UGen300)"""
import numpy as np
from tracker import CentroidTracker, LineCounter, iou
from ocr_engine import db_boxes, ctc_decode, normalize_plate


def test_iou():
    assert abs(iou((0, 0, 10, 10), (5, 5, 15, 15)) - 25 / 175) < 1e-6


def test_tracker_keeps_id_and_counts_once():
    tr = CentroidTracker(); lc = LineCounter(line_y=100)
    ids = set()
    for k in range(12):                    # 一台車由上往下穿過 y=100
        y = 40 + k * 12
        tracks = tr.update([("car", 0.9, (100, y, 160, y + 40))])
        ids.add(tracks[0].id); lc.update(tracks)
    assert len(ids) == 1, "同一台車 ID 應穩定"
    assert lc.counts["car"]["down"] == 1 and lc.total() == 1


def test_two_objects_different_ids_and_directions():
    tr = CentroidTracker(); lc = LineCounter(line_y=100)
    for k in range(12):
        dets = [("car", 0.9, (50, 40 + k * 12, 110, 80 + k * 12)), ("person", 0.9, (400, 160 - k * 12, 430, 220 - k * 12))]
        lc.update(tr.update(dets))
    assert lc.counts["car"]["down"] == 1 and lc.counts["person"]["up"] == 1


def test_track_expires():
    tr = CentroidTracker(max_missed=2)
    tr.update([("car", 0.9, (0, 0, 10, 10))])
    for _ in range(4): tr.update([])
    assert tr.tracks == {}


def test_db_boxes_finds_text_region():
    prob = np.zeros((100, 200), np.float32); prob[40:60, 50:150] = 0.9
    b = db_boxes(prob)
    assert len(b) == 1 and b[0][0] <= 50 and b[0][2] >= 149 and b[0][4] > 0.8


def test_ctc_decode():
    cs = ["<blank>", "A", "B", "1"]
    T = np.zeros((7, 4));
    for t, c in enumerate([1, 1, 0, 2, 0, 3, 3]): T[t, c] = 1.0
    assert ctc_decode(T, cs)[0] == "AB1"


def test_normalize_plate():
    assert normalize_plate("ABC-1234") == "ABC-1234"
    assert normalize_plate("abc1234") == "ABC-1234"
    assert normalize_plate("1234-AB") == "1234-AB"
    assert normalize_plate("AB0-123") == "AB-0123"   # O/0 混淆後仍能標準化
    assert normalize_plate("台北市") is None
    assert normalize_plate("") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("✅", name)
    print("全部通過")
