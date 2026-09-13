"""離線測試(不需要 UGen300):FaceDB 存取與比對、NMS、對齊。"""
import os
import sys
import tempfile
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_engine as fe  # noqa: E402


def unit(seed):
    v = np.random.RandomState(seed).randn(512).astype(np.float32); return v / np.linalg.norm(v)


def test_db_roundtrip_and_match():
    d = os.path.join(tempfile.mkdtemp(), "faces.json")
    db = fe.FaceDB(d); a, b = unit(1), unit(2)
    db.add("甲", a, np.zeros((80, 80, 3), np.uint8)); db.add("乙", b)
    db2 = fe.FaceDB(d)
    assert db2.names() == ["甲", "乙"] and db2.people[0]["thumb"] is not None and db2.people[1]["thumb"] is None
    name, sim = db2.match(a * 0.9 + unit(3) * 0.1); assert name == "甲" and sim > 0.8
    name, sim = db2.match(unit(9)); assert name is None and sim < 0.45
    assert db2.remove_last() == "乙" and fe.FaceDB(d).names() == ["甲"]


def test_add_same_name_replaces():
    db = fe.FaceDB(os.path.join(tempfile.mkdtemp(), "f.json"))
    assert db.add("A", unit(1)) == (False, True); assert db.add("A", unit(2)) == (True, True); assert len(db.people) == 1


def test_match_margin_rejects_ambiguous():
    db = fe.FaceDB(os.path.join(tempfile.mkdtemp(), "f.json")); a = unit(1)
    db.add("甲", a); db.add("乙", a * 0.98 + unit(5) * 0.02)
    assert db.match(a)[0] is None                      # 兩人幾乎一樣 → 分不清,不報到
    db.add("乙", unit(6)); assert db.match(a)[0] == "甲"


def test_dict_db_and_bad_vec_are_tolerated():
    p = os.path.join(tempfile.mkdtemp(), "f.json"); open(p, "w").write('{"name": "x"}')
    db = fe.FaceDB(p); assert db.people == [] and db.last_error
    open(p, "w", encoding="utf-8").write('[{"name": "短", "vec": [1, 2]}]')
    assert fe.FaceDB(p).people == []


def test_save_failure_returns_false():
    d = tempfile.mkdtemp(); db = fe.FaceDB(os.path.join(d, "f.json"))
    os.makedirs(os.path.join(d, "f.json"))           # 同名資料夾擋住寫入
    existed, saved = db.add("A", unit(1)); assert not saved and db.last_error


def test_square_crop_is_square():
    img = np.zeros((300, 400, 3), np.uint8); c = fe.square_crop(img, (100, 50, 160, 140))
    assert abs(c.shape[0] - c.shape[1]) <= 1


def test_template_matches_insightface_arcface_src():
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
    assert np.allclose(fe._ARCFACE_TEMPLATE, src)
    assert fe.FaceEmbedder.align(np.zeros((300, 300, 3), np.uint8), src * 2 + 50).shape == (112, 112, 3)


def test_square_crop_at_border_is_square():
    img = np.zeros((300, 400, 3), np.uint8); c = fe.square_crop(img, (0, 0, 60, 90))
    assert c.shape[0] == c.shape[1]


def test_nan_vector_rejected():
    db = fe.FaceDB(os.path.join(tempfile.mkdtemp(), "f.json")); v = unit(1); v[0] = np.nan
    assert db.add("N", v) == (False, False) and db.people == []
    db.add("A", unit(2)); assert db.match(v)[0] is None


def test_partial_bad_db_keeps_good_rows():
    p = os.path.join(tempfile.mkdtemp(), "f.json")
    good = [round(float(x), 5) for x in unit(1)]
    open(p, "w", encoding="utf-8").write(json.dumps([{"name": "A", "vec": good}, {"name": "B", "vec": "oops"}, {"name": "C", "vec": good}]))
    db = fe.FaceDB(p); assert db.names() == ["A", "C"] and db.last_error


def test_align_degenerate_returns_none():
    assert fe.FaceEmbedder.align(np.zeros((100, 100, 3), np.uint8), np.zeros((5, 2), np.float32)) is None


def test_corrupt_db_is_empty():
    p = os.path.join(tempfile.mkdtemp(), "f.json"); open(p, "w").write("{bad")
    assert fe.FaceDB(p).people == []


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32); sc = np.array([0.9, 0.8, 0.7])
    assert sorted(fe._nms(boxes, sc, 0.4)) == [0, 2]


def test_align_identity_template():
    img = np.zeros((200, 200, 3), np.uint8)
    out = fe.FaceEmbedder.align(img, fe._ARCFACE_TEMPLATE + 40)
    assert out.shape == (112, 112, 3)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
