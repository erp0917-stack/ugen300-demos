"""離線測試(不需要 UGen300):FaceDB 存取與比對、NMS、對齊。"""
import os
import sys
import tempfile

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
    db.add("A", unit(1)); db.add("A", unit(2)); assert len(db.people) == 1


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
