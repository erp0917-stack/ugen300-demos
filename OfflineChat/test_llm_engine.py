"""離線單元測試:python test_llm_engine.py(不需 UGen300)"""
from llm_engine import ThinkSplitter, clean_answer


def feed_all(pieces):
    sp = ThinkSplitter(); th = an = ""
    for p in pieces:
        t, a = sp.feed(p); th += t; an += a
    t, a = sp.flush(); th += t; an += a
    return th, an


def test_no_think():
    th, an = feed_all(["你好", ",我是", "助理。"])
    assert th == "" and an == "你好,我是助理。"


def test_think_then_answer():
    th, an = feed_all(["<think>", "先想", "一下", "</think>", "答案", "在此"])
    assert th == "先想一下" and an == "答案在此"


def test_tag_split_across_tokens():
    th, an = feed_all(["<th", "ink>思考", "中</th", "ink>回答"])
    assert th == "思考中" and an == "回答"


def test_partial_lt_in_answer_is_not_lost():
    th, an = feed_all(["a < b", " and c"])
    assert an == "a < b and c"


def test_clean_answer():
    assert clean_answer("好的<|im_end|>") == "好的"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("✅", name)
    print("全部通過")
