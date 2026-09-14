"""離線單元測試:python test_llm_engine.py(不需 UGen300)"""
from datetime import datetime

from llm_engine import ThinkSplitter, clean_answer, date_context, build_system_prompt, to_traditional, MODELS


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


def test_date_context_has_date_weekday_cutoff():
    s = date_context(datetime(2026, 9, 14, 9, 5), cutoff="2024 年底")
    assert "2026-09-14" in s and "星期一" in s and "09:05" in s and "2024 年底" in s


def test_build_system_prompt_per_kind_and_model():
    chat = build_system_prompt("chat", "Qwen3-1.7B", datetime(2026, 9, 14)); code = build_system_prompt("code", "Qwen2.5-Coder-1.5B", datetime(2026, 9, 14))
    assert "繁體中文" in chat and "程式碼" not in chat and "2024 年底" in chat
    assert "```python" in code and "2024 年中" in code
    assert all("hef" in c and "cutoff" in c and c["kind"] in ("chat", "code") for c in MODELS.values())


def test_to_traditional():
    out = to_traditional("软件代码")
    assert out in ("軟體程式碼", "软件代码")   # 沒裝 opencc 就原樣回傳
    keep = "演算法在台灣很常見,numpy.array() 與 Hailo-10H 不變,周杰倫也不變"
    assert to_traditional(keep) == keep
    assert to_traditional("这个算法的数据") in ("這個演算法的資料", "这个算法的数据")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("全部通過")
