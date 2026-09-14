"""離線測試(不需要 UGen300):Markdown 程式碼區塊切分、串流未閉合、沙箱執行。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeblocks as C  # noqa: E402


def test_split_text_and_code():
    t = "說明一下\n```python\nprint(1)\n```\n結尾"
    assert C.split_segments(t) == [("text", "說明一下\n"), ("code", "python", "print(1)"), ("text", "\n結尾")]


def test_lang_alias_and_empty():
    assert C.code_blocks("```py\nx=1\n```") == [("python", "x=1")]
    assert C.code_blocks("```\nplain\n```") == [("text", "plain")]


def test_unclosed_fence_streaming():
    assert C.code_blocks("開頭\n```python\nfor i in range(3):\n    pri") == [("python", "for i in range(3):\n    pri")]


def test_last_code_prefers_python():
    t = "```bash\npip install x\n```\n```python\nimport x\n```\n```bash\nls\n```"
    assert C.last_code(t) == ("python", "import x")
    assert C.last_code("沒有程式") == (None, None)


def test_strip_code_to_text():
    assert C.strip_code_to_text("A\n```python\nx\n```\nB") == "A\n[python 程式碼 → 右側]\nB"


def test_run_python_ok_error_timeout_blocked():
    r = C.run_python("print('hi'); print(1+1)"); assert r["ok"] and r["stdout"].strip() == "hi\n2".strip() or r["stdout"] == "hi\n2\n"
    r = C.run_python("1/0"); assert not r["ok"] and "ZeroDivisionError" in r["stderr"]
    r = C.run_python("while True: pass", timeout=1.0); assert r["timeout"] and not r["ok"]
    r = C.run_python("import shutil; shutil.rmtree('C:/')"); assert r["blocked"] and not r["ok"]


def test_unicode_output():
    r = C.run_python("print('中文 OK')"); assert r["ok"] and "中文 OK" in r["stdout"]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
