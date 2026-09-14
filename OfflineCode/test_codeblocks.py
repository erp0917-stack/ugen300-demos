"""離線測試(不需要 UGen300):Markdown 程式碼區塊切分、串流未閉合、AST 安全檢查、沙箱執行。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeblocks as C  # noqa: E402


def test_split_text_and_code():
    t = "說明一下\n```python\nprint(1)\n```\n結尾"
    assert C.split_segments(t) == [("text", "說明一下\n"), ("code", "python", "print(1)"), ("text", "\n結尾")]


def test_lang_alias_empty_and_crlf():
    assert C.code_blocks("```py\nx=1\n```") == [("python", "x=1")]
    assert C.code_blocks("```\nplain\n```") == [("text", "plain")]
    assert C.code_blocks("```python\r\nx=1\r\ny=2\r\n```") == [("python", "x=1\ny=2")]


def test_unclosed_fence_streaming():
    assert C.code_blocks("開頭\n```python\nfor i in range(3):\n    pri") == [("python", "for i in range(3):\n    pri")]
    assert C.strip_code_to_text("開頭\n```python\nx") == "開頭\n[python 程式碼 → 右側]"


def test_last_code_prefers_python_and_skips_empty():
    t = "```bash\npip install x\n```\n```python\nimport x\n```\n```bash\nls\n```"
    assert C.last_code(t) == ("python", "import x")
    assert C.last_code("```python\n\n```\n```bash\nls\n```") == ("bash", "ls")
    assert C.last_code("沒有程式") == (None, None)


def test_strip_code_to_text():
    assert C.strip_code_to_text("A\n```python\nx\n```\nB") == "A\n[python 程式碼 → 右側]\nB"


def test_guard_blocks_real_threats():
    bad = ["import os\nos.system('rd /s /q x')", "from pathlib import Path\nPath('x').unlink()", "open('C:/x.txt','w').write('x')",
           "import shutil\nshutil.rmtree('x')", "__import__('os').remove('x')", "exec('import os')", "import importlib\nimportlib.import_module('os')",
           "import socket", "import urllib.request", "import subprocess", "import sys\nsys.modules['os']", "().__class__.__subclasses__()",
           "import os as o\no.system('x')", "getattr(__builtins__, 'open')"]
    for code in bad:
        assert C.looks_dangerous(code), code


def test_guard_allows_normal_code():
    good = ["y = model / 2", "# 不要用 subprocess\nprint('subprocess is a module')", "rmdir_count = 0", "'{}'.format(1)",
            "import math, random\nprint(math.sqrt(2))", "from collections import Counter", "import numpy as np\nprint(np.arange(3))",
            "class A:\n    def __init__(self): pass\n    def __repr__(self): return 'A'\nif __name__ == '__main__': print(A())",
            "with open('in.txt') as f: pass", "import sys\nsys.setrecursionlimit(5000)", "x = [1,2]; x.remove(1); s = 'a'.replace('a','b')",
            "def f(:\n"]   # 語法錯誤交給沙箱報
    for code in good:
        assert C.looks_dangerous(code) is None, code


def test_run_python_ok_error_timeout_blocked():
    r = C.run_python("print('hi'); print(1+1)"); assert r["ok"] and r["stdout"] == "hi\n2\n"
    r = C.run_python("1/0"); assert not r["ok"] and "ZeroDivisionError" in r["stderr"]
    r = C.run_python("while True: pass", timeout=1.0); assert r["timeout"] and not r["ok"]
    r = C.run_python("import shutil; shutil.rmtree('C:/')"); assert r["blocked"] and not r["ok"] and "import shutil" in r["stderr"]


def test_input_does_not_wait_for_timeout():
    t = time.monotonic(); r = C.run_python("x = input()", timeout=5.0)
    assert not r["ok"] and "EOFError" in r["stderr"] and time.monotonic() - t < 3


def test_output_capped_and_unicode():
    r = C.run_python("print('中文 OK')"); assert r["ok"] and "中文 OK" in r["stdout"]
    r = C.run_python("for i in range(300000): print(i)"); assert r["ok"] and len(r["stdout"]) < C.MAX_OUT + 200 and "已截斷" in r["stdout"]


def test_numpy_available_in_sandbox():
    r = C.run_python("import numpy as np\nprint(np.arange(3).sum())")
    assert r["ok"] and r["stdout"].strip() == "3", r


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
