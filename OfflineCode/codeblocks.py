# -*- coding: utf-8 -*-
"""
codeblocks.py —— 純邏輯:把 LLM 的 Markdown 回答切成「文字段 / 程式碼區塊」,支援串流中逐步解析;
另提供 Python 沙箱執行(AST 白名單檢查 → 子程序 → 逾時殺整棵程序樹 → 輸出截斷)。不需要 UGen300,可離線測試。

    seg = split_segments(text)  → [("text", "說明…"), ("code", "python", "print(1)"), ...]
    code = last_code(text)      → (lang, code) 或 (None, None)
    why = looks_dangerous(code) → None(可執行)或一句給觀眾看的原因
    r = run_python(code)        → dict(ok, stdout, stderr, elapsed, timeout, blocked)

安全邊界是「demo 等級」:擋掉 1.5B 模型被慫恿寫出的刪檔、改系統、外連、動態 import;不是對抗惡意攻擊者的沙箱。
"""
import ast
import os
import re
import subprocess
import sys
import tempfile
import time

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\n")
LANG_ALIAS = {"py": "python", "python3": "python", "js": "javascript", "sh": "bash", "shell": "bash", "": "text"}
MAX_OUT = 200_000          # 輸出字元上限(print 一百萬行不會把 UI 凍住)


def split_segments(text):
    """回傳 [("text", s) | ("code", lang, s)];未閉合的 ``` 視為「還在串流中的程式碼」,照樣回傳。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []; pos = 0
    while True:
        m = _FENCE.search(text, pos)
        if not m:
            rest = text[pos:]
            if rest.strip(): out.append(("text", rest))
            break
        if m.start() > pos and text[pos:m.start()].strip(): out.append(("text", text[pos:m.start()]))
        lang = LANG_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        end = text.find("```", m.end())
        if end < 0:
            out.append(("code", lang, text[m.end():])); break
        out.append(("code", lang, text[m.end():end].rstrip("\n")))
        pos = end + 3
    return out


def code_blocks(text):
    return [(s[1], s[2]) for s in split_segments(text) if s[0] == "code"]


def last_code(text, prefer=("python",)):
    """最後一個程式碼區塊(優先 python);沒有回傳 (None, None)。"""
    blocks = code_blocks(text)
    if not blocks: return None, None
    for lang in prefer:
        for l, c in reversed(blocks):
            if l == lang and c.strip(): return l, c
    return blocks[-1]


# ---------- 沙箱前檢查(AST) ----------
ALLOWED_MODULES = {
    "math", "cmath", "random", "statistics", "decimal", "fractions", "itertools", "functools", "operator",
    "collections", "heapq", "bisect", "string", "re", "json", "datetime", "time", "calendar", "typing",
    "dataclasses", "enum", "abc", "copy", "textwrap", "unicodedata", "array", "numbers", "pprint", "numpy",
    "csv", "io", "sys", "hashlib", "logging", "argparse", "asyncio", "threading", "queue", "sqlite3", "base64", "struct", "secrets", "uuid", "__future__",
}
FILE_CALLS = {"connect", "FileHandler", "basicConfig"}   # sqlite3 / logging 這些不經 open() 的寫檔入口,同樣只允許純檔名
_ = {
}
BANNED_CALLS = {"exec", "eval", "compile", "__import__", "breakpoint", "globals", "vars", "getattr", "setattr", "delattr", "open_code"}
BANNED_ATTRS = {"modules", "meta_path", "path_hooks", "settrace", "setprofile", "_getframe"}   # sys.modules['os'] 之類的繞過口
SAFE_DUNDERS = {"__name__", "__main__", "__init__", "__repr__", "__str__", "__eq__", "__lt__", "__le__", "__gt__", "__ge__",
                "__len__", "__iter__", "__next__", "__getitem__", "__setitem__", "__contains__", "__call__", "__enter__",
                "__exit__", "__hash__", "__add__", "__sub__", "__mul__", "__truediv__", "__doc__", "__post_init__", "__bool__",
                "__neg__", "__abs__", "__slots__", "__dict__", "__annotations__", "__ne__", "__radd__", "__mod__", "__floordiv__",
                "__pow__", "__version__", "__file__", "__all__", "__iadd__", "__isub__", "__reversed__", "__index__", "__int__", "__float__"}


def _mode_of(call):
    if len(call.args) > 1: return call.args[1]
    for kw in call.keywords:
        if kw.arg == "mode": return kw.value
    return None


def looks_dangerous(code):
    """回傳 None(可送進沙箱)或一句給觀眾看的原因。語法錯誤不擋,讓沙箱自己報 SyntaxError。
    規則:import 白名單、危險內建呼叫黑名單、危險屬性黑名單、非常見雙底線名稱、open() 寫入模式。"""
    try: tree = ast.parse(code)
    except SyntaxError: return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES: return f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES: return f"from {node.module} import …"
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name in BANNED_CALLS: return f"{name}()"
            if name in FILE_CALLS:
                for arg in list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == "filename"]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value != ":memory:" and (any(c in arg.value for c in ":/\\") or ".." in arg.value):
                        return f"{name}(…) 指向暫存目錄以外的路徑"
            if name == "open":
                m = _mode_of(node)
                writing = m is not None and (not isinstance(m, ast.Constant) or any(c in str(m.value) for c in "wax+"))
                if writing:
                    # 寫檔只允許「純檔名的常數」(落在沙箱暫存目錄);含路徑分隔、磁碟機、.. 或非常數的一律擋
                    target = node.args[0] if node.args else None
                    plain = isinstance(target, ast.Constant) and isinstance(target.value, str) and target.value and not any(c in target.value for c in ":/\\") and ".." not in target.value
                    if not plain: return "open(…, 寫入模式)"
        elif isinstance(node, (ast.Attribute, ast.Name)):
            ident = node.attr if isinstance(node, ast.Attribute) else node.id
            if ident in BANNED_ATTRS: return ident
            if ident.startswith("__") and ident.endswith("__") and ident not in SAFE_DUNDERS: return ident
    return None


# ---------- 沙箱執行 ----------
def _cap(s):
    if s is None: return ""
    if len(s) <= MAX_OUT: return s
    return s[:MAX_OUT] + f"\n… (輸出過長,已截斷,共 {len(s):,} 字元)"


def _kill_tree(p):
    """逾時時殺整棵程序樹(Windows 上孫程序會繼承 stdout 管線,只殺子程序會讓 communicate 卡住)。"""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        p.kill()
    try: p.wait(timeout=2)
    except subprocess.TimeoutExpired: pass


def run_python(code, timeout=10.0, python=None):
    """在暫存目錄用子程序跑一段 Python,回傳 dict(ok, stdout, stderr, elapsed, timeout, blocked)。
    -I(隔離模式,仍可用系統 site-packages 如 numpy)-X utf8;stdin 關閉,input() 會立刻 EOFError 而不是等逾時。"""
    bad = looks_dangerous(code)
    if bad: return dict(ok=False, stdout="", stderr=f"為了安全,示範環境不執行含「{bad}」的程式。", elapsed=0.0, timeout=False, blocked=True)
    python = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="ugen_code_") as d:
        path = os.path.join(d, "snippet.py")
        with open(path, "w", encoding="utf-8") as f: f.write(code)
        env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")}
        env["TEMP"] = env["TMP"] = d
        t = time.monotonic()
        p = subprocess.Popen([python, "-I", "-X", "utf8", path], cwd=d, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            out, err = p.communicate(timeout=timeout)
            return dict(ok=p.returncode == 0, stdout=_cap(out), stderr=_cap(err), elapsed=time.monotonic() - t, timeout=False, blocked=False)
        except subprocess.TimeoutExpired:
            _kill_tree(p)
            try: out, err = p.communicate(timeout=2)
            except subprocess.TimeoutExpired: out, err = "", ""
            return dict(ok=False, stdout=_cap(out), stderr=f"執行超過 {timeout:.0f} 秒,已中止(可能是無窮迴圈或算太久)。",
                        elapsed=time.monotonic() - t, timeout=True, blocked=False)


_TRAILING_FENCE = re.compile(r"\n?```[A-Za-z0-9_+#.-]*[ \t]*$")


def strip_code_to_text(text):
    """把回答中的程式碼區塊換成「[程式碼見右側]」,給聊天泡泡用。串流中尾端還沒換行的 ``` 先隱藏,不會閃出「```python」。"""
    parts = []
    for s in split_segments(text):
        parts.append(s[1] if s[0] == "text" else f"[{s[1]} 程式碼 → 右側]")
    out = "\n".join(p.strip("\n") for p in parts)
    return _TRAILING_FENCE.sub("", out)
