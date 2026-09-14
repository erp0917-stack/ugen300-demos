# -*- coding: utf-8 -*-
"""
codeblocks.py —— 純邏輯:把 LLM 的 Markdown 回答切成「文字段 / 程式碼區塊」,支援串流中逐步解析;
另提供 Python 沙箱執行(子程序、逾時、暫存目錄)。不需要 UGen300,可離線測試。

    seg = split_segments(text)  → [("text", "說明…"), ("code", "python", "print(1)"), ...]
    code = last_code(text)      → (lang, code) 或 (None, None)
    r = run_python(code)        → dict(ok, stdout, stderr, elapsed, timeout)
"""
import os
import re
import subprocess
import sys
import tempfile
import time

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\r?\n")
LANG_ALIAS = {"py": "python", "python3": "python", "js": "javascript", "sh": "bash", "shell": "bash", "": "text"}


def split_segments(text):
    """回傳 [("text", s) | ("code", lang, s)];未閉合的 ``` 視為「還在串流中的程式碼」,照樣回傳。"""
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


DANGEROUS = ("shutil.rmtree", "os.remove", "os.rmdir", "os.unlink", "subprocess", "format(", "rmdir", "del /", "rm -rf", "winreg", "ctypes")


def looks_dangerous(code):
    """粗略擋掉會刪檔、改系統的片段(demo 用,不是安全邊界)。回傳命中的關鍵字或 None。"""
    low = code.lower()
    for k in DANGEROUS:
        if k in low and k != "format(": return k
    return None


def run_python(code, timeout=10.0, python=None):
    """在暫存目錄用子程序跑一段 Python,回傳 dict(ok, stdout, stderr, elapsed, timeout, blocked)。"""
    bad = looks_dangerous(code)
    if bad: return dict(ok=False, stdout="", stderr=f"為了安全,示範環境不執行含「{bad}」的程式。", elapsed=0.0, timeout=False, blocked=True)
    python = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="ugen_code_") as d:
        path = os.path.join(d, "snippet.py")
        with open(path, "w", encoding="utf-8") as f: f.write(code)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        t = time.time()
        try:
            p = subprocess.run([python, "-I", "-X", "utf8", path], cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return dict(ok=p.returncode == 0, stdout=p.stdout, stderr=p.stderr, elapsed=time.time() - t, timeout=False, blocked=False)
        except subprocess.TimeoutExpired as e:
            return dict(ok=False, stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                        stderr=f"執行超過 {timeout:.0f} 秒,已中止(可能是無窮迴圈或在等輸入)。", elapsed=time.time() - t, timeout=True, blocked=False)


def strip_code_to_text(text):
    """把回答中的程式碼區塊換成「[程式碼見右側]」,給聊天泡泡用。"""
    parts = []
    for s in split_segments(text):
        parts.append(s[1] if s[0] == "text" else f"[{s[1]} 程式碼 → 右側]")
    return "\n".join(p.strip("\n") for p in parts)
