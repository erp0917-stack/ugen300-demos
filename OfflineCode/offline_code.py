"""
offline_code.py —— 離線 Code:Claude Code 風格的本機程式助理。左邊對話、右邊程式碼與執行結果。

模型:Qwen2.5-Coder-1.5B(UGen300 本機,約 8 token/秒)。回答裡的 ```python 區塊會即時抽到右側編輯器,
一鍵「執行」在沙箱子程序跑(10 秒逾時、暫存目錄、擋刪檔指令),結果顯示在下方主控台。全程不需要網路。

用法:python offline_code.py [--model Qwen2.5-Coder-1.5B]
      python offline_code.py --selftest                 (載模型、出一題、抽程式碼、執行、印結果、離開)
      python offline_code.py --snapshot out.png         (自動出題 → 執行 → 截圖 → 離開)
"""
import argparse
import os
import socket
import sys
import threading
import traceback
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat, QSyntaxHighlighter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QLineEdit, QScrollArea, QSizePolicy, QPlainTextEdit, QSplitter, QFileDialog, QMessageBox)

import hailo_vdevice
import codeblocks as CB
from llm_engine import LLMEngine, MODELS, DEFAULT_CODE_MODEL, to_traditional, date_context

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = [("質數函式", "寫一個 Python 函式 is_prime(n) 判斷質數,並印出 1 到 30 之間的質數。"),
         ("讀 CSV 算平均", "用 Python 標準函式庫寫一段程式:建立一個含 5 筆成績的 CSV 字串,讀取後算平均並印出。"),
         ("FizzBuzz", "用 Python 寫 FizzBuzz,印出 1 到 20。"),
         ("解釋錯誤", "這個錯誤是什麼意思、怎麼修:TypeError: can only concatenate str (not \"int\") to str"),
         ("排序物件", "用 Python 寫一段程式:有一個學生 list(name, score),依分數由高到低排序後印出前三名。")]
STYLE = """
QMainWindow, QWidget { background: #1b1b1f; color: #ececec; font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "Noto Sans TC"; }
QLabel#title { font-size: 22px; font-weight: 700; }
QLabel#sub { color: #9aa0a6; font-size: 13px; }
QLabel#status { color: #9aa0a6; font-size: 14px; }
QLabel#offline { color: #37d67a; font-size: 14px; font-weight: 700; border: 1px solid #37d67a; border-radius: 6px; padding: 3px 8px; }
QLabel#online { color: #ff6b6b; font-size: 14px; font-weight: 700; border: 1px solid #ff6b6b; border-radius: 6px; padding: 3px 8px; }
QLabel#pane { color: #c9c9d1; font-size: 15px; font-weight: 700; }
QLineEdit, QPushButton { background: #2a2a30; border: 1px solid #3a3a42; border-radius: 8px; padding: 8px 12px; font-size: 16px; color: #ececec; }
QLineEdit { font-size: 17px; }
QPushButton#send { background: #2f6fed; border: none; font-weight: 700; }
QPushButton#run { background: #1f8f4e; border: none; font-weight: 700; }
QPushButton:disabled { background: #3a3a42; color: #888; }
QPushButton.quick { font-size: 13px; padding: 5px 10px; border-radius: 12px; }
QScrollArea { border: none; }
QLabel.bubble { font-size: 17px; line-height: 1.5; padding: 12px 16px; border-radius: 14px; }
QLabel#user { background: #2f6fed; color: white; }
QLabel#bot { background: #2a2a30; }
QPlainTextEdit { background: #121216; color: #e6e6e6; border: 1px solid #3a3a42; border-radius: 8px; font-family: Consolas, "Cascadia Mono", monospace; font-size: 15px; padding: 8px; }
QPlainTextEdit#console { background: #0e0e11; font-size: 14px; }
"""


class PyHighlighter(QSyntaxHighlighter):
    """夠用的 Python 上色:關鍵字、字串、註解、數字。"""
    KW = ("def class return if elif else for while in not and or import from as try except finally with lambda yield "
          "pass break continue True False None print range len is global nonlocal raise assert del").split()

    def __init__(self, doc):
        super().__init__(doc)
        def fmt(color, bold=False):
            f = QTextCharFormat(); f.setForeground(QColor(color)); f.setFontWeight(700 if bold else 400); return f
        import re
        self.rules = [(re.compile(r"\b(%s)\b" % "|".join(self.KW)), fmt("#7aa2f7", True)),
                      (re.compile(r"\b\d+(\.\d+)?\b"), fmt("#ff9e64")),
                      (re.compile(r"(\"[^\"\n]*\"|'[^'\n]*')"), fmt("#9ece6a")),
                      (re.compile(r"#[^\n]*"), fmt("#6b6b78"))]

    def highlightBlock(self, text):
        for rx, f in self.rules:
            for m in rx.finditer(text): self.setFormat(m.start(), m.end() - m.start(), f)


class LoadWorker(QThread):
    status = Signal(str); done = Signal(bool, str)
    def __init__(self, engine, model): super().__init__(); self.engine, self.model = engine, model
    def run(self):
        try: self.engine.load(self.model, on_status=self.status.emit); self.done.emit(True, "")
        except Exception as e: self.done.emit(False, f"{type(e).__name__}: {e}")  # noqa: BLE001


class GenWorker(QThread):
    piece = Signal(str); finished_ok = Signal(float, float, int); failed = Signal(str)
    def __init__(self, engine, text): super().__init__(); self.engine, self.text = engine, text
    def run(self):
        try:
            for kind, s in self.engine.stream(self.text, max_tokens=600):
                if kind == "answer": self.piece.emit(s)
            r = self.engine.rate; self.finished_ok.emit(r.tps, r.ttft, r.n)
        except Exception as e: self.failed.emit(f"{type(e).__name__}: {e}")  # noqa: BLE001


class RunWorker(QThread):
    done = Signal(dict)
    def __init__(self, code): super().__init__(); self.code = code
    def run(self): self.done.emit(CB.run_python(self.code))


def _net_online(timeout=0.4):
    try: socket.create_connection(("1.1.1.1", 53), timeout=timeout).close(); return True
    except OSError: return False


class Bubble(QLabel):
    def __init__(self, text, who):
        super().__init__(text)
        self.setObjectName(who); self.setProperty("class", "bubble")
        self.setWordWrap(True); self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setMaximumWidth(560); self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        f = QFont(); f.setPointSize(12); self.setFont(f)


class CodeApp(QMainWindow):
    net_sig = Signal(bool)

    def __init__(self, model, snapshot=""):
        super().__init__()
        self.net_sig.connect(self._set_net)
        self.engine = LLMEngine(kind="code"); self.model = model; self.snapshot = snapshot
        self.setWindowTitle("UGen300 離線 Code"); self.resize(1400, 820)
        root = QWidget(); self.setCentralWidget(root); v = QVBoxLayout(root); v.setContentsMargins(20, 14, 20, 14); v.setSpacing(8)

        top = QHBoxLayout()
        t = QLabel("離線 Code"); t.setObjectName("title"); top.addWidget(t)
        sub = QLabel("Claude Code 風格 · Qwen2.5-Coder 1.5B 跑在 UGen300 · 程式碼不出這台電腦"); sub.setObjectName("sub"); top.addWidget(sub)
        top.addStretch(1)
        self.date = QLabel(f"今天 {datetime.now():%Y-%m-%d}"); self.date.setObjectName("status"); top.addWidget(self.date)
        self.net = QLabel("網路檢查中…"); self.net.setObjectName("status"); top.addWidget(self.net)
        v.addLayout(top)
        self.status = QLabel("載入模型中…"); self.status.setObjectName("status"); v.addWidget(self.status)

        split = QSplitter(Qt.Horizontal); v.addWidget(split, 1)
        # 左:對話
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 8, 0); lv.setSpacing(8)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.feed = QWidget(); self.feed_l = QVBoxLayout(self.feed); self.feed_l.setSpacing(10); self.feed_l.addStretch(1)
        self.scroll.setWidget(self.feed); lv.addWidget(self.scroll, 1)
        quick = QHBoxLayout()
        for label, prompt in QUICK:
            b = QPushButton(label); b.setProperty("class", "quick"); b.clicked.connect(lambda _=False, p=prompt: self._ask(p)); quick.addWidget(b)
        quick.addStretch(1); lv.addLayout(quick)
        bottom = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("描述你要的程式,Enter 送出(模型載入中…)"); self.input.setEnabled(False)
        self.input.returnPressed.connect(self._send); bottom.addWidget(self.input, 1)
        self.send_btn = QPushButton("送出"); self.send_btn.setObjectName("send"); self.send_btn.setEnabled(False); self.send_btn.clicked.connect(self._send); bottom.addWidget(self.send_btn)
        self.clear_btn = QPushButton("清除"); self.clear_btn.clicked.connect(self._clear); bottom.addWidget(self.clear_btn)
        lv.addLayout(bottom)
        split.addWidget(left)
        # 右:程式碼 + 主控台
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(8, 0, 0, 0); rv.setSpacing(8)
        rh = QHBoxLayout(); self.pane = QLabel("程式碼(可直接修改)"); self.pane.setObjectName("pane"); rh.addWidget(self.pane); rh.addStretch(1)
        self.run_btn = QPushButton("▶ 執行"); self.run_btn.setObjectName("run"); self.run_btn.clicked.connect(self._run); self.run_btn.setEnabled(False); rh.addWidget(self.run_btn)
        self.copy_btn = QPushButton("複製"); self.copy_btn.clicked.connect(self._copy); rh.addWidget(self.copy_btn)
        self.save_btn = QPushButton("存檔"); self.save_btn.clicked.connect(self._save); rh.addWidget(self.save_btn)
        rv.addLayout(rh)
        self.code = QPlainTextEdit(); self.code.setPlaceholderText("模型回答裡的程式碼會出現在這裡…"); self.hl = PyHighlighter(self.code.document())
        self.code.textChanged.connect(lambda: self.run_btn.setEnabled(bool(self.code.toPlainText().strip()) and self.lang in ("python", "text")))
        rv.addWidget(self.code, 3)
        ch = QHBoxLayout(); cl = QLabel("執行結果"); cl.setObjectName("pane"); ch.addWidget(cl); ch.addStretch(1)
        self.run_info = QLabel(""); self.run_info.setObjectName("status"); ch.addWidget(self.run_info); rv.addLayout(ch)
        self.console = QPlainTextEdit(); self.console.setObjectName("console"); self.console.setReadOnly(True); rv.addWidget(self.console, 2)
        split.addWidget(right); split.setSizes([620, 760])

        self.cur_bot = None; self.gen = None; self.full = ""; self.lang = "python"; self.runner = None
        self._add_bubble("你好!我是跑在 UGen300 上的本機程式助理。描述你要的程式,我寫在右邊,按「執行」就能跑。", "bot")
        self.loader = LoadWorker(self.engine, model); self.loader.status.connect(self.status.setText); self.loader.done.connect(self._loaded); self.loader.start()
        self.net_timer = QTimer(self); self.net_timer.timeout.connect(self._check_net); self.net_timer.start(4000); self._check_net()

    # ----- 網路徽章 -----
    def _check_net(self):
        threading.Thread(target=lambda: self.net_sig.emit(_net_online()), daemon=True).start()

    def _set_net(self, on):
        self.net.setText("網路:連線中" if on else "網路:已斷線 ✓ 仍可運作"); self.net.setObjectName("online" if on else "offline")
        self.net.style().unpolish(self.net); self.net.style().polish(self.net)

    # ----- 對話 -----
    def _add_bubble(self, text, who):
        row = QHBoxLayout(); b = Bubble(text, who)
        if who == "user": row.addStretch(1); row.addWidget(b)
        else: row.addWidget(b); row.addStretch(1)
        w = QWidget(); w.setLayout(row); self.feed_l.insertWidget(self.feed_l.count() - 1, w)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))
        return b

    def _loaded(self, ok, err):
        if not ok:
            self.status.setText("模型載入失敗:" + err); self._add_bubble("模型載入失敗:\n" + err + "\n請確認 UGen300 已插上、沒有其他 demo 正在使用它、models/ 內有 .hef。", "bot"); return
        self.status.setText(f"{self.model} 就緒 · {MODELS[self.model]['tps']} · " + date_context()[:22])
        self.input.setEnabled(True); self.send_btn.setEnabled(True); self.input.setPlaceholderText("描述你要的程式,Enter 送出"); self.input.setFocus()
        if self.snapshot: QTimer.singleShot(300, lambda: self._ask(QUICK[0][1]))

    def _send(self):
        txt = self.input.text().strip()
        if txt: self.input.clear(); self._ask(txt)

    def _ask(self, txt):
        if self.gen is not None and self.gen.isRunning(): return
        self._add_bubble(txt, "user"); self.cur_bot = self._add_bubble("", "bot"); self.full = ""
        self.input.setEnabled(False); self.send_btn.setEnabled(False); self.status.setText("撰寫中…")
        self.gen = GenWorker(self.engine, txt)
        self.gen.piece.connect(self._piece); self.gen.finished_ok.connect(self._done); self.gen.failed.connect(self._fail); self.gen.start()

    def _piece(self, s):
        self.full += s
        self.cur_bot.setText(to_traditional(CB.strip_code_to_text(self.full)) or "…")
        lang, code = CB.last_code(self.full)
        if code is not None:
            self.lang = lang; self.pane.setText(f"程式碼({lang},可直接修改)")
            if self.code.toPlainText() != code:
                self.code.setPlainText(code); self.code.moveCursor(QTextCursor.End)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def _done(self, tps, ttft, n):
        self.status.setText(f"{self.model} · {tps:.1f} token/秒 · 首字 {ttft:.2f} 秒 · {n} tokens · 全程離線")
        self.input.setEnabled(True); self.send_btn.setEnabled(True); self.input.setFocus()
        if not CB.code_blocks(self.full): self.cur_bot.setText(to_traditional(self.full))
        if self.snapshot: QTimer.singleShot(300, self._run)

    def _fail(self, err):
        self.cur_bot.setText((self.cur_bot.text() + "\n[推論失敗] " + err).strip()); self.status.setText("推論失敗:" + err)
        self.input.setEnabled(True); self.send_btn.setEnabled(True)

    def _clear(self):
        self.engine.reset()
        while self.feed_l.count() > 1:
            it = self.feed_l.takeAt(0); w = it.widget()
            if w: w.deleteLater()
        self.code.clear(); self.console.clear(); self.run_info.setText(""); self._add_bubble("對話已清除。", "bot")

    # ----- 程式碼區 -----
    def _run(self):
        code = self.code.toPlainText()
        if not code.strip() or (self.runner is not None and self.runner.isRunning()): return
        self.console.setPlainText("執行中…"); self.run_btn.setEnabled(False); self.run_info.setText("")
        self.runner = RunWorker(code); self.runner.done.connect(self._ran); self.runner.start()

    def _ran(self, r):
        self.console.clear(); c = self.console.textCursor()
        def put(text, color):
            f = QTextCharFormat(); f.setForeground(QColor(color)); c.insertText(text, f)
        if r["stdout"]: put(r["stdout"], "#9ece6a")
        if r["stderr"]: put(("\n" if r["stdout"] else "") + r["stderr"], "#ff6b6b")
        if not r["stdout"] and not r["stderr"]: put("(沒有輸出)", "#8b8b93")
        self.run_info.setText(("✓ 成功" if r["ok"] else "✗ 失敗") + f" · {r['elapsed']:.2f} 秒" + (" · 沙箱逾時" if r["timeout"] else ""))
        self.run_btn.setEnabled(True)
        if self.snapshot: QTimer.singleShot(500, self._take_snapshot)

    def _copy(self):
        QApplication.clipboard().setText(self.code.toPlainText()); self.run_info.setText("已複製到剪貼簿")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "存檔", os.path.join(HERE, "snippet.py"), "Python (*.py);;所有檔案 (*)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f: f.write(self.code.toPlainText())
                self.run_info.setText("已存到 " + os.path.basename(path))
            except OSError as e: QMessageBox.warning(self, "存檔失敗", str(e))

    def _take_snapshot(self):
        self.grab().save(self.snapshot); print("[snapshot]", self.snapshot); QApplication.instance().quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_CODE_MODEL, choices=[m for m, c in MODELS.items()])
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    if args.selftest:
        e = LLMEngine(kind="code").load(args.model, on_status=print)
        ans, _ = e.ask_all("寫一個 Python 函式 fact(n) 回傳 n 的階乘,並印出 fact(5)。", max_tokens=200)
        lang, code = CB.last_code(ans)
        r = CB.run_python(code) if code else dict(ok=False, stdout="", stderr="回答裡沒有程式碼區塊")
        print(f"[selftest] {args.model} OK | {e.rate.tps:.1f} tok/s | 程式碼={'有' if code else '無'} | 執行={'OK' if r['ok'] else 'FAIL'} | 輸出={r['stdout'].strip()[:40]!r} {r['stderr'].strip()[:60]!r}")
        return 0
    app = QApplication(sys.argv); app.setStyleSheet(STYLE)
    w = CodeApp(args.model, args.snapshot); w.show()
    return app.exec()


if __name__ == "__main__":
    code = 0
    try:
        code = main() or 0
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:  # noqa: BLE001  任何未預期錯誤都要走硬退出,否則 HailoRT 收尾會讓 UGen300 掉線
        traceback.print_exc(); code = 1
    hailo_vdevice.exit_now(code)
