"""
offline_chat.py —— 離線 ChatGPT:PySide6 聊天視窗 + UGen300 本機 LLM 串流。

用法:python offline_chat.py [--model Qwen3-1.7B|Llama3.2-1B|Qwen2.5-1.5B]
      python offline_chat.py --selftest                 (載模型、問一題、印結果、離開)
      python offline_chat.py --snapshot out.png         (自動問一題、答完截圖、離開)
切換模型 = 重啟程序(HailoRT 5.3.2 同程序內換 LLM 會 INTERNAL_FAILURE)。
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QFont, QTextCursor, QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QComboBox, QLineEdit, QScrollArea, QFrame, QSizePolicy)

from llm_engine import LLMEngine, MODELS, DEFAULT_MODEL

STYLE = """
QMainWindow, QWidget { background: #1b1b1f; color: #ececec; font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "Noto Sans TC"; }
QLabel#title { font-size: 22px; font-weight: 700; }
QLabel#status { color: #9aa0a6; font-size: 14px; }
QLabel#offline { color: #37d67a; font-size: 14px; font-weight: 700; border: 1px solid #37d67a; border-radius: 6px; padding: 3px 8px; }
QLabel#online { color: #ff6b6b; font-size: 14px; font-weight: 700; border: 1px solid #ff6b6b; border-radius: 6px; padding: 3px 8px; }
QComboBox, QLineEdit, QPushButton { background: #2a2a30; border: 1px solid #3a3a42; border-radius: 8px; padding: 8px 12px; font-size: 16px; color: #ececec; }
QLineEdit { font-size: 18px; }
QPushButton#send { background: #2f6fed; border: none; font-weight: 700; }
QPushButton#send:disabled { background: #3a3a42; color: #888; }
QScrollArea { border: none; }
QLabel.bubble { font-size: 18px; line-height: 1.5; padding: 12px 16px; border-radius: 14px; }
QLabel#user { background: #2f6fed; color: white; }
QLabel#bot { background: #2a2a30; }
QLabel#think { background: #232327; color: #8b8b93; font-size: 13px; font-style: italic; border-radius: 10px; padding: 8px 12px; }
"""


class LoadWorker(QThread):
    status = Signal(str); done = Signal(bool, str)
    def __init__(self, engine, model): super().__init__(); self.engine, self.model = engine, model
    def run(self):
        try:
            self.engine.load(self.model, on_status=self.status.emit); self.done.emit(True, "")
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, f"{type(e).__name__}: {e}")


class GenWorker(QThread):
    piece = Signal(str, str); finished_ok = Signal(float, float, int); failed = Signal(str)
    def __init__(self, engine, text): super().__init__(); self.engine, self.text = engine, text
    def run(self):
        try:
            for kind, s in self.engine.stream(self.text):
                self.piece.emit(kind, s)
            r = self.engine.rate; self.finished_ok.emit(r.tps, r.ttft, r.n)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


def _net_online(timeout=0.4):
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=timeout).close(); return True
    except OSError:
        return False


class Bubble(QLabel):
    def __init__(self, text, who):
        super().__init__(text)
        self.setObjectName(who); self.setProperty("class", "bubble")
        self.setWordWrap(True); self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setMaximumWidth(760); self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        f = QFont(); f.setPointSize(13 if who != "think" else 10); self.setFont(f)

    def append(self, s): self.setText(self.text() + s)


class ThinkBubble(Bubble):
    """Qwen3 的 <think> 內容:預設摺疊成一行「思考中…(點開)」,點一下展開/收合。"""
    def __init__(self):
        super().__init__("思考中…(點一下展開)", "think"); self.full = ""; self.expanded = False
        self.setCursor(Qt.PointingHandCursor)

    def append(self, s):
        self.full += s; self._refresh()

    def _refresh(self):
        n = len(self.full)
        self.setText(self.full if self.expanded else f"思考中…({n} 字,點一下展開)")

    def mousePressEvent(self, e):
        self.expanded = not self.expanded; self._refresh()


class Chat(QMainWindow):
    net_sig = Signal(bool)

    def __init__(self, model, snapshot=""):
        super().__init__()
        self.net_sig.connect(self._set_net)
        self.engine = LLMEngine(); self.model = model; self.snapshot = snapshot
        self.setWindowTitle("UGen300 離線 ChatGPT"); self.resize(1100, 760)
        root = QWidget(); self.setCentralWidget(root); v = QVBoxLayout(root); v.setContentsMargins(20, 14, 20, 14); v.setSpacing(10)

        top = QHBoxLayout()
        t = QLabel("離線 ChatGPT"); t.setObjectName("title"); top.addWidget(t)
        top.addSpacing(16)
        self.combo = QComboBox(); self.combo.addItems(list(MODELS)); self.combo.setCurrentText(model)
        self.combo.currentTextChanged.connect(self._switch_model); top.addWidget(self.combo)
        self.clear_btn = QPushButton("清除對話"); self.clear_btn.clicked.connect(self._clear); top.addWidget(self.clear_btn)
        top.addStretch(1)
        self.net = QLabel("網路檢查中…"); self.net.setObjectName("status"); top.addWidget(self.net)
        self.cost = QLabel("0 元/月 · 資料不出這台電腦"); self.cost.setObjectName("offline"); top.addWidget(self.cost)
        v.addLayout(top)

        self.status = QLabel("載入模型中…"); self.status.setObjectName("status"); v.addWidget(self.status)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.feed = QWidget(); self.feed_l = QVBoxLayout(self.feed); self.feed_l.setSpacing(10); self.feed_l.addStretch(1)
        self.scroll.setWidget(self.feed); v.addWidget(self.scroll, 1)

        bottom = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("輸入問題,Enter 送出(模型載入中…)"); self.input.setEnabled(False)
        self.input.returnPressed.connect(self._send); bottom.addWidget(self.input, 1)
        self.send_btn = QPushButton("送出"); self.send_btn.setObjectName("send"); self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._send); bottom.addWidget(self.send_btn)
        v.addLayout(bottom)

        self.cur_bot = None; self.cur_think = None; self.gen = None
        self._add_bubble("你好!我是跑在 UGen300 上的本機 AI,現在可以拔掉網路線再問我問題。", "bot")
        self.loader = LoadWorker(self.engine, model); self.loader.status.connect(self.status.setText)
        self.loader.done.connect(self._loaded); self.loader.start()
        self.net_timer = QTimer(self); self.net_timer.timeout.connect(self._check_net); self.net_timer.start(4000); self._check_net()

    # ----- 網路徽章(拔線的瞬間會變綠)-----
    def _check_net(self):
        def run():
            self.net_sig.emit(_net_online())
        threading.Thread(target=run, daemon=True).start()

    def _set_net(self, on):
        if on:
            self.net.setText("網路:連線中"); self.net.setObjectName("online")
        else:
            self.net.setText("網路:已斷線 ✓ 仍可運作"); self.net.setObjectName("offline")
        self.net.style().unpolish(self.net); self.net.style().polish(self.net)

    # ----- 對話 -----
    def _add_bubble(self, text, who):
        row = QHBoxLayout(); b = Bubble(text, who)
        if who == "user": row.addStretch(1); row.addWidget(b)
        else: row.addWidget(b); row.addStretch(1)
        w = QWidget(); w.setLayout(row)
        self.feed_l.insertWidget(self.feed_l.count() - 1, w)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))
        return b

    def _loaded(self, ok, err):
        if not ok:
            self.status.setText("模型載入失敗:" + err + "  (請確認 UGen300 已插上、models/ 內有 .hef)")
            self._add_bubble("模型載入失敗:\n" + err, "bot"); return
        self.status.setText(f"{self.model} 就緒 · {MODELS[self.model]['tps']} · 等待提問")
        self.input.setEnabled(True); self.send_btn.setEnabled(True); self.input.setPlaceholderText("輸入問題,Enter 送出"); self.input.setFocus()
        if self.snapshot:
            QTimer.singleShot(300, lambda: self._ask("用三句話說明為什麼學生應該學邊緣 AI。"))

    def _send(self):
        txt = self.input.text().strip()
        if txt: self.input.clear(); self._ask(txt)

    def _ask(self, txt):
        self._add_bubble(txt, "user")
        self.cur_think = None; self.cur_bot = self._add_bubble("", "bot")
        self.input.setEnabled(False); self.send_btn.setEnabled(False); self.status.setText("思考中…")
        self.gen = GenWorker(self.engine, txt)
        self.gen.piece.connect(self._piece); self.gen.finished_ok.connect(self._done); self.gen.failed.connect(self._fail); self.gen.start()

    def _piece(self, kind, s):
        if kind == "think":
            if self.cur_think is None:
                self.cur_think = ThinkBubble(); row = QHBoxLayout(); row.addWidget(self.cur_think); row.addStretch(1)
                w = QWidget(); w.setLayout(row); self.feed_l.insertWidget(self.feed_l.count() - 2, w)
            self.cur_think.append(s)
        else:
            self.cur_bot.append(s)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def _done(self, tps, ttft, n):
        self.status.setText(f"{self.model} · {tps:.1f} token/秒 · 首字 {ttft:.2f} 秒 · {n} tokens · 全程離線")
        self.input.setEnabled(True); self.send_btn.setEnabled(True); self.input.setFocus()
        if self.snapshot:
            QTimer.singleShot(400, self._take_snapshot)

    def _fail(self, err):
        self.cur_bot.append("\n[推論失敗] " + err); self.status.setText("推論失敗:" + err)
        self.input.setEnabled(True); self.send_btn.setEnabled(True)

    def _clear(self):
        self.engine.reset()
        while self.feed_l.count() > 1:
            it = self.feed_l.takeAt(0); w = it.widget()
            if w: w.deleteLater()
        self._add_bubble("對話已清除。", "bot")

    def _switch_model(self, name):
        if name == self.model: return
        self.status.setText(f"切換到 {name}:重新啟動中…")
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "--model", name], cwd=os.path.dirname(os.path.abspath(__file__)))
        QTimer.singleShot(300, QApplication.instance().quit)

    def _take_snapshot(self):
        self.grab().save(self.snapshot); print("[snapshot]", self.snapshot); QApplication.instance().quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--snapshot", default="")
    args = ap.parse_args()
    if args.selftest:
        e = LLMEngine().load(args.model, on_status=print)
        ans, _ = e.ask_all("一句話自我介紹。", max_tokens=60)
        print(f"[selftest] {args.model} OK | {e.rate.tps:.1f} tok/s | {ans[:80]}"); return
    app = QApplication(sys.argv); app.setStyleSheet(STYLE)
    w = Chat(args.model, args.snapshot); w.show()
    code = app.exec()
    import hailo_vdevice
    hailo_vdevice.exit_now(code)


if __name__ == "__main__":
    main()
