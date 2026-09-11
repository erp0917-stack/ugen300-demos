# -*- coding: utf-8 -*-
"""
meeting_gui.py — WQ1 離線會議記錄(PySide6 桌面 GUI)
=========================================================
深色現代介面,包裝 transcribe.py / summarize.py / record_audio.py,
全程本機推論、不連網。介面預設 English,可切繁/簡。

VDevice 一次只能存在一個(Hailo-10H 硬限制),所以推論順序必須是:
    load Whisper → transcribe → transcribe.close()
                ↓ (一定要 close,LLM 才搶得到 VDevice)
    load LLM     → summarize  → summarize.close()
這兩段推論加起來十幾到幾十秒,放在 InferenceWorker(QThread)裡跑,
用 signal/slot 把結果回送主執行緒更新畫面,GUI 不會凍住。
"""

import os
import re
import sys
import time

import numpy as np
import sounddevice as sd

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from record_audio import CHANNELS, SAMPLE_RATE, _save_wav
from summarize import close as summarize_close
from summarize import summarize
from transcribe import close as transcribe_close
from transcribe import transcribe

# OpenCC 沒裝就略過,英文模式本來就不需要;繁體模式 summarize 已輸出繁體
try:
    from opencc import OpenCC
    _CC_S2T = OpenCC("s2t")
    _CC_T2S = OpenCC("t2s")
    _HAS_OPENCC = True
except Exception:
    _HAS_OPENCC = False


# ════════════════════════════════════════════════════════════════════════
#  介面字串總表:要改字、加語言只動這裡
# ════════════════════════════════════════════════════════════════════════
STRINGS = {
    "title":            {"en": "UGen300 · Meeting Notes",
                         "zh": "UGen300 · 會議記錄",
                         "cn": "UGen300 · 会议记录"},
    "badge_offline":    {"en": "On-device · offline",
                         "zh": "本機 · 離線",
                         "cn": "本机 · 离线"},
    "lang_en":          {"en": "English", "zh": "English", "cn": "English"},
    "lang_zh":          {"en": "繁體中文", "zh": "繁體中文", "cn": "繁体中文"},
    "lang_cn":          {"en": "简体中文", "zh": "简体中文", "cn": "简体中文"},
    "mic_label":        {"en": "Microphone", "zh": "麥克風", "cn": "麦克风"},
    "mic_default":      {"en": "System default", "zh": "系統預設", "cn": "系统默认"},

    "stage_transcribe": {"en": "Transcribe", "zh": "語音轉文字", "cn": "语音转文字"},
    "stage_summarize":  {"en": "Summarize",  "zh": "整理摘要",   "cn": "整理摘要"},
    "status_waiting":   {"en": "waiting",    "zh": "等待中",     "cn": "等待中"},
    "status_running":   {"en": "running… {sec}s",
                         "zh": "處理中… {sec} 秒",
                         "cn": "处理中… {sec} 秒"},
    "status_done":      {"en": "done · {sec}s",
                         "zh": "完成 · {sec} 秒",
                         "cn": "完成 · {sec} 秒"},

    "card_transcript":  {"en": "Transcript", "zh": "逐字稿", "cn": "逐字稿"},
    "card_summary":     {"en": "Summary",    "zh": "會議摘要", "cn": "会议摘要"},
    "sec_keypoints":    {"en": "Key Points",   "zh": "會議重點", "cn": "会议重点"},
    "sec_actions":      {"en": "Action Items", "zh": "待辦事項", "cn": "待办事项"},
    "sec_decisions":    {"en": "Decisions",    "zh": "決議事項", "cn": "决议事项"},

    "empty_transcript": {"en": "Transcript will appear here after recording.",
                         "zh": "錄音結束後逐字稿會顯示在這裡。",
                         "cn": "录音结束后逐字稿会显示在这里。"},
    "empty_summary":    {"en": "Summary will appear here after the transcript is ready.",
                         "zh": "逐字稿完成後摘要會顯示在這裡。",
                         "cn": "逐字稿完成后摘要会显示在这里。"},
    "hint_idle":        {"en": "Click the mic to start recording.",
                         "zh": "點麥克風開始錄音。",
                         "cn": "点麦克风开始录音。"},
    "hint_done_ready":  {"en": "Done. Click the mic to record another segment.",
                         "zh": "完成。點麥克風錄下一段。",
                         "cn": "完成。点麦克风录下一段。"},
    "hint_recording":   {"en": "Recording… click again to stop.",
                         "zh": "錄音中…再點一次停止。",
                         "cn": "录音中…再点一次停止。"},
    "hint_processing":  {"en": "Processing on UGen300…",
                         "zh": "UGen300 處理中…",
                         "cn": "UGen300 处理中…"},
    "err_no_audio":     {"en": "No audio captured. Try again.",
                         "zh": "沒有收到聲音,請再試一次。",
                         "cn": "没有收到声音,请再试一次。"},
    "err_generic":      {"en": "Something went wrong: {e}",
                         "zh": "處理時發生問題:{e}",
                         "cn": "处理时发生问题:{e}"},
    "footer":           {"en": "Everything runs on UGen300. Nothing leaves this device.",
                         "zh": "全程在 UGen300 上運算,資料不離開本機。",
                         "cn": "全程在 UGen300 上运算,数据不离开本机。"},
}


def T(key, lang, **kw):
    s = STRINGS.get(key, {}).get(lang) or STRINGS.get(key, {}).get("en") or key
    return s.format(**kw) if kw else s


# ════════════════════════════════════════════════════════════════════════
#  簡繁轉換 + 摘要分段解析
# ════════════════════════════════════════════════════════════════════════
def localize_cjk(text, lang):
    """介面=cn 時轉簡體;=zh 時保險轉繁體;=en 原樣回傳。OpenCC 沒裝就跳過。"""
    if not text or not _HAS_OPENCC:
        return text
    try:
        if lang == "cn":
            return _CC_T2S.convert(text)
        if lang == "zh":
            return _CC_S2T.convert(text)
    except Exception:
        pass
    return text


_SECTION_PATS = (
    ("keypoints", (r"會議重點\s*[:：]?", r"会议重点\s*[:：]?", r"Key\s*Points\s*[:：]?")),
    ("actions",   (r"待辦事項\s*[:：]?", r"待办事项\s*[:：]?", r"Action\s*Items\s*[:：]?")),
    ("decisions", (r"決議事項\s*[:：]?", r"决议事项\s*[:：]?", r"Decisions\s*[:：]?")),
)


def parse_summary(text):
    """切成 {'keypoints', 'actions', 'decisions'};找不到至少兩個段落就回 None。"""
    if not text or not isinstance(text, str):
        return None
    hits = []
    for key, pats in _SECTION_PATS:
        for pat in pats:
            m = re.search(pat, text)
            if m:
                hits.append((m.start(), m.end(), key))
                break
    if len(hits) < 2:
        return None
    hits.sort()
    out = {"keypoints": "", "actions": "", "decisions": ""}
    for i, (_s, e, k) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        out[k] = text[e:nxt].strip()
    return out


# ════════════════════════════════════════════════════════════════════════
#  背景錄音 thread:用 sd.InputStream 收 frames,停止時 _save_wav
# ════════════════════════════════════════════════════════════════════════
class RecorderThread(QThread):
    finished_ok = Signal(str)   # wav path
    failed = Signal(str)

    def __init__(self, device, out_path):
        super().__init__()
        self._device = device
        self._out_path = out_path
        self._stop = False
        self._frames = []

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            def cb(indata, n, t, status):
                self._frames.append(indata.copy())
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype="int16", device=self._device, callback=cb):
                while not self._stop:
                    self.msleep(50)
            if not self._frames:
                self.failed.emit("no_audio")
                return
            audio = np.concatenate(self._frames, axis=0)
            _save_wav(self._out_path, audio)
            self.finished_ok.emit(self._out_path)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════
#  推論 worker:嚴格依序 transcribe → close → summarize → close
# ════════════════════════════════════════════════════════════════════════
class InferenceWorker(QThread):
    transcribe_started = Signal()
    transcribe_done = Signal(str, float)
    summarize_started = Signal()
    summarize_done = Signal(str, float)
    failed = Signal(str)

    def __init__(self, wav_path, whisper_lang):
        super().__init__()
        self._wav = wav_path
        self._lang = whisper_lang

    def run(self):
        try:
            self.transcribe_started.emit()
            t0 = time.time()
            transcript = transcribe(self._wav, language=self._lang)
            self.transcribe_done.emit(transcript or "", time.time() - t0)

            # Hailo VDevice 一次只能一個;LLM 載入前必須先釋放 Whisper
            try:
                transcribe_close()
            except Exception:
                pass

            self.summarize_started.emit()
            t1 = time.time()
            summary = summarize(transcript or "")
            self.summarize_done.emit(summary or "", time.time() - t1)

            try:
                summarize_close()
            except Exception:
                pass
        except Exception as e:
            try:
                transcribe_close()
            except Exception:
                pass
            try:
                summarize_close()
            except Exception:
                pass
            self.failed.emit(f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════
#  圓形麥克風按鈕:閒置=青色 + 麥克風形狀;錄音=紅色 + 方塊 + 脈衝
# ════════════════════════════════════════════════════════════════════════
class RecordButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 180)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self._recording = False
        self._pulse = 0.0

        self._anim = QPropertyAnimation(self, b"pulse")
        self._anim.setDuration(1300)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)

    def set_state(self, *, recording=None, busy=None):
        if recording is not None:
            self._recording = recording
            if recording:
                self._anim.start()
            else:
                self._anim.stop()
                self._pulse = 0.0
        if busy is not None:
            self.setEnabled(not busy)
        self.update()

    def _get_pulse(self):
        return self._pulse

    def _set_pulse(self, v):
        self._pulse = v
        self.update()

    pulse = Property(float, _get_pulse, _set_pulse)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = min(w, h) / 2.0 - 14

        # 外圈脈衝(僅錄音中)
        if self._recording:
            ring = r + 6 + 14 * self._pulse
            alpha = int(160 * (1.0 - self._pulse))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(229, 70, 88, alpha))
            p.drawEllipse(QRectF(cx - ring, cy - ring, ring * 2, ring * 2))

        # 主圓
        if not self.isEnabled():
            main_color = QColor(70, 78, 92)
        elif self._recording:
            main_color = QColor(229, 70, 88)
        else:
            main_color = QColor(46, 161, 151)
        p.setPen(QPen(QColor(255, 255, 255, 26), 2))
        p.setBrush(main_color)
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # 中心圖示
        if self._recording:
            sq = r * 0.42
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 235))
            p.drawRoundedRect(QRectF(cx - sq / 2, cy - sq / 2, sq, sq), 6, 6)
        else:
            body_w = r * 0.42
            body_h = r * 0.68
            body_top = cy - r * 0.55
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 235))
            p.drawRoundedRect(
                QRectF(cx - body_w / 2, body_top, body_w, body_h),
                body_w / 2, body_w / 2,
            )
            pen = QPen(QColor(255, 255, 255, 235), max(3.0, r * 0.06))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            arc_w = body_w * 1.85
            arc_h = body_h * 0.62
            arc_rect = QRectF(cx - arc_w / 2,
                              body_top + body_h * 0.35,
                              arc_w, arc_h)
            p.drawArc(arc_rect, 0, -180 * 16)
            stand_top = body_top + body_h * 0.35 + arc_h * 0.5
            p.drawLine(int(cx), int(stand_top),
                       int(cx), int(stand_top + r * 0.18))
            p.drawLine(int(cx - r * 0.18), int(stand_top + r * 0.18),
                       int(cx + r * 0.18), int(stand_top + r * 0.18))


# ════════════════════════════════════════════════════════════════════════
#  進度列:○ → 旋轉 → ✓  / waiting · running…Ns · done · N.Ns
# ════════════════════════════════════════════════════════════════════════
SPINNER_CHARS = "◐◓◑◒"


class StageRow(QWidget):
    def __init__(self, label_key, lang):
        super().__init__()
        self._label_key = label_key
        self._lang = lang
        self._state = "waiting"
        self._spin_idx = 0
        self._t_started = 0.0
        self._elapsed = 0.0

        h = QHBoxLayout(self)
        h.setContentsMargins(2, 6, 2, 6)
        h.setSpacing(10)

        self.icon = QLabel("○")
        self.icon.setObjectName("stageIcon")
        self.icon.setFixedWidth(18)
        self.icon.setAlignment(Qt.AlignCenter)

        self.name = QLabel("")
        self.name.setObjectName("stageName")

        self.status = QLabel("")
        self.status.setObjectName("stageStatus")
        self.status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        h.addWidget(self.icon)
        h.addWidget(self.name)
        h.addStretch()
        h.addWidget(self.status)
        self.retranslate(lang)

    def retranslate(self, lang):
        self._lang = lang
        self.name.setText(T(self._label_key, lang))
        self._refresh()

    def reset(self):
        self._state = "waiting"
        self._refresh()

    @Slot()
    def start(self):
        self._state = "running"
        self._t_started = time.time()
        self._spin_idx = 0
        self._refresh()

    def done(self, elapsed):
        self._state = "done"
        self._elapsed = elapsed
        self._refresh()

    def tick(self):
        if self._state != "running":
            return
        self._spin_idx = (self._spin_idx + 1) % len(SPINNER_CHARS)
        self._refresh()

    def _refresh(self):
        if self._state == "waiting":
            self.icon.setText("○")
            self.icon.setStyleSheet("color:#5C6470;")
            self.status.setText(T("status_waiting", self._lang))
        elif self._state == "running":
            self.icon.setText(SPINNER_CHARS[self._spin_idx])
            self.icon.setStyleSheet("color:#2EA197;")
            el = time.time() - self._t_started
            self.status.setText(T("status_running", self._lang, sec=f"{el:.0f}"))
        else:  # done
            self.icon.setText("✓")
            self.icon.setStyleSheet("color:#3FB950;")
            self.status.setText(T("status_done", self._lang, sec=f"{self._elapsed:.1f}"))


# ════════════════════════════════════════════════════════════════════════
#  卡片
# ════════════════════════════════════════════════════════════════════════
class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 16)
        v.setSpacing(10)
        self.title = QLabel("")
        self.title.setObjectName("cardTitle")
        v.addWidget(self.title)
        self._v = v

    def add(self, widget):
        self._v.addWidget(widget)


# ════════════════════════════════════════════════════════════════════════
#  主視窗
# ════════════════════════════════════════════════════════════════════════
SECTION_KEYS = (
    ("keypoints", "sec_keypoints"),
    ("actions",   "sec_actions"),
    ("decisions", "sec_decisions"),
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._lang = "en"
        self._device = None
        self._state = "idle"   # idle / recording / processing
        self._has_completed_round = False  # 第一輪做完後 hint 改成「再錄一段」
        self._raw_transcript = ""
        self._raw_summary = ""
        self._recorder = None
        self._worker = None

        self.setWindowTitle(T("title", self._lang))
        self.resize(1080, 760)
        self.setMinimumSize(900, 660)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        rv = QVBoxLayout(root)
        rv.setContentsMargins(28, 22, 28, 22)
        rv.setSpacing(16)

        rv.addLayout(self._build_topbar())
        rv.addLayout(self._build_controls())
        rv.addLayout(self._build_center(), 0)
        rv.addWidget(self._build_results(), 1)

        self.lbl_footer = QLabel("")
        self.lbl_footer.setObjectName("footer")
        self.lbl_footer.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.lbl_footer)

        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(140)
        self._spin_timer.timeout.connect(self._tick_spinners)
        self._spin_timer.start()

        self.setStyleSheet(self._stylesheet())
        self._apply_language()

    # ── 頂列:標題 + 徽章 ───────────────────────────────────────────
    def _build_topbar(self):
        row = QHBoxLayout()
        self.lbl_title = QLabel("")
        self.lbl_title.setObjectName("appTitle")
        row.addWidget(self.lbl_title)
        row.addStretch()
        self.lbl_badge = QLabel("")
        self.lbl_badge.setObjectName("badge")
        row.addWidget(self.lbl_badge)
        return row

    # ── 次列:語言切換 + 麥克風選單 ─────────────────────────────────
    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        self.lang_group = QButtonGroup(self)
        self.lang_group.setExclusive(True)
        self.lang_buttons = {}
        for code, key in (("en", "lang_en"), ("zh", "lang_zh"), ("cn", "lang_cn")):
            b = QPushButton("")
            b.setObjectName("langBtn")
            b.setCheckable(True)
            if code == self._lang:
                b.setChecked(True)
            b.clicked.connect(lambda _checked=False, c=code: self._on_lang_change(c))
            self.lang_group.addButton(b)
            self.lang_buttons[code] = b
            row.addWidget(b)
        row.addStretch()

        self.lbl_mic = QLabel("")
        self.lbl_mic.setObjectName("subtle")
        row.addWidget(self.lbl_mic)
        self.mic_combo = QComboBox()
        self.mic_combo.setObjectName("micCombo")
        self.mic_combo.setMinimumWidth(280)
        self._populate_mic_combo()
        self.mic_combo.currentIndexChanged.connect(self._on_mic_change)
        row.addWidget(self.mic_combo)
        return row

    # ── 中央:錄音鈕 + 提示 + 兩段進度 ───────────────────────────────
    def _build_center(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        self.btn_mic = RecordButton()
        self.btn_mic.clicked.connect(self._on_mic_click)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.btn_mic)
        btn_row.addStretch()
        col.addLayout(btn_row)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setObjectName("hint")
        self.lbl_hint.setAlignment(Qt.AlignCenter)
        col.addWidget(self.lbl_hint)

        stage_box = QFrame()
        stage_box.setObjectName("stageBox")
        stage_box.setMinimumWidth(380)
        stage_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sv = QVBoxLayout(stage_box)
        sv.setContentsMargins(20, 6, 20, 6)
        sv.setSpacing(0)
        self.stage_t = StageRow("stage_transcribe", self._lang)
        self.stage_s = StageRow("stage_summarize", self._lang)
        sv.addWidget(self.stage_t)
        sv.addWidget(self.stage_s)

        wrap = QHBoxLayout()
        wrap.addStretch()
        wrap.addWidget(stage_box)
        wrap.addStretch()
        col.addLayout(wrap)
        return col

    # ── 結果區:Transcript 卡 + Summary 卡(三段) ─────────────────
    def _build_results(self):
        split = QSplitter(Qt.Horizontal)
        split.setObjectName("results")
        split.setHandleWidth(10)

        # Transcript card
        self.card_tr = Card()
        self.tr_view = QTextEdit()
        self.tr_view.setReadOnly(True)
        self.tr_view.setObjectName("transcriptView")
        self.card_tr.add(self.tr_view)
        split.addWidget(self.card_tr)

        # Summary card
        self.card_sm = Card()
        self.sm_scroll = QScrollArea()
        self.sm_scroll.setWidgetResizable(True)
        self.sm_scroll.setFrameShape(QFrame.NoFrame)
        self.sm_inner = QWidget()
        self.sm_inner.setObjectName("summaryInner")
        smv = QVBoxLayout(self.sm_inner)
        smv.setContentsMargins(0, 2, 0, 4)
        smv.setSpacing(14)

        self.sm_placeholder = QLabel("")
        self.sm_placeholder.setObjectName("placeholder")
        self.sm_placeholder.setWordWrap(True)
        smv.addWidget(self.sm_placeholder)

        self.sec_widgets = {}
        for key, _label_key in SECTION_KEYS:
            block = QFrame()
            block.setObjectName("summarySection")
            bv = QVBoxLayout(block)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.setSpacing(4)
            head = QLabel("")
            head.setObjectName("sectionHead")
            body = QLabel("")
            body.setObjectName("sectionBody")
            body.setWordWrap(True)
            body.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            bv.addWidget(head)
            bv.addWidget(body)
            smv.addWidget(block)
            self.sec_widgets[key] = (head, body, block)
        smv.addStretch()
        self.sm_scroll.setWidget(self.sm_inner)
        self.card_sm.add(self.sm_scroll)

        # 解析失敗時的整段 fallback
        self.sm_fallback = QTextEdit()
        self.sm_fallback.setReadOnly(True)
        self.sm_fallback.setObjectName("transcriptView")
        self.sm_fallback.hide()
        self.card_sm.add(self.sm_fallback)

        split.addWidget(self.card_sm)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        return split

    # ── QSS ─────────────────────────────────────────────────────────
    def _stylesheet(self):
        return """
        QWidget {
            color: #E6EDF3;
            background: #0E1116;
            font-family: "Segoe UI", "Microsoft JhengHei UI", "Microsoft JhengHei",
                         "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        }
        QWidget#root { background: #0E1116; }

        QLabel#appTitle { font-size: 22px; font-weight: 600; letter-spacing: 0.3px; }
        QLabel#badge {
            color: #7EE0C9;
            background: rgba(46, 161, 151, 0.16);
            border: 1px solid rgba(46, 161, 151, 0.40);
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 12px;
        }
        QLabel#subtle  { color: #8B949E; font-size: 12px; }
        QLabel#hint    { color: #8B949E; font-size: 13px; padding-top: 2px; }
        QLabel#footer  { color: #6E7681; font-size: 12px; padding-top: 6px; }

        QPushButton#langBtn {
            background: #161B22;
            border: 1px solid #2A323D;
            border-radius: 14px;
            padding: 5px 14px;
            color: #C9D1D9;
            font-size: 12px;
        }
        QPushButton#langBtn:hover    { border-color: #3D4858; color: #E6EDF3; }
        QPushButton#langBtn:checked  {
            background: rgba(46, 161, 151, 0.18);
            border-color: #2EA197;
            color: #FFFFFF;
        }
        QPushButton#langBtn:disabled {
            color: #4A5260;
            border-color: #1A2027;
            background: #11151B;
        }
        QPushButton#langBtn:disabled:checked {
            background: rgba(46, 161, 151, 0.06);
            border-color: rgba(46, 161, 151, 0.30);
            color: #7A8190;
        }

        QComboBox#micCombo {
            background: #161B22;
            border: 1px solid #2A323D;
            border-radius: 8px;
            padding: 5px 12px;
            min-height: 22px;
            color: #C9D1D9;
        }
        QComboBox#micCombo:hover  { border-color: #3D4858; }
        QComboBox#micCombo::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #161B22;
            color: #C9D1D9;
            border: 1px solid #2A323D;
            selection-background-color: rgba(46, 161, 151, 0.25);
            selection-color: #FFFFFF;
            outline: 0;
        }

        QFrame#stageBox {
            background: rgba(22, 27, 34, 0.55);
            border: 1px solid #1F2630;
            border-radius: 10px;
        }
        QLabel#stageIcon   { font-size: 14px; }
        QLabel#stageName   { font-size: 13px; color: #C9D1D9; }
        QLabel#stageStatus { font-size: 12px; color: #8B949E; }

        QFrame#card {
            background: #161B22;
            border: 1px solid #1F2630;
            border-radius: 12px;
        }
        QLabel#cardTitle {
            font-size: 14px; font-weight: 600; color: #E6EDF3;
            padding-bottom: 2px;
        }

        QTextEdit#transcriptView {
            background: transparent;
            border: none;
            color: #D7DEE6;
            font-size: 13px;
        }
        QScrollArea          { background: transparent; border: none; }
        QWidget#summaryInner { background: transparent; }

        QLabel#sectionHead {
            color: #7EE0C9;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.2px;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(46, 161, 151, 0.28);
        }
        QLabel#sectionBody  { color: #D7DEE6; font-size: 13px; padding-top: 2px; }
        QLabel#placeholder  { color: #6E7681; font-size: 13px; font-style: italic; }

        QSplitter::handle { background: transparent; }

        QScrollBar:vertical {
            background: transparent; width: 8px; margin: 4px 2px 4px 0;
        }
        QScrollBar::handle:vertical {
            background: #2A323D; border-radius: 4px; min-height: 30px;
        }
        QScrollBar::handle:vertical:hover { background: #3D4858; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """

    # ── 麥克風列舉 ──────────────────────────────────────────────────
    def _populate_mic_combo(self):
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        self.mic_combo.addItem(T("mic_default", self._lang), None)
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    self.mic_combo.addItem(d["name"], i)
        except Exception:
            pass
        self.mic_combo.blockSignals(False)

    def _on_mic_change(self, idx):
        self._device = self.mic_combo.itemData(idx)

    # ── 語言切換 ────────────────────────────────────────────────────
    def _on_lang_change(self, code):
        if code == self._lang:
            return
        # 切語言時把上一次別的語言留下的逐字稿/摘要清掉,回到初始狀態,
        # 才不會出現「中文 UI 顯示英文摘要」之類的違和畫面。
        # (錄音/推論進行中時語言按鈕已被 disable,不會走到這。)
        self._lang = code
        self._raw_transcript = ""
        self._raw_summary = ""
        self._has_completed_round = False
        self.stage_t.reset()
        self.stage_s.reset()
        self._apply_language()

    def _apply_language(self):
        self.setWindowTitle(T("title", self._lang))
        self.lbl_title.setText(T("title", self._lang))
        self.lbl_badge.setText("● " + T("badge_offline", self._lang))
        for code, key in (("en", "lang_en"), ("zh", "lang_zh"), ("cn", "lang_cn")):
            self.lang_buttons[code].setText(T(key, self._lang))
            self.lang_buttons[code].setChecked(code == self._lang)
        self.lbl_mic.setText(T("mic_label", self._lang))
        if self.mic_combo.count() > 0:
            self.mic_combo.blockSignals(True)
            self.mic_combo.setItemText(0, T("mic_default", self._lang))
            self.mic_combo.blockSignals(False)
        self.stage_t.retranslate(self._lang)
        self.stage_s.retranslate(self._lang)
        self.card_tr.title.setText(T("card_transcript", self._lang))
        self.card_sm.title.setText(T("card_summary", self._lang))
        self.lbl_footer.setText(T("footer", self._lang))
        self._update_hint()
        self._render_transcript()
        self._render_summary()

    def _update_hint(self):
        if self._state == "idle":
            key = "hint_done_ready" if self._has_completed_round else "hint_idle"
            self.lbl_hint.setText(T(key, self._lang))
        elif self._state == "recording":
            self.lbl_hint.setText(T("hint_recording", self._lang))
        else:
            self.lbl_hint.setText(T("hint_processing", self._lang))

    def _set_lang_buttons_enabled(self, enabled):
        """跑錄音或推論時鎖住語言按鈕,避免 mid-cycle 切語言造成 whisper_lang
        跟 UI 對不上。"""
        for b in self.lang_buttons.values():
            b.setEnabled(enabled)

    def _tick_spinners(self):
        self.stage_t.tick()
        self.stage_s.tick()

    # ── 主流程:錄音 → 推論 ─────────────────────────────────────────
    def _on_mic_click(self):
        if self._state == "idle":
            self._start_recording()
        elif self._state == "recording":
            self._stop_recording()

    def _start_recording(self):
        # state == idle 才會進來,上一輪 QThread 都該結束了。賦值前先 detach
        # 舊參考,讓 Python GC 在 Qt 連線都散完後乾淨釋放(避免「QThread: Destroyed
        # while thread is still running」之類的警告)。
        self._recorder = None
        self._worker = None

        self._raw_transcript = ""
        self._raw_summary = ""
        self._render_transcript()
        self._render_summary()
        self.stage_t.reset()
        self.stage_s.reset()

        self._state = "recording"
        self.btn_mic.set_state(recording=True)
        self._set_lang_buttons_enabled(False)
        self._update_hint()

        out_path = os.path.join(os.getcwd(), "meeting.wav")
        self._recorder = RecorderThread(self._device, out_path)
        self._recorder.finished_ok.connect(self._on_recording_done)
        self._recorder.failed.connect(self._on_recording_failed)
        self._recorder.start()

    def _stop_recording(self):
        if self._recorder is not None:
            self._recorder.request_stop()

    @Slot(str)
    def _on_recording_done(self, wav_path):
        self._state = "processing"
        self.btn_mic.set_state(recording=False, busy=True)
        self._update_hint()

        whisper_lang = "en" if self._lang == "en" else "zh"
        self._worker = InferenceWorker(wav_path, whisper_lang)
        self._worker.transcribe_started.connect(self.stage_t.start)
        self._worker.transcribe_done.connect(self._on_transcribe_done)
        self._worker.summarize_started.connect(self.stage_s.start)
        self._worker.summarize_done.connect(self._on_summarize_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @Slot(str)
    def _on_recording_failed(self, why):
        self._state = "idle"
        self.btn_mic.set_state(recording=False, busy=False)
        self._set_lang_buttons_enabled(True)
        if why == "no_audio":
            self.lbl_hint.setText(T("err_no_audio", self._lang))
        else:
            self.lbl_hint.setText(T("err_generic", self._lang, e=why))

    @Slot(str, float)
    def _on_transcribe_done(self, text, elapsed):
        self.stage_t.done(elapsed)
        self._raw_transcript = text or ""
        self._render_transcript()

    @Slot(str, float)
    def _on_summarize_done(self, text, elapsed):
        self.stage_s.done(elapsed)
        self._raw_summary = text or ""
        self._render_summary()

    @Slot(str)
    def _on_worker_failed(self, msg):
        self.lbl_hint.setText(T("err_generic", self._lang, e=msg))

    def _on_worker_finished(self):
        self._has_completed_round = True
        self._state = "idle"
        self.btn_mic.set_state(busy=False)
        self._set_lang_buttons_enabled(True)
        self._update_hint()

    # ── 結果渲染 ────────────────────────────────────────────────────
    def _render_transcript(self):
        if not self._raw_transcript:
            self.tr_view.setPlainText(T("empty_transcript", self._lang))
        else:
            self.tr_view.setPlainText(localize_cjk(self._raw_transcript, self._lang))

    def _render_summary(self):
        # 三個 section header 任何時候都先把字補上(語言切換時也要更新)
        for key, label_key in SECTION_KEYS:
            head, _body, _block = self.sec_widgets[key]
            head.setText(T(label_key, self._lang))

        if not self._raw_summary:
            self.sm_scroll.show()
            self.sm_fallback.hide()
            self.sm_placeholder.show()
            self.sm_placeholder.setText(T("empty_summary", self._lang))
            for key, _ in SECTION_KEYS:
                _h, _b, block = self.sec_widgets[key]
                block.hide()
            return

        text = localize_cjk(self._raw_summary, self._lang)
        parsed = parse_summary(text)
        if parsed is None:
            # 模型脫序:整段塞到 fallback,不分區
            self.sm_scroll.hide()
            self.sm_fallback.show()
            self.sm_fallback.setPlainText(text)
            return

        self.sm_fallback.hide()
        self.sm_scroll.show()
        self.sm_placeholder.hide()
        for key, _ in SECTION_KEYS:
            _h, body, block = self.sec_widgets[key]
            body.setText(parsed[key] if parsed[key] else "—")
            block.show()

    # ── 關閉視窗:錄音中要先停 ──────────────────────────────────────
    def closeEvent(self, ev):
        if self._recorder is not None and self._recorder.isRunning():
            self._recorder.request_stop()
            self._recorder.wait(1000)
        # 推論 worker 還在跑就交給 atexit(transcribe/summarize 各自有掛)收尾
        ev.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
