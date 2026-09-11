"""
photo_restore.py —— 老照片修復站:拖照片進來 → 勾選 提亮/去雜訊/放大 → UGen300 修復 → 前後對比滑桿 → 另存。

用法:python photo_restore.py [--image 路徑]
      python photo_restore.py --selftest                    (載三個模型、修一張、印耗時)
      python photo_restore.py --image x.jpg --snapshot out.png   (自動修復後截圖離開)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QCheckBox, QSlider, QFileDialog, QFrame, QSizePolicy)

from restore_engine import RestoreEngine, MODELS, ORDER

STYLE = """
QMainWindow, QWidget { background: #1b1b1f; color: #ececec; font-family: "Microsoft JhengHei UI", "Microsoft JhengHei"; font-size: 16px; }
QLabel#title { font-size: 24px; font-weight: 700; }
QLabel#sub { color: #9aa0a6; font-size: 14px; }
QLabel#offline { color: #37d67a; font-weight: 700; border: 1px solid #37d67a; border-radius: 6px; padding: 3px 8px; font-size: 14px; }
QLabel#status { color: #ffa028; font-size: 15px; }
QPushButton { background: #2a2a30; border: 1px solid #3a3a42; border-radius: 8px; padding: 10px 14px; }
QPushButton#go { background: #2f6fed; border: none; font-weight: 700; font-size: 18px; }
QPushButton:disabled { color: #777; }
QCheckBox { font-size: 17px; spacing: 8px; } QCheckBox::indicator { width: 20px; height: 20px; }
QFrame#panel { background: #232327; border-radius: 12px; }
"""


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class CompareView(QWidget):
    """左邊原圖、右邊修復後,拖動分界線比較。"""
    def __init__(self):
        super().__init__()
        self.before = None; self.after = None; self.split = 0.5
        self.setMinimumSize(640, 480); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_images(self, before_bgr, after_bgr):
        self.before = QPixmap.fromImage(to_qimage(before_bgr)) if before_bgr is not None else None
        self.after = QPixmap.fromImage(to_qimage(after_bgr)) if after_bgr is not None else None
        self.update()

    def _rect(self, pm):
        W, H = self.width(), self.height()
        s = min(W / pm.width(), H / pm.height())
        w, h = int(pm.width() * s), int(pm.height() * s)
        return QRect((W - w) // 2, (H - h) // 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#111114"))
        if self.before is None:
            p.setPen(QColor("#777")); p.drawText(self.rect(), Qt.AlignCenter, "把照片拖進來,或按「開啟照片」"); return
        base = self.after if self.after is not None else self.before
        r = self._rect(base)
        p.drawPixmap(r, base)
        if self.after is not None:
            # 左半邊蓋上原圖(縮放到同一框)
            sx = int(r.x() + r.width() * self.split)
            src_w = int(self.before.width() * self.split)
            p.drawPixmap(QRect(r.x(), r.y(), sx - r.x(), r.height()), self.before, QRect(0, 0, src_w, self.before.height()))
            p.setPen(QPen(QColor("#ffa028"), 3)); p.drawLine(sx, r.y(), sx, r.y() + r.height())
            p.setPen(QColor("#fff")); p.drawText(r.x() + 12, r.y() + 30, "修復前"); p.drawText(r.x() + r.width() - 80, r.y() + 30, "修復後")

    def mouseMoveEvent(self, e):
        if self.before is None or not (e.buttons() & Qt.LeftButton): return
        r = self._rect(self.after or self.before)
        self.split = min(1.0, max(0.0, (e.position().x() - r.x()) / max(1, r.width()))); self.update()

    def mousePressEvent(self, e): self.mouseMoveEvent(e)


class Worker(QThread):
    status = Signal(str); done = Signal(object, float); failed = Signal(str)
    def __init__(self, engine, img, steps, strengths): super().__init__(); self.e, self.img, self.steps, self.strengths = engine, img, steps, strengths
    def run(self):
        try:
            t = time.time(); self.e.load(on_status=self.status.emit)
            out = self.e.pipeline(self.img, self.steps, self.strengths, on_status=self.status.emit)
            self.done.emit(out, time.time() - t)
        except Exception as ex:  # noqa: BLE001
            self.failed.emit(f"{type(ex).__name__}: {ex}")


class Win(QMainWindow):
    def __init__(self, image="", snapshot=""):
        super().__init__()
        self.engine = RestoreEngine(); self.img = None; self.out = None; self.snapshot = snapshot; self.path = ""
        self.setWindowTitle("UGen300 老照片修復站"); self.resize(1280, 800); self.setAcceptDrops(True)
        root = QWidget(); self.setCentralWidget(root); h = QHBoxLayout(root); h.setContentsMargins(16, 12, 16, 12); h.setSpacing(14)
        self.view = CompareView(); h.addWidget(self.view, 1)

        panel = QFrame(); panel.setObjectName("panel"); panel.setFixedWidth(340); v = QVBoxLayout(panel); v.setContentsMargins(18, 18, 18, 18); v.setSpacing(12)
        t = QLabel("老照片修復站"); t.setObjectName("title"); v.addWidget(t)
        s = QLabel("三個模型都在 UGen300 上跑\n不上傳、不排隊、一次買斷"); s.setObjectName("sub"); v.addWidget(s)
        b = QLabel("離線 · UGen300"); b.setObjectName("offline"); b.setAlignment(Qt.AlignCenter); v.addWidget(b)
        self.open_btn = QPushButton("開啟照片…"); self.open_btn.clicked.connect(self._open); v.addWidget(self.open_btn)
        v.addSpacing(6)
        self.cb = {}
        for k in ORDER:
            c = QCheckBox(MODELS[k]["label"]); c.setChecked(True); self.cb[k] = c; v.addWidget(c)
        v.addWidget(QLabel("提亮強度"))
        self.strength = QSlider(Qt.Horizontal); self.strength.setRange(20, 100); self.strength.setValue(65); v.addWidget(self.strength)
        self.go = QPushButton("開始修復"); self.go.setObjectName("go"); self.go.clicked.connect(self._go); self.go.setEnabled(False); v.addWidget(self.go)
        self.status = QLabel("等待照片"); self.status.setObjectName("status"); self.status.setWordWrap(True); v.addWidget(self.status)
        v.addStretch(1)
        self.save_btn = QPushButton("另存修復後照片…"); self.save_btn.clicked.connect(self._save); self.save_btn.setEnabled(False); v.addWidget(self.save_btn)
        tip = QLabel("拖動畫面中的橘線比較前後"); tip.setObjectName("sub"); v.addWidget(tip)
        h.addWidget(panel)
        if image:
            self._load(image)
            if snapshot:
                QTimer.singleShot(300, self._go)

    # ----- 檔案 -----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        p = e.mimeData().urls()[0].toLocalFile()
        if p: self._load(p)

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(self, "選擇照片", "", "圖片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if p: self._load(p)

    def _load(self, p):
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)  # 支援中文路徑
        if img is None:
            self.status.setText("讀不到這個檔案"); return
        # 太大的圖先縮到長邊 1600,避免放大後超過 3200
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            s = 1600 / max(h, w); img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        self.img, self.out, self.path = img, None, p
        self.view.set_images(img, None); self.go.setEnabled(True); self.save_btn.setEnabled(False)
        self.status.setText(f"已載入 {os.path.basename(p)}({w}×{h})")

    def _go(self):
        if self.img is None: return
        steps = {k for k, c in self.cb.items() if c.isChecked()}
        if not steps: self.status.setText("至少勾一個處理"); return
        self.go.setEnabled(False); self.open_btn.setEnabled(False)
        self.w = Worker(self.engine, self.img, steps, {"brighten": self.strength.value() / 100.0})
        self.w.status.connect(self.status.setText); self.w.done.connect(self._done); self.w.failed.connect(self._fail); self.w.start()

    def _done(self, out, dt):
        self.out = out; self.view.set_images(self.img, out)
        self.status.setText(f"完成!{dt:.1f} 秒 · 輸出 {out.shape[1]}×{out.shape[0]}")
        self.go.setEnabled(True); self.open_btn.setEnabled(True); self.save_btn.setEnabled(True)
        if self.snapshot:
            QTimer.singleShot(400, lambda: (self.grab().save(self.snapshot), print("[snapshot]", self.snapshot), QApplication.instance().quit()))

    def _fail(self, err):
        self.status.setText("修復失敗:" + err + "\n請確認 UGen300 已插上。"); self.go.setEnabled(True); self.open_btn.setEnabled(True)

    def _save(self):
        if self.out is None: return
        base, _ = os.path.splitext(self.path)
        p, _ = QFileDialog.getSaveFileName(self, "另存", base + "_修復.png", "PNG (*.png);;JPEG (*.jpg)")
        if p:
            ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", self.out)
            buf.tofile(p); self.status.setText("已儲存:" + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=""); ap.add_argument("--snapshot", default=""); ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        img = cv2.imread(args.image) if args.image else np.random.randint(0, 256, (480, 640, 3), np.uint8)
        e = RestoreEngine(); t = time.time(); e.load(on_status=print)
        out = e.pipeline(img, set(ORDER), {"brighten": 0.65}); print(f"[selftest] OK {time.time() - t:.1f}s -> {out.shape}"); return
    app = QApplication(sys.argv); app.setStyleSheet(STYLE)
    w = Win(args.image, args.snapshot); w.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
