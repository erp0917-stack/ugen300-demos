# -*- coding: utf-8 -*-
"""
hailo_detect.py
===============
這是「眼睛」：叫 UGen300 跑 YOLOv8 物件偵測，
從一張畫面中找出物件（人、手機、杯子…），回傳每個物件的框和名稱。

已對接完成：InferModel API + 內建 NMS 的 YOLO .hef（yolov8 / yolov11 系列皆可），
輸出直接是「每個物件 (類別, 信心度, 框)」。

   官方參考：github.com/hailo-ai/Hailo-Application-Code-Examples
   → runtime/python/object_detection
"""

import os
import numpy as np
import cv2
from coco_labels import COCO_LABELS

try:
    # 新版 InferModel 路徑（Hailo-10H / USB 用這套）只需要這幾個
    from hailo_platform import VDevice, HEF, FormatType
    _HAS_HAILO = True
except Exception:
    _HAS_HAILO = False


def _looks_nested(seq):
    """判斷一個 list 的第一個元素本身是否還是一組（用來辨識有沒有 batch 維度）。"""
    try:
        first = seq[0]
        return isinstance(first, (list, tuple, np.ndarray))
    except Exception:
        return False


class ObjectDetector:
    """
    用法：
        det = ObjectDetector("yolov8m.hef")
        results = det.infer(frame)   # frame 是 OpenCV BGR 影像
        # results 是 list，每個元素 (label, score, (x1, y1, x2, y2))，座標已換算回原圖
    """

    def __init__(self, hef_path, conf_threshold=0.5, timeout_ms=None):
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        self.timeout_ms = timeout_ms
        self.input_w = 640
        self.input_h = 640

        if not _HAS_HAILO:
            print("[hailo_detect] 尚未安裝 hailo_platform，目前為「讀結構模式」，infer() 回傳空清單。")
            self._ready = False
            return

        # ===== 載入模型到 UGen300（InferModel 新版 API，對應 USB / Hailo-10H）=====
        # 實機確認：VDevice() 可開、VDevice 有 create_infer_model、裝置為 usb/xxx。
        # 舊的 ConfigureParams + InferVStreams 不支援此 USB 裝置，改用 InferModel。
        import hailo_vdevice
        self.vdevice = hailo_vdevice.get()  # 單例 + 自動重試
        self.infer_model = hailo_vdevice.create_infer_model(hef_path)   # 含重試與保活
        if self.timeout_ms is None: self.timeout_ms = hailo_vdevice.RUN_TIMEOUT_MS
        try:
            self.infer_model.set_batch_size(1)
        except Exception as e:
            print(f"[hailo_detect] set_batch_size 略過：{e}")

        # 輸入餵 UINT8（相機影像）、輸出取 FLOAT32（方便解析）
        try:
            self.infer_model.input().set_format_type(FormatType.UINT8)
        except Exception as e:
            print(f"[hailo_detect] input set_format_type 略過：{e}")
        try:
            self.infer_model.output().set_format_type(FormatType.FLOAT32)
        except Exception as e:
            print(f"[hailo_detect] output set_format_type 略過：{e}")

        # 讀輸入大小（通常 640x640x3）
        try:
            ishape = tuple(self.infer_model.input().shape)
            if len(ishape) >= 2:
                self.input_h, self.input_w = int(ishape[0]), int(ishape[1])
        except Exception as e:
            print(f"[hailo_detect] 讀 input shape 失敗，沿用預設 640：{e}")

        # 記住輸出 shape（給推論時配置輸出緩衝區）
        try:
            self.output_shape = tuple(self.infer_model.output().shape)
        except Exception:
            self.output_shape = None

        # 設定（configure）一次，之後重複跑推論
        self.configured = self.infer_model.configure(); hailo_vdevice.keep(self.configured)

        self._ready = True
        print(f"[hailo_detect] UGen300 模型載入完成（InferModel），"
              f"輸入 {self.input_w}x{self.input_h}，輸出 shape {self.output_shape}")

    def _preprocess(self, frame):
        """縮放成模型輸入大小，回傳 (H,W,3) uint8 連續陣列（InferModel 餵單張）"""
        self._orig_h, self._orig_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def infer(self, frame):
        """丟一張畫面，回傳偵測到的物件清單（InferModel 推論）"""
        if not self._ready:
            return []

        input_data = self._preprocess(frame)

        # 用 InferModel 跑一次同步推論
        bindings = self.configured.create_bindings()
        bindings.input().set_buffer(input_data)

        # 配置輸出緩衝區
        out_shape = self.output_shape if self.output_shape else (self.infer_model.output().shape)
        out_buf = np.zeros(out_shape, dtype=np.float32)
        bindings.output().set_buffer(out_buf)

        self.configured.run([bindings], self.timeout_ms)

        raw = bindings.output().get_buffer()
        return self._parse_output(raw)

    def _parse_output(self, raw):
        """
        ✅ 已對接：把 UGen300 的 YOLOv8（內建 NMS）輸出解析成
                  [(label, score, (x1,y1,x2,y2)), ...]

        Hailo 官方 yolov8m.hef 多半帶「內建 NMS 後處理」，
        infer() 回傳的每個框原始格式為 (y_min, x_min, y_max, x_max, score)，
        座標已正規化到 0~1。這裡轉成 (x1, y1, x2, y2) 再交給
        _scale_box() 換算回原圖像素。

        本函式對三種常見輸出形狀都能解析（實機跑起來框不出來時看下方備註）：
          A. 物件陣列 / list：output[0] 是「每類一組」的清單（最常見）
          B. ndim==4 的 ndarray：(batch, num_classes, max_dets, 5)
          C. ndim==3 的 ndarray：(num_classes, max_dets, 5)
        """
        detections = []

        # raw 是 dict：{輸出層名稱: 值}，取出唯一（或第一個）輸出
        if isinstance(raw, dict):
            if not raw:
                return []
            output = next(iter(raw.values()))
        else:
            output = raw

        # 取出「這一張畫面、每個類別一組偵測」的結構：per_class
        if isinstance(output, np.ndarray) and output.dtype == object:
            # A. 物件陣列：output[0] 是長度=類別數的 list
            per_class = output[0]
        elif isinstance(output, np.ndarray) and output.ndim == 4:
            # B. (batch, num_classes, max_dets, 5) → 取 batch 0
            per_class = output[0]
        elif isinstance(output, np.ndarray) and output.ndim == 3:
            # C. (num_classes, max_dets, 5)
            per_class = output
        else:
            # 其他（純 list）：直接當「每類一組」用
            per_class = output[0] if (len(output) == 1 and _looks_nested(output)) else output

        for class_id, class_dets in enumerate(per_class):
            if class_dets is None:
                continue
            arr = np.asarray(class_dets, dtype=np.float32)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, arr.shape[-1])  # 統一成 (n_dets, 5)
            for row in arr:
                if row.shape[0] < 5:
                    continue
                ymin, xmin, ymax, xmax, score = (
                    float(row[0]), float(row[1]),
                    float(row[2]), float(row[3]), float(row[4]),
                )
                if score < self.conf_threshold:
                    continue
                # 原始為 (ymin, xmin, ymax, xmax) → 轉成 (x1, y1, x2, y2)
                box = self._scale_box((xmin, ymin, xmax, ymax))
                detections.append((self.label_name(class_id), score, box))

        return detections

    def _scale_box(self, box_norm):
        """已寫好：把模型座標換算回原圖像素 (x1,y1,x2,y2)"""
        x1, y1, x2, y2 = box_norm
        sx = self._orig_w if x1 <= 1.0 else self._orig_w / self.input_w
        sy = self._orig_h if y1 <= 1.0 else self._orig_h / self.input_h
        return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    @staticmethod
    def label_name(class_id):
        """class_id 轉成名稱"""
        if 0 <= class_id < len(COCO_LABELS):
            return COCO_LABELS[class_id]
        return f"id_{class_id}"

    def close(self):
        # 釋放 InferModel 的 configured 與 VDevice
        # 2026-09-10:Windows 9 月更新後 USB close 會失敗並讓 UGen300 掉線,
        # 預設不主動 release,讓程序結束時裝置自行重新列舉(實測可存活)。
        # 需要舊行為時設環境變數 HAILO_RELEASE_VDEVICE=1。
        if os.environ.get("HAILO_RELEASE_VDEVICE") != "1":
            return
        try:
            cim = getattr(self, "configured", None)
            if cim is not None and hasattr(cim, "__exit__"):
                cim.__exit__(None, None, None)
        except Exception:
            pass
        # VDevice 由 hailo_vdevice.release() 在程序結尾統一處理,這裡不碰
