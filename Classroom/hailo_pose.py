# -*- coding: utf-8 -*-
"""
hailo_pose.py  ——  UGen300 / Hailo-10H 姿態偵測（新版 InferModel API）
====================================================================
這是 Y2/Y3/Y4/Y5 共用的「眼睛」：叫 UGen300 跑 YOLOv8s-pose .hef，
回傳畫面中每個人的 17 個關鍵點 (x, y, confidence)。

兩個用法：
  pose = PoseEstimator("yolov8s_pose.hef")
  # 單人版（Y2/Y3/Y5 用）：回傳第一個人的 17 點，沒人回傳 None
  kps  = pose.infer(frame)
  # 多人版（Y4 舉手統計用）：回傳所有人的清單，每人 17 點
  people = pose.infer_multi(frame)

對接重點（已完成）：
  ✅ 走新版 InferModel API（VDevice().create_infer_model(...)），對應 USB / Hailo-10H
  ✅ 多輸出（9 層：3 個 scale × box / class / keypoints）
  ✅ YOLOv8 pose 完整解碼：DFL 軟最大化 → 還原 bbox + keypoints → NMS
  ✅ 座標換算回原圖

參考：Hailo-Application-Code-Examples / runtime/python/pose_estimation
      （decoder + NMS 演算法等同官方範例，我們把它內嵌進來，
       這樣 Y2~Y5 不用多裝 utils 檔案。）
"""

import os
import numpy as np
import cv2

try:
    # 新版 InferModel 路徑（Hailo-10H / USB 用這套）只需要這幾個
    from hailo_platform import VDevice, HEF, FormatType
    _HAS_HAILO = True
except Exception:
    _HAS_HAILO = False


# ── YOLOv8-pose 解碼用的固定參數（與官方範例一致）──────────────────
_REG_MAX = 15          # regression_length，加 1 得 16 個 bin（DFL）
_STRIDES = [8, 16, 32]  # 三個輸出 scale 對應的 stride
_KEYPOINTS = 17
_KP_CH = _KEYPOINTS * 3  # 51
_DET_CH = (_REG_MAX + 1) * 4  # 64

_NMS_IOU = 0.5   # Classroom:0.7 偶爾讓一人出兩副骨架(人數、舉手 +1),收緊到 0.5
_SCORE_PREFILTER = 0.001  # NMS 前的粗過濾門檻；最終以使用者 conf_threshold 為準
_MAX_DETECTIONS = 300


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _xywh2xyxy(xywh):
    out = np.empty_like(xywh)
    out[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    out[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    out[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    out[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return out


def _nms_indices(boxes, scores, iou_thresh):
    """純 numpy NMS，回傳保留下來的 index（已排序，分數高的在前）。"""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


class PoseEstimator:
    """
    用法：
        pose = PoseEstimator("yolov8s_pose.hef")
        kps  = pose.infer(frame)             # 單人：list[17] of (x,y,conf) 或 None
        people = pose.infer_multi(frame)     # 多人：list[人][17] of (x,y,conf)
    """

    def __init__(self, hef_path, conf_threshold=0.5):
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        self.input_w = 640
        self.input_h = 640

        if not _HAS_HAILO:
            print("[hailo_pose] 尚未安裝 hailo_platform，infer() 會回傳 None / []")
            self._ready = False
            return

        # ===== 用新版 InferModel API（Hailo-10H / USB）載入模型 =====
        self.hef = HEF(hef_path)
        import hailo_vdevice
        self.vdevice = hailo_vdevice.get()  # 單例 + 自動重試(見 hailo_vdevice.py)
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        try:
            self.infer_model.set_batch_size(1)
        except Exception as e:
            print(f"[hailo_pose] set_batch_size 略過：{e}")

        # 從 HEF 取得所有輸出層名稱（多輸出必須逐一綁定）
        try:
            self.output_names = [oi.name for oi in self.hef.get_output_vstream_infos()]
        except Exception as e:
            print(f"[hailo_pose] 讀 output vstream infos 失敗：{e}")
            self.output_names = []

        # 輸入餵 UINT8（相機影像），輸出取 FLOAT32（方便 numpy 後處理）
        try:
            self.infer_model.input().set_format_type(FormatType.UINT8)
        except Exception as e:
            print(f"[hailo_pose] input set_format_type 略過：{e}")
        for name in self.output_names:
            try:
                self.infer_model.output(name).set_format_type(FormatType.FLOAT32)
            except Exception as e:
                print(f"[hailo_pose] output({name}) set_format_type 略過：{e}")

        # 讀輸入大小（通常 640x640x3）
        try:
            ishape = tuple(self.infer_model.input().shape)
            if len(ishape) >= 2:
                self.input_h, self.input_w = int(ishape[0]), int(ishape[1])
        except Exception as e:
            print(f"[hailo_pose] 讀 input shape 失敗，沿用 640：{e}")

        # 記每個輸出的 shape（給推論時配置輸出緩衝區）
        self.output_shapes = {}
        for name in self.output_names:
            try:
                self.output_shapes[name] = tuple(self.infer_model.output(name).shape)
            except Exception:
                self.output_shapes[name] = None

        # 設定（configure）一次，之後重複跑推論
        self.configured = self.infer_model.configure()

        self._ready = True
        print(
            f"[hailo_pose] UGen300 模型載入完成（InferModel），"
            f"輸入 {self.input_w}x{self.input_h}，輸出層 {len(self.output_names)} 個"
        )

    # ----------------------------------------------------------
    def _preprocess(self, frame):
        """縮放成模型輸入大小，回傳 (H,W,3) uint8 連續陣列（InferModel 餵單張）。"""
        self._orig_h, self._orig_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def _run_inference(self, frame):
        """跑一次同步推論，回傳 {output_name: ndarray}（已加上 batch 維度）。"""
        input_data = self._preprocess(frame)

        bindings = self.configured.create_bindings()
        bindings.input().set_buffer(input_data)

        out_bufs = {}
        for name in self.output_names:
            shape = self.output_shapes.get(name)
            if shape is None:
                shape = tuple(self.infer_model.output(name).shape)
            buf = np.zeros(shape, dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            out_bufs[name] = buf

        try:
            self.configured.run([bindings], 2000)
        except TypeError:
            self.configured.run([bindings])

        # 確保每個輸出都有 batch 維度（不同 HailoRT 版本可能省略）
        raw = {}
        for name, buf in out_bufs.items():
            arr = bindings.output(name).get_buffer()
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            raw[name] = arr
        return raw

    # ----------------------------------------------------------
    def infer(self, frame):
        """單人版：回傳第一個人的 17 個 (x, y, conf)，找不到回傳 None。"""
        people = self.infer_multi(frame)
        return people[0] if people else None

    def infer_multi(self, frame):
        """多人版：回傳每個人 17 個 (x, y, conf) 的 list；沒人時回傳 []。"""
        if not self._ready:
            return []
        raw = self._run_inference(frame)
        return self._parse_output(raw)

    # ----------------------------------------------------------
    def _parse_output(self, raw_detections):
        """
        把 9 層輸出（3 scale × box/class/keypoint）依形狀對齊，
        做 DFL 解碼 + NMS，回傳 list[人][17] of (x, y, conf)。
        """
        # 用 shape 對照表抓對應 output（與官方範例同款，不依賴名稱）
        shape_to_arr = {}
        for arr in raw_detections.values():
            shape_to_arr[tuple(arr.shape)] = arr

        try:
            endnodes = [
                shape_to_arr[(1, 20, 20, _DET_CH)],
                shape_to_arr[(1, 20, 20, 1)],
                shape_to_arr[(1, 20, 20, _KP_CH)],
                shape_to_arr[(1, 40, 40, _DET_CH)],
                shape_to_arr[(1, 40, 40, 1)],
                shape_to_arr[(1, 40, 40, _KP_CH)],
                shape_to_arr[(1, 80, 80, _DET_CH)],
                shape_to_arr[(1, 80, 80, 1)],
                shape_to_arr[(1, 80, 80, _KP_CH)],
            ]
        except KeyError as e:
            print(f"[hailo_pose] 找不到預期輸出 shape：{e}；實際 shapes={list(shape_to_arr.keys())}")
            return []

        raw_boxes_3 = endnodes[0:7:3]
        cls_scores_3 = endnodes[1:8:3]
        kpts_3 = endnodes[2:9:3]

        # class scores 拼成 (1, total_anchors, 1)
        scores = np.concatenate(
            [np.reshape(s, (-1, s.shape[1] * s.shape[2], 1)) for s in cls_scores_3], axis=1
        )

        # 對應 20/40/80 的 stride（從大 stride 往小：32, 16, 8）
        strides = _STRIDES[::-1]

        decoded_xywh = None
        decoded_kpts = None

        for box_dist, kpts, stride in zip(raw_boxes_3, kpts_3, strides):
            shape_y = self.input_h // stride
            shape_x = self.input_w // stride
            grid_x = np.arange(shape_x) + 0.5
            grid_y = np.arange(shape_y) + 0.5
            gx, gy = np.meshgrid(grid_x, grid_y)
            ct_col = gx.flatten() * stride
            ct_row = gy.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)  # (H*W, 4)

            # DFL：把 64 個 channel 視為 4 邊 × 16 個 bin → softmax 加權平均
            box_dist = np.reshape(
                box_dist, (-1, box_dist.shape[1] * box_dist.shape[2], 4, _REG_MAX + 1)
            )
            reg_range = np.arange(_REG_MAX + 1)
            box_distance = _softmax(box_dist, axis=-1) * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = np.sum(box_distance, axis=-1) * stride
            # 還原成 xmin/ymin/xmax/ymax（左上負方向）
            box_distance = np.concatenate(
                [box_distance[:, :, :2] * (-1.0), box_distance[:, :, 2:]], axis=-1
            )
            decode_box = np.expand_dims(center, axis=0) + box_distance
            xmin = decode_box[:, :, 0]; ymin = decode_box[:, :, 1]
            xmax = decode_box[:, :, 2]; ymax = decode_box[:, :, 3]
            xywh = np.transpose(
                np.array([(xmin + xmax) / 2, (ymin + ymax) / 2,
                          xmax - xmin, ymax - ymin]),
                (1, 2, 0),
            )
            decoded_xywh = xywh if decoded_xywh is None else np.concatenate(
                [decoded_xywh, xywh], axis=1
            )

            # keypoints：先把通道 reshape 成 (1, H*W, 17, 3)，再做位置回推
            kpts = np.reshape(kpts, (-1, kpts.shape[1] * kpts.shape[2], _KEYPOINTS, 3)).copy()
            kpts[..., :2] = 2.0 * kpts[..., :2]
            kpts[..., :2] = stride * (kpts[..., :2] - 0.5) + np.expand_dims(
                center[..., :2], axis=1
            )
            decoded_kpts = kpts if decoded_kpts is None else np.concatenate(
                [decoded_kpts, kpts], axis=1
            )

        # 拼成 (1, total_anchors, 4 + 1 + 51)
        decoded_kpts_flat = np.reshape(decoded_kpts, (1, -1, _KP_CH))
        predictions = np.concatenate([decoded_xywh, scores, decoded_kpts_flat], axis=2)

        # 取第一張 batch、依分數粗篩
        x = predictions[0]
        conf = x[:, 4]
        keep_mask = conf > _SCORE_PREFILTER
        x = x[keep_mask]
        if x.shape[0] == 0:
            return []

        xyxy = _xywh2xyxy(x[:, :4])
        scores_kept = x[:, 4]
        # 先按分數取前 _MAX_DETECTIONS，再做 NMS
        order = scores_kept.argsort()[::-1][:_MAX_DETECTIONS]
        xyxy = xyxy[order]
        scores_kept = scores_kept[order]
        kpts_kept = x[order, 5:5 + _KP_CH].reshape(-1, _KEYPOINTS, 3)

        keep = _nms_indices(xyxy, scores_kept, _NMS_IOU)
        if len(keep) == 0:
            return []
        scores_final = scores_kept[keep]
        kpts_final = kpts_kept[keep]

        # 套使用者的 conf_threshold（針對「人」的整體可信度）
        sel = scores_final >= self.conf_threshold
        if not sel.any():
            return []
        scores_final = scores_final[sel]
        kpts_final = kpts_final[sel]

        # 分數降序排
        order2 = scores_final.argsort()[::-1]
        kpts_final = kpts_final[order2]

        # 模型空間 → 原圖（直接縮放，沒有 letterbox）
        sx = self._orig_w / float(self.input_w)
        sy = self._orig_h / float(self.input_h)

        people = []
        for person in kpts_final:
            person_kp = []
            for j in range(_KEYPOINTS):
                x_m = float(person[j, 0])
                y_m = float(person[j, 1])
                vis_logit = float(person[j, 2])
                conf_kp = 1.0 / (1.0 + np.exp(-vis_logit))  # sigmoid
                person_kp.append((x_m * sx, y_m * sy, float(conf_kp)))
            people.append(person_kp)
        return people

    # ----------------------------------------------------------
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
        try:
            vd = getattr(self, "vdevice", None)
            if vd is not None and hasattr(vd, "release"):
                vd.release()
        except Exception:
            pass
