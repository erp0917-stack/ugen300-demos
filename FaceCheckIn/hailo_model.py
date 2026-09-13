# -*- coding: utf-8 -*-
"""
hailo_model.py —— 通用多輸出 HEF 包裝(InferModel API,UGen300 / Hailo-10H;同名 HEF 在 Hailo-8L 也適用)。

    m = HailoModel("scrfd_10g.hef")
    outs = m.infer(rgb_uint8_HWC)     # 回傳 {輸出層名稱: float32 ndarray(去掉 batch 維)}
    m.input_h, m.input_w              # 模型輸入大小
"""
import numpy as np
from hailo_platform import HEF, FormatType

import hailo_vdevice


class HailoModel:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.vdevice = hailo_vdevice.get()
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        try: self.infer_model.set_batch_size(1)
        except Exception: pass
        self.output_names = [o.name for o in self.hef.get_output_vstream_infos()]
        self.infer_model.input().set_format_type(FormatType.UINT8)
        for n in self.output_names:
            self.infer_model.output(n).set_format_type(FormatType.FLOAT32)
        ishape = tuple(self.infer_model.input().shape)
        self.input_h, self.input_w = int(ishape[0]), int(ishape[1])
        self.output_shapes = {n: tuple(self.infer_model.output(n).shape) for n in self.output_names}
        self.configured = self.infer_model.configure()

    def infer(self, rgb):
        """rgb: (input_h, input_w, 3) uint8 連續陣列。"""
        b = self.configured.create_bindings()
        b.input().set_buffer(np.ascontiguousarray(rgb, dtype=np.uint8))
        bufs = {}
        for n in self.output_names:
            bufs[n] = np.zeros(self.output_shapes[n], dtype=np.float32); b.output(n).set_buffer(bufs[n])
        try: self.configured.run([b], 10000)
        except TypeError: self.configured.run([b])
        out = {}
        for n in self.output_names:
            a = np.asarray(b.output(n).get_buffer(), dtype=np.float32)
            if a.ndim == 4 and a.shape[0] == 1: a = a[0]
            out[n] = a
        return out

    def close(self):
        pass  # 不 release(見 hailo_vdevice.py)
