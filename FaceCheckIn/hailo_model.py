# -*- coding: utf-8 -*-
"""
hailo_model.py —— 通用多輸出 HEF 包裝(InferModel API,UGen300 / Hailo-10H;同名 HEF 在 Hailo-8L 也適用)。

    m = HailoModel("scrfd_10g.hef")
    outs = m.infer(rgb_uint8_HWC)     # 回傳 {輸出層名稱: float32 ndarray(去掉 batch 維)}
    m.input_h, m.input_w              # 模型輸入大小

所有 HailoRT 物件建好即交給 hailo_vdevice.keep() 保活(見該檔說明);推論逾時視為不可恢復(m.broken=True)。
hailo_platform 延遲到建構時才 import,離線測試不需要 HailoRT。
"""
import numpy as np

import hailo_vdevice


class HailoModel:
    def __init__(self, hef_path, timeout_ms=None):
        from hailo_platform import HEF, FormatType
        self.hef = HEF(hef_path)
        self.vdevice = hailo_vdevice.get()
        self.infer_model = hailo_vdevice.create_infer_model(hef_path)      # 含重試與保活
        self.timeout_ms = hailo_vdevice.RUN_TIMEOUT_MS if timeout_ms is None else timeout_ms
        self.broken = False
        try: self.infer_model.set_batch_size(1)
        except Exception: pass
        self.output_names = [o.name for o in self.hef.get_output_vstream_infos()]
        self.infer_model.input().set_format_type(FormatType.UINT8)
        for n in self.output_names:
            self.infer_model.output(n).set_format_type(FormatType.FLOAT32)
        ishape = tuple(self.infer_model.input().shape)
        self.input_h, self.input_w = int(ishape[0]), int(ishape[1])
        self.output_shapes = {n: tuple(self.infer_model.output(n).shape) for n in self.output_names}
        self.configured = self.infer_model.configure(); hailo_vdevice.keep(self.configured)
        self._make_bindings()

    def _make_bindings(self):
        """bindings 與輸出緩衝建一次重複使用(省每幀配置成本);出錯後可換一組乾淨的。"""
        self._bindings = self.configured.create_bindings(); hailo_vdevice.keep(self._bindings)
        self._bufs = {n: np.zeros(self.output_shapes[n], dtype=np.float32) for n in self.output_names}
        for n, buf in self._bufs.items(): self._bindings.output(n).set_buffer(buf)

    def infer(self, rgb):
        """rgb: (input_h, input_w, 3) uint8。回傳的陣列是內部緩衝的複本,呼叫端可放心保存。"""
        if self.broken: raise RuntimeError("此模型先前推論逾時,已停用")
        if rgb.shape[:2] != (self.input_h, self.input_w):
            raise ValueError(f"輸入大小 {rgb.shape[:2]} 不符模型 {(self.input_h, self.input_w)}")
        self._bindings.input().set_buffer(np.ascontiguousarray(rgb, dtype=np.uint8))
        try:
            self.configured.run([self._bindings], self.timeout_ms)
        except Exception as e:
            if hailo_vdevice.is_timeout(e): self.broken = True     # 舊 job 可能還在寫 _bufs,這組永不再用
            else: self._make_bindings()                            # 其他錯誤:換一組乾淨的讓上層重試
            raise
        out = {}
        for n in self.output_names:
            a = np.array(self._bindings.output(n).get_buffer(), dtype=np.float32, copy=True)
            if a.ndim == 4 and a.shape[0] == 1: a = a[0]
            out[n] = a
        return out

    def close(self):
        pass  # 不 release(見 hailo_vdevice.py)
