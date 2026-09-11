"""
llm_engine.py —— 離線 LLM 引擎:直接用 hailo_platform.genai.LLM 載 .hef,逐 token 串流。

不走 hailo-ollama(其 qwen3 需上網 pull);VDevice 用 hailo_vdevice 單例,結束不 release。
純邏輯部分(think 標籤切分、tok/s 統計)獨立成函式,可離線測試。
"""
import os
import re
import threading
import time

import hailo_vdevice

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "Qwen3-1.7B":   dict(hef=os.path.join(_HERE, "..", "models", "Qwen3-1.7B-Instruct.hef"),   tps="~5 tok/s", think=True),
    "Llama3.2-1B":  dict(hef=os.path.join(_HERE, "..", "models", "Llama3.2-1B-Instruct.hef"),  tps="~10 tok/s", think=False),
    "Qwen2.5-1.5B": dict(hef=os.path.join(_HERE, "..", "models", "Qwen2.5-1.5B-Instruct.hef"), tps="~7 tok/s", think=False),
}
DEFAULT_MODEL = "Qwen3-1.7B"
STOP_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>")
SYSTEM_PROMPT = "你是一個在本機離線執行的 AI 助理。一律使用繁體中文(台灣用語)回答,不要出現簡體字,簡潔直接。"


# ---------- 純邏輯(可離線測試) ----------
class ThinkSplitter:
    """把串流文字切成 think / answer 兩段。Qwen3 會先輸出 <think>…</think> 再回答。"""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""; self.in_think = False; self.think = ""; self.answer = ""

    def feed(self, piece):
        """餵一段文字,回傳 (think_delta, answer_delta)。"""
        self.buf += piece
        td = ad = ""
        while self.buf:
            if self.in_think:
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    # 可能標籤被切一半:保留最後 len(CLOSE)-1 個字元
                    keep = len(self.CLOSE) - 1
                    emit, self.buf = self.buf[:-keep] if len(self.buf) > keep else "", self.buf[-keep:] if len(self.buf) > keep else self.buf
                    td += emit; self.think += emit; break
                td += self.buf[:i]; self.think += self.buf[:i]
                self.buf = self.buf[i + len(self.CLOSE):]; self.in_think = False
            else:
                i = self.buf.find(self.OPEN)
                if i < 0:
                    keep = len(self.OPEN) - 1
                    if len(self.buf) > keep:
                        emit, self.buf = self.buf[:-keep], self.buf[-keep:]
                    else:
                        emit, self.buf = "", self.buf
                    ad += emit; self.answer += emit; break
                ad += self.buf[:i]; self.answer += self.buf[:i]
                self.buf = self.buf[i + len(self.OPEN):]; self.in_think = True
        return td, ad

    def flush(self):
        """串流結束時把殘留 buffer 吐出。"""
        rest = self.buf; self.buf = ""
        if self.in_think:
            self.think += rest; return rest, ""
        self.answer += rest; return "", rest


class TokRate:
    def __init__(self):
        self.n = 0; self.t0 = None; self.t_first = None

    def start(self): self.t0 = time.time(); self.n = 0; self.t_first = None
    def tick(self):
        self.n += 1
        if self.t_first is None: self.t_first = time.time()
    @property
    def tps(self):
        if not self.t_first or self.n < 2: return 0.0
        return (self.n - 1) / max(1e-6, time.time() - self.t_first)
    @property
    def ttft(self):
        return (self.t_first - self.t0) if (self.t0 and self.t_first) else 0.0


def clean_answer(text):
    """去掉模型偶爾殘留的結尾標記。"""
    return re.sub(r"<\|im_end\|>|<\|eot_id\|>|<\|end_of_text\|>", "", text).strip()


# ---------- 引擎(需要裝置) ----------
class LLMEngine:
    def __init__(self):
        self.model_name = None; self.llm = None; self.vd = None
        self.history = []           # [{"role","content"}]
        self._lock = threading.Lock()   # 只保護載入/重設,不跨越串流
        self._busy = False
        self.rate = TokRate()

    def load(self, name=DEFAULT_MODEL, on_status=None):
        from hailo_platform.genai import LLM
        cfg = MODELS[name]
        if not os.path.exists(cfg["hef"]):
            raise FileNotFoundError(f"找不到模型檔:{cfg['hef']}")
        with self._lock:
            if self.llm is not None:
                try: self.llm.release()       # 只釋放模型層,VDevice 不動
                except Exception: pass
                self.llm = None
            if on_status: on_status(f"載入 {name}…")
            self.vd = hailo_vdevice.get()
            t = time.time()
            self.llm = LLM(self.vd, cfg["hef"])
            self.model_name = name; self.history = []
            if on_status: on_status(f"{name} 載入完成({time.time() - t:.1f}s)")
        return self

    def reset(self):
        self.history = []
        if self._busy:
            return  # 生成中不動 context,等這一則結束;歷史已清空
        with self._lock:
            if self.llm is not None:
                try: self.llm.clear_context()
                except Exception: pass

    def stream(self, user_text, max_tokens=512, temperature=0.4):
        """產生器:逐段 yield (kind, text),kind ∈ {'think','answer'}。結束前會 flush。"""
        assert self.llm is not None, "模型尚未載入"
        self.history.append({"role": "user", "content": user_text})
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        sp = ThinkSplitter(); self.rate.start(); full = ""
        # 注意:不在持鎖狀態下 yield(否則 UI 在生成中按「清除」會死等);以 busy 旗標防止重入
        if self._busy:
            raise RuntimeError("上一則還在生成中")
        self._busy = True
        try:
            try: self.llm.clear_context()
            except Exception: pass
            with self.llm.generate(prompt=prompt, max_generated_tokens=max_tokens,
                                   temperature=temperature, do_sample=True, top_p=0.9, seed=None) as gen:
                for tok in gen:
                    self.rate.tick(); piece = tok if isinstance(tok, str) else str(tok)
                    full += piece
                    # 模型偶爾把結尾標記當成文字吐出:當作停止符號,截掉並結束
                    if any(s in full for s in STOP_MARKERS):
                        cut = min(full.find(s) for s in STOP_MARKERS if s in full)
                        piece = piece[:max(0, len(piece) - (len(full) - cut))]
                        td, ad = sp.feed(piece)
                        if td: yield "think", td
                        if ad: yield "answer", ad
                        break
                    td, ad = sp.feed(piece)
                    if td: yield "think", td
                    if ad: yield "answer", ad
        finally:
            self._busy = False
        td, ad = sp.flush()
        if td: yield "think", td
        if ad: yield "answer", ad
        self.history.append({"role": "assistant", "content": clean_answer(sp.answer)})

    def ask_all(self, user_text, max_tokens=256):
        """非串流版(自測用)。"""
        parts = {"think": "", "answer": ""}
        for k, t in self.stream(user_text, max_tokens=max_tokens):
            parts[k] += t
        return clean_answer(parts["answer"]), parts["think"]
