"""
llm_engine.py —— 離線 LLM 引擎:直接用 hailo_platform.genai.LLM 載 .hef,逐 token 串流。
(OfflineChat 與 OfflineCode 共用;兩份請保持一致)

不走 hailo-ollama(其 qwen3 需上網 pull);VDevice 用 hailo_vdevice 單例,結束不 release。
純邏輯部分(think 標籤切分、tok/s 統計、系統提示組裝、簡轉繁)獨立成函式,可離線測試。

「資料怎麼保持最新」:模型知識凍結在訓練截止日,所以每次對話都把「今天日期 + 知識截止」寫進系統提示,
讓它會算日期、不會瞎編近況;真正的新資料要靠本機文件檢索(RAG),不在這一版範圍。
"""
import os
import re
import threading
import time
from datetime import datetime

import hailo_vdevice

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "..", "models")
MODELS = {
    "Qwen3-1.7B":         dict(hef=os.path.join(_MODELS_DIR, "Qwen3-1.7B-Instruct.hef"),         tps="~5 tok/s",  think=True,  cutoff="2024 年底", kind="chat"),
    "Llama3.2-1B":        dict(hef=os.path.join(_MODELS_DIR, "Llama3.2-1B-Instruct.hef"),        tps="~10 tok/s", think=False, cutoff="2023 年底", kind="chat"),
    "Qwen2.5-1.5B":       dict(hef=os.path.join(_MODELS_DIR, "Qwen2.5-1.5B-Instruct.hef"),       tps="~7 tok/s",  think=False, cutoff="2024 年中", kind="chat"),
    "Qwen2.5-Coder-1.5B": dict(hef=os.path.join(_MODELS_DIR, "Qwen2.5-Coder-1.5B-Instruct.hef"), tps="~8 tok/s",  think=False, cutoff="2024 年中", kind="code"),
}
DEFAULT_MODEL = "Qwen3-1.7B"
DEFAULT_CODE_MODEL = "Qwen2.5-Coder-1.5B"
STOP_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>")
WEEKDAYS = "一二三四五六日"

CHAT_PROMPT = ("你是一個在本機離線執行的 AI 助理(跑在 UGen300 USB AI 加速器上)。"
               "一律使用繁體中文(台灣用語)回答,不要出現簡體字,簡潔直接。")
CODE_PROMPT = ("你是一個在本機離線執行的程式助理。回答一律用繁體中文(台灣用語)簡短說明,"
               "程式碼一律放在 ```python(或對應語言)區塊內,只給一個完整可執行的區塊,程式裡的註解用英文;"
               "沒有指定語言時用 Python。不要重複題目,不要多餘客套。")


# ---------- 純邏輯(可離線測試) ----------
def date_context(now=None, cutoff="2024 年底"):
    """每次對話都注入:今天日期、星期、以及模型知識截止日。"""
    now = now or datetime.now()
    return (f"今天是 {now:%Y-%m-%d} 星期{WEEKDAYS[now.weekday()]},現在時間 {now:%H:%M}(台北)。"
            f"你的訓練資料只到 {cutoff}。只有在被問到 {cutoff} 之後發生的新聞、事件、新產品時,才回答「我的資料只到 {cutoff},這個我不知道」;"
            f"其他一般知識、解釋、建議、寫作、程式問題,都要正常完整地回答。")


def build_system_prompt(kind="chat", model=DEFAULT_MODEL, now=None):
    cfg = MODELS.get(model, {})
    base = CODE_PROMPT if kind == "code" else CHAT_PROMPT
    return base + "\n" + date_context(now, cfg.get("cutoff", "2024 年底"))


try:
    from opencc import OpenCC
    _CC = OpenCC("s2t")       # 只轉字、不做台灣異體(s2tw 會把「台灣」改「臺灣」;s2twp 會把「演算法」改「演演算法」)
except Exception:  # noqa: BLE001
    _CC = None
# 常見大陸用語 → 台灣用語(在 s2tw 之後套用,鍵是轉字後的寫法);刻意不收「支持/訪問/項目」這類兩岸都用的詞
_TW_WORDS = [("軟件", "軟體"), ("硬件", "硬體"), ("數據", "資料"), ("網絡", "網路"), ("代碼", "程式碼"), ("默認", "預設"),
             ("質量", "品質"), ("優化", "最佳化"), ("視頻", "影片"), ("屏幕", "螢幕"), ("內存", "記憶體"), ("文件夾", "資料夾"),
             ("用戶", "使用者"), ("打印", "列印"), ("登錄", "登入"), ("界面", "介面"), ("字體", "字型"), ("信息", "資訊"),
             ("服務器", "伺服器"), ("計算機", "電腦"), ("鼠標", "滑鼠"), ("硬盤", "硬碟"), ("程序員", "程式設計師"),
             ("臺", "台")]   # OpenCC 連 s2t 都會把「台」轉成「臺」,台灣慣用「台」


def to_traditional(text):
    """簡轉繁 + 台灣用語;沒裝 opencc 就原樣回傳。程式碼不要丟進來(識別字會被改)。已是繁體的字不會被改壞。"""
    if not text: return text
    if _CC: text = _CC.convert(text)
    for a, b in _TW_WORDS: text = text.replace(a, b)
    text = re.sub(r"(?<![演計])算法", "演算法", text)       # 「算法」→「演算法」;「演算法」「計算法則」不動
    return text


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
                    keep = len(self.CLOSE) - 1
                    emit, self.buf = (self.buf[:-keep], self.buf[-keep:]) if len(self.buf) > keep else ("", self.buf)
                    td += emit; self.think += emit; break
                td += self.buf[:i]; self.think += self.buf[:i]
                self.buf = self.buf[i + len(self.CLOSE):]; self.in_think = False
            else:
                i = self.buf.find(self.OPEN)
                if i < 0:
                    keep = len(self.OPEN) - 1
                    emit, self.buf = (self.buf[:-keep], self.buf[-keep:]) if len(self.buf) > keep else ("", self.buf)
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

    def start(self): self.t0 = time.monotonic(); self.n = 0; self.t_first = None
    def tick(self):
        self.n += 1
        if self.t_first is None: self.t_first = time.monotonic()
    @property
    def tps(self):
        if not self.t_first or self.n < 2: return 0.0
        return (self.n - 1) / max(1e-6, time.monotonic() - self.t_first)
    @property
    def ttft(self):
        return (self.t_first - self.t0) if (self.t0 and self.t_first) else 0.0


def clean_answer(text):
    """去掉模型偶爾殘留的結尾標記。"""
    text = re.sub(r"<\|im_end\|>|<\|eot_id\|>|<\|end_of_text\|>", "", text)
    return re.sub(r"<\|[A-Za-z_]*$", "", text).strip()      # 被切成兩個 token 的殘缺結尾標記(<|im)也砍掉


# ---------- 引擎(需要裝置) ----------
class LLMEngine:
    def __init__(self, kind="chat"):
        self.kind = kind
        self.model_name = None; self.llm = None; self.vd = None
        self.history = []           # [{"role","content"}]
        self._lock = threading.Lock()   # 只保護載入/重設,不跨越串流
        self._busy = False; self._discard = False
        self.rate = TokRate()

    def load(self, name=None, on_status=None):
        from hailo_platform.genai import LLM
        name = name or (DEFAULT_CODE_MODEL if self.kind == "code" else DEFAULT_MODEL)
        cfg = MODELS[name]
        if not os.path.exists(cfg["hef"]):
            raise FileNotFoundError(f"找不到模型檔:{cfg['hef']}")
        with self._lock:
            if self.llm is not None:
                raise RuntimeError("同一個程序不能重載模型(HailoRT 5.3.2 會 INTERNAL_FAILURE),請重啟程式")
            if on_status: on_status(f"載入 {name}…")
            self.vd = hailo_vdevice.get()
            t = time.monotonic(); last = None
            for i in range(3):
                try:
                    self.llm = LLM(self.vd, cfg["hef"]); hailo_vdevice.keep(self.llm); break
                except Exception as e:  # noqa: BLE001  前一個程序剛硬退出時裝置端 session 未收,會撞逾時
                    last = e
                    if not (hailo_vdevice.is_timeout(e) or any(k in str(e) for k in hailo_vdevice._TRANSIENT)): raise
                    if on_status: on_status(f"裝置忙碌,{i + 1}/3 重試…")
                    time.sleep(2.0)
            if self.llm is None: raise last
            self.model_name = name; self.history = []
            if on_status: on_status(f"{name} 載入完成({time.monotonic() - t:.1f}s)")
        return self

    def reset(self):
        with self._lock:
            self.history = []
            if self._busy:
                self._discard = True; return  # 生成中不動 context,等這一則結束後也不把它塞回 history
            if self.llm is not None:
                try: self.llm.clear_context()
                except Exception: pass

    def stream(self, user_text, max_tokens=512, temperature=None):
        """產生器:逐段 yield (kind, text),kind ∈ {'think','answer'}。結束前會 flush。"""
        assert self.llm is not None, "模型尚未載入"
        if temperature is None: temperature = 0.2 if self.kind == "code" else 0.4
        # 注意:不在持鎖狀態下 yield(否則 UI 在生成中按「清除」會死等);busy 檢查、history 操作、clear_context 都在鎖內
        with self._lock:
            if self._busy:
                raise RuntimeError("上一則還在生成中")
            self._busy = True; self._discard = False
            user_msg = {"role": "user", "content": user_text}
            self.history.append(user_msg)
            recent = self.history[-8:]
            while recent and recent[0]["role"] != "user": recent = recent[1:]      # 切片不能以 assistant 開頭
            prompt = [{"role": "system", "content": build_system_prompt(self.kind, self.model_name)}] + recent
            try: self.llm.clear_context()
            except Exception: pass
        sp = ThinkSplitter(); self.rate.start(); full = ""
        try:
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
        with self._lock:
            if self._discard or not self.history or self.history[-1] is not user_msg:
                self.history = []; self._discard = False      # 生成中被清除:這一則不留
            else:
                self.history.append({"role": "assistant", "content": clean_answer(sp.answer)})

    def ask_all(self, user_text, max_tokens=256):
        """非串流版(自測用)。"""
        parts = {"think": "", "answer": ""}
        for k, t in self.stream(user_text, max_tokens=max_tokens):
            parts[k] += t
        return clean_answer(parts["answer"]), parts["think"]
