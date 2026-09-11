# -*- coding: utf-8 -*-
"""
outfit_advisor.py  ——  穿搭大師（V2）
====================================================
站到筆電鏡頭前（站遠一點讓鏡頭照到全身）→ 按鍵 → 15 秒倒數 → VLM 看穿著，
逐字串流印出「穿搭建議」（三段式）。全程離線，影像不上雲。

操作（視窗有焦點時按鍵）：
   空白鍵 → 15 秒倒數自拍（單人/一般用）→ 穿搭建議
   m      → 30 秒倒數自拍（多人用，給大家喬位置的時間）→ 穿搭建議
   c      → 立刻拍照（不倒數），給湊近鏡頭用 → 穿搭建議
   q      → 離開

穿搭建議固定分三段、逐字串流出來（畫面會動）：
   【穿搭現狀】目前穿著描述
   【建議改善】2~3 點可行建議
   【總結】一句話給信心

繁體中文：小模型常漏出簡體字。程式收到後一律先轉繁體再顯示
   （優先用 OpenCC，沒裝就用內建常用字對照表頂著）。

視窗用「加大的固定視窗」（像瑜珈大師那樣），不是全螢幕：
  - WINDOW_AUTOSIZE，拖不動 → 不會「一拉變小」；不全螢幕 → 不蓋住終端機。

跑法：python outfit_advisor.py --source 0   （筆電內建 0，外接 webcam 通常是 1）
（可選）最完整的簡轉繁：pip install opencc-python-reimplemented
"""

import argparse

import cv2

from vlm_client import ask_about_image_stream
from selfie_timer import countdown_selfie

WINDOW_TITLE = "V2 穿搭大師（按 q 離開）"


# ── 簡體 → 繁體 轉換（優先 OpenCC，沒裝就用內建常用字對照表） ────────────
def _build_fallback_s2t():
    """內建常用簡→繁單字對照（涵蓋日常 + 穿搭描述常見字；不含易混淆的多對一）。"""
    pairs = {
        "这": "這", "来": "來", "们": "們", "个": "個", "东": "東", "车": "車",
        "长": "長", "时": "時", "现": "現", "场": "場", "风": "風", "协": "協",
        "调": "調", "体": "體", "质": "質", "审": "審", "简": "簡", "约": "約",
        "单": "單", "为": "為", "较": "較", "会": "會", "宽": "寬", "松": "鬆",
        "适": "適", "应": "應", "让": "讓", "过": "過", "还": "還", "没": "沒",
        "说": "說", "话": "話", "问": "問", "题": "題", "张": "張", "开": "開",
        "关": "關", "门": "門", "间": "間", "实": "實", "际": "際", "业": "業",
        "产": "產", "务": "務", "动": "動", "团": "團", "国": "國", "园": "園",
        "图": "圖", "圆": "圓", "学": "學", "觉": "覺", "网": "網", "络": "絡",
        "线": "線", "红": "紅", "绿": "綠", "蓝": "藍", "黄": "黃", "颜": "顏",
        "设": "設", "计": "計", "师": "師", "样": "樣", "总": "總", "结": "結",
        "议": "議", "评": "評", "价": "價", "优": "優", "点": "點", "选": "選",
        "择": "擇", "标": "標", "类": "類", "别": "別", "显": "顯", "当": "當",
        "装": "裝", "饰": "飾", "数": "數", "据": "據", "处": "處", "须": "須",
        "确": "確", "认": "認", "与": "與", "给": "給", "职": "職", "气": "氣",
        "经": "經", "济": "濟", "继": "繼", "续": "續", "传": "傳", "统": "統",
        "细": "細", "节": "節", "构": "構", "层": "層", "纹": "紋", "织": "織",
        "纯": "純", "丝": "絲", "纤": "纖", "维": "維", "尽": "盡", "丰": "豐",
        "满": "滿", "种": "種", "灵": "靈", "韵": "韻", "档": "檔",
        "脱": "脫", "镜": "鏡", "头": "頭", "脸": "臉", "颊": "頰",
        "营": "營", "养": "養", "卫": "衛", "态": "態", "势": "勢", "众": "眾",
        "庄": "莊", "严": "嚴", "齐": "齊",
        # 衣著 / 形容詞 / 常見補強
        "贴": "貼", "紧": "緊", "软": "軟", "谐": "諧", "衬": "襯", "裤": "褲",
        "袜": "襪", "领": "領", "阔": "闊", "飘": "飄", "摆": "擺", "边": "邊",
        "卷": "捲", "绑": "綁", "围": "圍", "浅": "淺", "纪": "紀", "录": "錄",
        "买": "買", "卖": "賣", "贵": "貴", "钱": "錢", "靓": "靚", "亲": "親",
        "热": "熱", "爱": "愛", "欢": "歡", "丑": "醜", "够": "夠", "随": "隨",
        "带": "帶", "习": "習", "惯": "慣", "尝": "嘗", "试": "試", "进": "進",
        "运": "運", "丽": "麗",
    }
    return str.maketrans(pairs)


try:
    from opencc import OpenCC
    _CC = OpenCC("s2t")

    def _to_trad(s):
        try:
            return _CC.convert(s)
        except Exception:
            return s
    _CONV_MODE = "OpenCC"
except Exception:
    _FALLBACK_TABLE = _build_fallback_s2t()

    def _to_trad(s):
        return s.translate(_FALLBACK_TABLE)
    _CONV_MODE = "內建對照表（建議 pip install opencc-python-reimplemented 以求完整）"


# ── 文字建議（散文，三段、逐字串流） ────────────────────────────────────
ADVICE_SYSTEM = (
    "你是專業的服裝穿搭顧問。請用繁體中文（台灣用語），依下面三段格式評論畫面中這個人的"
    "整體穿著，每段標題用【】標出並各自換行，語氣親切、給人信心：\n"
    "【穿搭現狀】用 1~2 句描述目前穿著（單品、顏色、整體風格）。\n"
    "【建議改善】給 2~3 點具體、可行的改善建議。\n"
    "【總結】用一句話總結，給對方信心。\n"
    "規則：只用繁體中文，禁止簡體字，不要英文（專有名詞除外）、不要客套話、不要亂碼。"
)
ADVICE_QUESTION = "請依【穿搭現狀】【建議改善】【總結】三段格式，評論畫面中這個人的整體穿搭。"
ADVICE_MAX_TOKENS = 256

CAP_W, CAP_H = 1280, 720
FOOTER = "\n（空白=15秒  m=多人30秒  c=立刻拍  q=離開）"


# ── 逐字串流印出（轉繁體後再印，這就是「會動」的來源） ──────────────────
def _stream_print(piece):
    print(_to_trad(piece), end="", flush=True)


def advise(frame):
    print("\n" + "=" * 56)
    print("🤔 UGen300 看圖中…（本機運算 · 不連網，答案會逐字出現）\n", flush=True)
    print("👗 穿搭大師：\n", flush=True)
    ask_about_image_stream(frame, ADVICE_QUESTION,
                           on_token=_stream_print,
                           system_prompt=ADVICE_SYSTEM,
                           max_tokens=ADVICE_MAX_TOKENS)
    print("\n" + "=" * 56 + FOOTER)


def _selfie_then_advise(cap, seconds):
    shot, cancelled = countdown_selfie(cap, WINDOW_TITLE,
                                       seconds=seconds, flip_horizontal=True)
    if cancelled or shot is None:
        print("[outfit] 倒數取消")
        return
    advise(shot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam 編號（筆電內建 0，外接通常 1）")
    ap.add_argument("--countdown", type=int, default=15,
                    help="空白鍵的倒數秒數（單人，預設 15）")
    ap.add_argument("--multi-countdown", type=int, default=30,
                    help="m 鍵的倒數秒數（多人，預設 30）")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[outfit] 無法開啟影像來源：{args.source}（外接 webcam 試 --source 1）")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW_TITLE, 0, 0)

    print(f"[outfit] 穿搭大師啟動。站遠一點讓鏡頭照到全身。")
    print(f"         視窗大小 = {real_w}x{real_h}（加大固定視窗，非全螢幕，不會被拖小）")
    print(f"         簡轉繁：{_CONV_MODE}")
    print(f"         空白 = {args.countdown} 秒倒數 → 穿搭建議")
    print(f"         m    = {args.multi_countdown} 秒倒數 → 穿搭建議（多人）")
    print(f"         c    = 立刻拍照 → 穿搭建議（湊近鏡頭用）")
    print(f"         q    = 離開")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 36), (26, 26, 46), -1)
            cv2.putText(frame,
                        f"SPACE={args.countdown}s  m={args.multi_countdown}s multi  "
                        f"c=snap  q=quit",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 216), 2)
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                print("[outfit] c=立刻拍照（不倒數）")
                advise(frame.copy())
                continue

            if key == ord(" "):
                print(f"[outfit] 空白：開始 {args.countdown} 秒倒數自拍…")
                _selfie_then_advise(cap, args.countdown)
                continue

            if key == ord("m"):
                print(f"[outfit] m=多人 {args.multi_countdown} 秒倒數自拍…")
                _selfie_then_advise(cap, args.multi_countdown)
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[outfit] 已結束。")


if __name__ == "__main__":
    import hailo_vdevice
    main()
    hailo_vdevice.exit_now()
