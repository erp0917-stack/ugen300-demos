"""
nutrition.py —— 熱量估算的純邏輯:VLM 提示詞、JSON 容錯解析、今日累計(存本機檔案)。
不碰裝置,可離線測試。

解析策略(小模型常不守格式):
  1. 去掉 ``` 圍欄;把沒加引號的 key 補上引號;單引號→雙引號。
  2. 不嘗試整包 json.loads,而是逐個 {...} 物件解析,壞的跳過 → 陣列被截斷也能救回前面的項目。
  3. key 別名對照(品名/name → item、熱量/calories → kcal …)。
"""
import json
import os
import re
from datetime import date

PROMPT = (
    "你是營養師。辨識照片中的每一種食物或飲料,估算份量、熱量(大卡)與蛋白質(公克)。"
    "只輸出 JSON 陣列,key 固定用 item、portion、kcal、protein_g,全部用雙引號,不要其他文字。"
    '範例:[{"item":"滷肉飯","portion":"1 碗","kcal":550,"protein_g":18},{"item":"珍珠奶茶","portion":"700ml","kcal":480,"protein_g":6}]'
    "。若照片裡沒有食物,輸出 []。"
)

_ALIASES = {
    "item": ("item", "name", "food", "品名", "名稱", "食物", "品項", "菜名"),
    "portion": ("portion", "serving", "amount", "份量", "分量", "量", "數量"),
    "kcal": ("kcal", "calories", "calorie", "energy", "熱量", "卡路里", "大卡"),
    "protein_g": ("protein_g", "protein", "蛋白質", "蛋白质"),
}
_ALIAS_LOOKUP = {a: k for k, al in _ALIASES.items() for a in al}


def _to_int(v, default=0):
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(round(v))
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return int(round(float(m.group()))) if m else default


def _normalize_json_text(s):
    s = re.sub(r"```(?:json)?", "", s)
    s = s.replace("'", '"')
    # 沒加引號的 key:  {portion:"x"  或  , kcal:12
    s = re.sub(r'([{,]\s*)([A-Za-z_][\w]*|[一-鿿]+)\s*:', r'\1"\2":', s)
    # 多餘的 + 號:  "kcal": +14
    s = re.sub(r':\s*\+(\d)', r': \1', s)
    return s


def _normalize_item(d):
    out = {"item": "", "portion": "", "kcal": 0, "protein_g": 0}
    for k, v in d.items():
        key = _ALIAS_LOOKUP.get(str(k).strip().lower()) or _ALIAS_LOOKUP.get(str(k).strip())
        if key == "item":
            out["item"] = str(v).strip()
        elif key == "portion":
            out["portion"] = str(v).strip()
        elif key == "kcal":
            out["kcal"] = max(0, _to_int(v))
        elif key == "protein_g":
            out["protein_g"] = max(0, _to_int(v))
    return out if out["item"] else None


def parse_items(text):
    """從 VLM 回覆中盡量抓出食物項目。失敗回傳 []。"""
    if not text:
        return []
    s = _normalize_json_text(text)
    items = []
    for m in re.finditer(r"\{[^{}]*\}", s):
        chunk = re.sub(r",\s*}", "}", m.group())          # 尾逗號
        try:
            d = json.loads(chunk)
        except json.JSONDecodeError:
            # 最後一招:把 key:value 用正則抓出來
            d = {k: v for k, v in re.findall(r'"([^"]+)"\s*:\s*"?([^",}]*)"?', chunk)}
        if isinstance(d, dict):
            it = _normalize_item(d)
            if it:
                items.append(it)
    return items


def totals(items):
    return dict(kcal=sum(i["kcal"] for i in items), protein_g=sum(i["protein_g"] for i in items))


class DailyLog:
    """今日累計。存在本機 json,跨日自動歸零。"""
    def __init__(self, path):
        self.path = path
        self.data = {"date": date.today().isoformat(), "meals": []}
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("date") == self.data["date"]:
                self.data = d
        except (OSError, ValueError):
            pass

    def add(self, items):
        self.data["meals"].append({"items": items, **totals(items)})
        self._save()

    def clear(self):
        self.data["meals"] = []; self._save()

    @property
    def kcal(self): return sum(m["kcal"] for m in self.data["meals"])
    @property
    def meals(self): return len(self.data["meals"])

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
