"""離線單元測試:python test_nutrition.py(不需 UGen300)"""
import os, tempfile
from nutrition import parse_items, totals, DailyLog


def test_clean_json():
    r = parse_items('[{"item":"炸雞腿便當","portion":"1 份","kcal":850,"protein_g":35}]')
    assert r == [dict(item="炸雞腿便當", portion="1 份", kcal=850, protein_g=35)]


def test_wrapped_in_text_and_fence():
    t = "好的,以下是估算:\n```json\n[{\"item\":\"珍珠奶茶\",\"portion\":\"700ml\",\"kcal\":\"約 500 大卡\",\"protein_g\":7.6}]\n```\n希望有幫助。"
    r = parse_items(t)
    assert r[0]["item"] == "珍珠奶茶" and r[0]["kcal"] == 500 and r[0]["protein_g"] == 8


def test_single_quotes_and_trailing_comma():
    r = parse_items("[{'item': '白飯', 'portion': '一碗', 'kcal': 280, 'protein_g': 5,},]")
    assert r and r[0]["item"] == "白飯" and r[0]["kcal"] == 280


def test_garbage_returns_empty():
    assert parse_items("這張照片沒有食物。") == []
    assert parse_items("") == []
    assert parse_items("[") == []


def test_unquoted_keys_and_truncated_array():
    t = '```json\n[\n  {\n    "item":"茶包",\n    portion:"一包",\n    kcal:0,\n    protein_g:0\n  },\n  {\n    "item":"牛奶",\n    portion:"一杯(250ml)",\n    kcal:367,\n    protein_g:+14.'
    r = parse_items(t)
    assert [i["item"] for i in r] == ["茶包"], r          # 截斷的第二項救不回是正常,第一項要在
    assert r[0]["portion"] == "一包"


def test_chinese_key_aliases_and_space_in_key():
    t = '[\n  {"品名":"牛奶","portion":"一杯","kcal":210," protein_g":3.5},\n  {"品名":"巧克力棒","份量":"一包","熱量":486,"蛋白質":7.9}\n]'
    r = parse_items(t)
    assert [i["item"] for i in r] == ["牛奶", "巧克力棒"]
    assert r[0]["protein_g"] == 4 and r[1]["kcal"] == 486 and r[1]["portion"] == "一包"


def test_totals():
    assert totals([dict(item="a", portion="", kcal=100, protein_g=3), dict(item="b", portion="", kcal=50, protein_g=2)]) == dict(kcal=150, protein_g=5)


def test_daily_log_persists_and_resets_by_date():
    p = os.path.join(tempfile.mkdtemp(), "today.json")
    log = DailyLog(p); log.add([dict(item="x", portion="", kcal=300, protein_g=1)])
    log2 = DailyLog(p); assert log2.kcal == 300 and log2.meals == 1
    log2.data["date"] = "2000-01-01"; log2._save()
    log3 = DailyLog(p); assert log3.kcal == 0, "跨日應歸零"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("✅", name)
    print("全部通過")
