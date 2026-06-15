# -*- coding: utf-8 -*-
"""
hand_raise_logic.py  ——  舉手統計的「大腦」（純邏輯，不依賴硬體）
================================================================
拿到「多個人」的關鍵點後，數出有幾個人舉手（手腕高過頭/肩）。
可獨立測試。

輸入是一個 list，每個元素是「一個人的 17 個關鍵點」。
（單人版的 hailo_pose 回傳一個人；多人版回傳多個人的清單——
  週一對接時請 Claude Code 把 hailo_pose 改成回傳多人，見 README。）
"""

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10


def is_hand_raised(person_kps, conf_threshold=0.5, margin=20):
    """
    判斷單一個人有沒有舉手：任一手腕高過『鼻子或肩膀』。
    回傳 True / False。
    """
    def get(idx):
        x, y, c = person_kps[idx]
        return (x, y) if c >= conf_threshold else None

    nose = get(NOSE)
    ls = get(LEFT_SHOULDER)
    rs = get(RIGHT_SHOULDER)
    lw = get(LEFT_WRIST)
    rw = get(RIGHT_WRIST)

    # 用鼻子；沒有就用肩膀平均高度當基準
    if nose is not None:
        base_y = nose[1]
    elif ls and rs:
        base_y = (ls[1] + rs[1]) / 2
    else:
        return False  # 連基準都沒有，無法判斷

    # 影像座標 y 往下變大，「舉高」= 手腕 y 比基準小
    for w in (lw, rw):
        if w is not None and w[1] < base_y - margin:
            return True
    return False


def count_raised_hands(people_kps, conf_threshold=0.5, margin=20):
    """
    輸入：people_kps = list，每個元素是一個人的 17 點
    回傳：(舉手人數, 總偵測人數)
    """
    if not people_kps:
        return 0, 0
    raised = sum(1 for p in people_kps if is_hand_raised(p, conf_threshold, margin))
    return raised, len(people_kps)
