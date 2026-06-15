# -*- coding: utf-8 -*-
"""
test_gesture_logic.py
=====================
用「假的關鍵點資料」測試 gesture_control 的判斷與防呆邏輯，
完全不需要 UGen300、不需要 webcam。

目的：證明「手勢判斷 + 防呆狀態機」是對的，讓 Matt 安心。
跑法：python test_gesture_logic.py
"""

from gesture_control import (
    detect_raw_gesture, GestureController, trigger_action,
    GESTURE_NONE, GESTURE_RIGHT, GESTURE_LEFT, GESTURE_BOTH,
    KP,
)


def make_keypoints(right_up=False, left_up=False):
    """
    做一組假的 17 點資料。
    預設大家都在「肚子高度」(y=300)，鼻子在 y=100。
    要舉手就把那隻手腕拉到 y=30（比鼻子 100 還高）。
    每個點格式 (x, y, confidence)，信心度都給 0.9。
    """
    kps = [(100, 300, 0.9) for _ in range(17)]
    kps[KP["nose"]] = (100, 100, 0.9)      # 頭在 y=100
    kps[KP["left_wrist"]] = (60, 30, 0.9) if left_up else (60, 320, 0.9)
    kps[KP["right_wrist"]] = (140, 30, 0.9) if right_up else (140, 320, 0.9)
    return kps


def feed(ctrl, keypoints, frames, label):
    """模擬連續 frames 格都是同一個畫面，印出每格結果"""
    print(f"\n--- 模擬：{label}（持續 {frames} 格）---")
    fired = []
    for i in range(frames):
        raw = detect_raw_gesture(keypoints)
        action = ctrl.update(raw)
        tag = f"  第{i+1}格 raw={raw:11s}"
        if action:
            tag += f"  → 觸發動作：{action}"
            fired.append(action)
            trigger_action(action)
        print(tag)
    return fired


def main():
    print("=" * 55)
    print("  D4 手勢邏輯測試（不需硬體）")
    print("=" * 55)

    # 先測「純判斷」對不對
    print("\n【測試一】純手勢判斷 detect_raw_gesture")
    assert detect_raw_gesture(make_keypoints(right_up=True)) == GESTURE_RIGHT
    assert detect_raw_gesture(make_keypoints(left_up=True)) == GESTURE_LEFT
    assert detect_raw_gesture(make_keypoints(right_up=True, left_up=True)) == GESTURE_BOTH
    assert detect_raw_gesture(make_keypoints()) == GESTURE_NONE
    print("  ✓ 舉右手→right_hand、舉左手→left_hand、雙手→both_hands、放下→none　全部正確")

    # 再測「防呆狀態機」
    print("\n【測試二】防呆狀態機 GestureController")
    ctrl = GestureController(confirm_frames=4, cooldown_sec=1.2)

    # 情境 1：舉右手 6 格 → 應該只翻「1 次」下一頁（連續確認 4 格後觸發，之後鎖住）
    fired = feed(ctrl, make_keypoints(right_up=True), 6, "舉右手不放（6 格）")
    assert fired.count("next") == 1, f"期望只翻 1 次，實際 {fired.count('next')} 次"
    print("  ✓ 一直舉著不會連翻，只翻 1 次（防『連續翻頁』成功）")

    # 情境 2：手放下 3 格（重新上膛）
    feed(ctrl, make_keypoints(), 3, "手放下（3 格）")

    # 情境 3：又舉右手 5 格 → 因為冷卻時間還沒過，這次可能不觸發或剛好觸發
    #         為了乾淨測試，先睡過冷卻時間
    import time
    time.sleep(1.3)
    fired2 = feed(ctrl, make_keypoints(right_up=True), 5, "放下後再舉右手（5 格）")
    assert fired2.count("next") == 1, "手放下再舉，應該能再翻 1 次"
    print("  ✓ 手放下再舉，可以再翻一次（防呆不會卡死）")

    # 情境 4：閃動 2 格右手就放下 → 不該觸發（連續格數不足）
    ctrl2 = GestureController(confirm_frames=4, cooldown_sec=1.2)
    f = []
    f += feed(ctrl2, make_keypoints(right_up=True), 2, "右手只閃 2 格")
    f += feed(ctrl2, make_keypoints(), 2, "馬上放下")
    assert f.count("next") == 0, "只閃 2 格不該觸發"
    print("  ✓ 雜訊閃動（只出現 2 格）不會誤觸發（防『誤判』成功）")

    print("\n" + "=" * 55)
    print("  全部測試通過 ✅　手勢邏輯與防呆機制正確")
    print("=" * 55)


if __name__ == "__main__":
    main()
