"""
hand_tracker.py —— 共用手部追蹤:姿態找手腕 → 裁手部方框 → hand_landmark_lite 21 點 → 映回原圖座標。

hand_landmark_lite.hef(HAILO10H)輸出:
  fc1 (63) = 21×(x,y,z),x/y 為 224×224 輸入的像素座標
  fc2 (1)  = 左右手機率
  fc3 (63) = 世界座標(公尺)
  fc4 (1)  = 有手的信心(實測沒手也常 >0.9,不可靠 → 用姿態手腕信心當主要判斷)
"""
import numpy as np
import cv2

from hailo_pose import PoseEstimator

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
HAND_INPUT = 224
# MediaPipe 21 點骨架連線
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Hand:
    __slots__ = ("side", "pts", "score", "box", "wrist")

    def __init__(self, side, pts, score, box, wrist):
        self.side, self.pts, self.score, self.box, self.wrist = side, pts, score, box, wrist  # pts: (21,2) 原圖座標


def hand_box_from_pose(kps, wrist_idx, elbow_idx, frame_h, conf=0.35):
    """由手腕與手肘推估手掌方框(手在前臂延長方向)。回傳 (x0,y0,size) 或 None。"""
    wx, wy, wc = kps[wrist_idx]
    if wc < conf:
        return None
    ex, ey, ec = kps[elbow_idx]
    if ec >= conf:
        fl = float(np.hypot(wx - ex, wy - ey))
        size = max(60.0, fl * 1.1)
        dx, dy = (wx - ex) / max(fl, 1e-6), (wy - ey) / max(fl, 1e-6)
        cx, cy = wx + dx * size * 0.35, wy + dy * size * 0.35   # 中心往手指方向推
    else:
        size = frame_h * 0.28; cx, cy = wx, wy - size * 0.2
    return cx - size / 2, cy - size / 2, size


class HandTracker:
    def __init__(self, pose_hef="yolov8s_pose.hef", hand_hef="hand_landmark_lite.hef", conf=0.3):
        import hailo_vdevice
        from hailo_platform import FormatType
        self.pose = PoseEstimator(pose_hef, conf_threshold=conf)      # 內部已用 hailo_vdevice 單例
        vd = hailo_vdevice.get()
        self.m = vd.create_infer_model(hand_hef); self.m.set_batch_size(1)
        for n in self.m.output_names:
            self.m.output(n).set_format_type(FormatType.FLOAT32)
        self.c = self.m.configure()
        names = self.m.output_names
        self.n_lm = next(n for n in names if n.endswith("fc1")); self.n_score = next(n for n in names if n.endswith("fc4"))
        self.n_hand = next(n for n in names if n.endswith("fc2"))
        self.last_people = []

    def _landmarks(self, crop_bgr):
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (HAND_INPUT, HAND_INPUT)), cv2.COLOR_BGR2RGB)
        b = self.c.create_bindings(); b.input().set_buffer(np.ascontiguousarray(rgb))
        outs = {n: np.zeros(tuple(self.m.output(n).shape), np.float32) for n in self.m.output_names}
        for n, a in outs.items(): b.output(n).set_buffer(a)
        self.c.run([b], 10000)
        lm = outs[self.n_lm].reshape(21, 3)
        return lm, float(outs[self.n_score][0]), float(outs[self.n_hand][0])

    def update(self, frame, max_hands=2):
        """回傳 list[Hand](最多 max_hands),依姿態偵測到的人依序取左右手。"""
        h, w = frame.shape[:2]
        people = self.pose.infer_multi(frame) or []
        self.last_people = people
        hands = []
        for kps in people:
            for side, wi, ei in (("R", R_WRIST, R_ELBOW), ("L", L_WRIST, L_ELBOW)):
                box = hand_box_from_pose(kps, wi, ei, h)
                if box is None:
                    continue
                x0, y0, size = box
                x0i, y0i = int(round(x0)), int(round(y0)); si = int(round(size))
                # 超出邊界就補黑邊,保持正方形
                pad = max(0, -x0i, -y0i, x0i + si - w, y0i + si - h)
                src = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else frame
                crop = src[y0i + pad:y0i + pad + si, x0i + pad:x0i + pad + si]
                if crop.size == 0:
                    continue
                lm, score, handed = self._landmarks(crop)
                pts = lm[:, :2] / HAND_INPUT * si + np.array([x0i, y0i], np.float32)
                hands.append(Hand(side, pts.astype(np.float32), score, (x0i, y0i, si), (kps[wi][0], kps[wi][1])))
                if len(hands) >= max_hands:
                    return hands
        return hands

    def close(self):
        self.pose.close()


def draw_hand(frame, hand, color=(80, 220, 120)):
    x0, y0, s = hand.box
    cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (90, 90, 90), 1)
    for a, b in HAND_EDGES:
        pa, pb = hand.pts[a], hand.pts[b]
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
    for p in hand.pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)
    return frame
