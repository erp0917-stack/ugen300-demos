# 開發用:驗證 SCRFD 解碼與 ArcFace 向量(不進 launcher)
import sys, time, cv2, numpy as np
from face_engine import FaceDetector, FaceEmbedder
img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else r"..\FACE1\face.jpg")
det = FaceDetector("scrfd_10g.hef", conf=0.5); emb = FaceEmbedder()
t = time.time(); faces = det.detect(img); dt = time.time() - t
print("faces", len(faces), f"{dt*1000:.0f} ms")
for f in faces[:3]: print(" box", f["box"], "score", round(f["score"], 3), "kps", f["kps"].astype(int).tolist())
if faces:
    f = faces[0]
    v1 = emb.embed(img, f["kps"])
    flipped = cv2.flip(img, 1); ff = det.detect(flipped)[0]; v2 = emb.embed(flipped, ff["kps"])
    v3 = emb.embed(img, f["kps"] + np.array([[6, 4]] * 5))  # 對齊點稍微偏移
    noise = np.random.randint(0, 255, img.shape, np.uint8); v4 = emb.embed(noise, f["kps"])
    print("同臉鏡像 sim", round(float(v1 @ v2), 3), " 對齊偏移 sim", round(float(v1 @ v3), 3), " 雜訊 sim", round(float(v1 @ v4), 3))
    x1, y1, x2, y2 = f["box"]; cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y in f["kps"].astype(int): cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.imwrite("probe_out.jpg", img); cv2.imwrite("probe_align.jpg", emb.align(cv2.imread(sys.argv[1] if len(sys.argv) > 1 else r"..\FACE1\face.jpg"), f["kps"]))
    print("寫出 probe_out.jpg / probe_align.jpg")

import hailo_vdevice; hailo_vdevice.exit_now()
