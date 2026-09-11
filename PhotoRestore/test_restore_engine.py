"""離線單元測試:python test_restore_engine.py(不需 UGen300;用假的推論函式)"""
import numpy as np
from restore_engine import plan_tiles, TileRunner, blend


def test_plan_tiles_covers_and_aligns_edge():
    t = plan_tiles(1000, 1300, 400, 600, overlap=16)
    ys = sorted({y for y, _ in t}); xs = sorted({x for _, x in t})
    assert ys[0] == 0 and ys[-1] == 600 and xs[0] == 0 and xs[-1] == 700
    assert plan_tiles(300, 500, 400, 600) == [(0, 0)]


def test_identity_infer_reconstructs_image_exactly():
    img = np.random.randint(0, 256, (700, 900, 3), np.uint8)
    out = TileRunner(lambda t: t, (400, 600), 1, 16).run(img)
    assert out.shape == img.shape and np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_scale2_infer_doubles_size():
    img = np.random.randint(0, 256, (600, 700, 3), np.uint8)
    up = lambda t: np.repeat(np.repeat(t, 2, 0), 2, 1)
    out = TileRunner(up, (512, 512), 2, 16).run(img)
    assert out.shape == (1200, 1400, 3)
    ref = np.repeat(np.repeat(img, 2, 0), 2, 1)
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1


def test_small_image_gets_padded_then_cropped():
    img = np.random.randint(0, 256, (100, 120, 3), np.uint8)
    out = TileRunner(lambda t: t, (400, 600), 1).run(img)
    assert out.shape == img.shape and np.array_equal(out, img)


def test_progress_callback_counts_tiles():
    calls = []
    TileRunner(lambda t: t, (400, 600), 1).run(np.zeros((1000, 1300, 3), np.uint8), lambda i, n: calls.append((i, n)))
    assert calls[-1][0] == calls[-1][1] == len(calls)


def test_blend_strength():
    a = np.zeros((10, 10, 3), np.uint8); b = np.full((10, 10, 3), 200, np.uint8)
    assert int(blend(a, b, 0.5)[0, 0, 0]) == 100
    assert blend(a, np.full((20, 20, 3), 200, np.uint8), 1.0).shape == (20, 20, 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("✅", name)
    print("全部通過")
