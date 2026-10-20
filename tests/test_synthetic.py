"""Research mode (cameras without a plate): transect detector and calibration on synthetic frames
with a known water level. Slow (~1 min); run with: pytest -m research"""
import os
import sys

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import calibrate, vision  # noqa: E402

rng = np.random.default_rng(0)
H, W = 720, 1280
BANK = np.random.default_rng(1).integers(0, 60, (H, W, 3), dtype=np.uint8)  # fixed scene texture
BANK = cv.GaussianBlur(BANK, (3, 3), 0)


def frame(water_row: int, shift: int = 0) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (90, 140, 110)                                            # vegetated bank
    img += BANK                                                        # bank texture
    for k in range(40):                                                # fixed landmarks for alignment
        x, y = (k * 97) % W, (k * 53) % 250 + 20
        cv.rectangle(img, (x, y), (x + 25, y + 18), (30 + k * 4, 60, 200 - k * 3), -1)
    water = np.full((H - water_row, W, 3), (120, 95, 60), np.uint8)    # muddy water
    water = cv.add(water, rng.integers(0, 8, water.shape, dtype=np.uint8))
    img[water_row:] = water
    img = cv.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))  # sensor noise
    if shift:
        img = cv.warpAffine(img, np.float32([[1, 0, shift], [0, 1, shift // 2]]), (W, H),
                            borderMode=cv.BORDER_REFLECT)
    return img


@pytest.mark.research
def test_transects_on_synthetic_frames():
    transects = [[[600, 300], [600, 700]], [[700, 300], [700, 700]], [[800, 300], [800, 700]]]
    ref = frame(500)
    aligner = vision.Aligner(ref)
    rows = rng.integers(340, 680, 60)
    stage = 20 - (rows - 340) * 0.03            # higher water = smaller row = larger stage (ft)
    fr, ok = [], 0
    for i, r in enumerate(rows):
        m = vision.measure(frame(int(r), shift=6 if i % 5 == 0 else 0), transects, aligner)
        ok += m["status"] == "ok"
        fr.append(m["fraction"] if m["status"] == "ok" else np.nan)
    fr = np.array(fr)
    good = ~np.isnan(fr)
    model = calibrate.fit(fr[good][:40], stage[good][:40])
    assert ok >= 55
    assert calibrate.metrics(stage[good][40:], calibrate.predict(model, fr[good][40:]))["rmse"] < 0.1
    dark = (frame(500) * 0.1).astype(np.uint8)
    assert vision.measure(dark, transects, aligner)["status"] == "rejected"
