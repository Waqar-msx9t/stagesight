"""Checks on real Hancock frames and on the agent's fault handling (run: python tests/test_plate.py).

Gauge values are the USGS record at the frame times; the plate reading uses no gauge data.
"""
import os
import sys

import cv2 as cv
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from stagesight.agent import Monitor, RulePlanner  # noqa: E402
from stagesight.reader import SiteReader  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
reader = SiteReader(os.path.join(ROOT, "sites", "hancock"))

# 1. plate reading on two real frames, within 0.5 ft of the gauge
for name, gauge in (("hancock_flood_17.61ft.jpg", 17.61), ("hancock_recession_12.47ft.jpg", 12.47)):
    r = reader.read(cv.imread(os.path.join(FIX, name)))
    print(f"{name}: camera {r['stage_ft']} ft, gauge {gauge} ft, status {r['status']}")
    assert r["status"] == "ok" and abs(r["stage_ft"] - gauge) <= 0.5

# 2. a blank frame is not read as a stage
blank = np.full((1920, 1080, 3), 128, np.uint8)
assert reader.read(blank)["stage_ft"] is None


# 3. agent: a gauge stuck 1.5 ft high while the camera reads the flood -> flagged, camera published
class FakeWorld:
    def __init__(self, img, gauge_ft):
        self.img, self.g, self.now = img, gauge_ft, np.datetime64("2026-07-29T16:00:00")

    def frame(self, back=0):
        return (self.now - np.timedelta64(back, "h"), self.img)

    def gauge(self, hours=3):
        n = int(hours * 4)
        return [(str(self.now - np.timedelta64(15 * k, "m")), self.g) for k in range(n)][::-1]

    def gauge_now(self, max_age_min=30):
        return self.g

    def upstream(self, hours=12):
        return []


flood = cv.imread(os.path.join(FIX, "hancock_flood_17.61ft.jpg"))
world = FakeWorld(flood, 17.61 + 1.5)
mon = Monitor(reader, world, RulePlanner())
actions = []
for k in range(3):
    world.now += np.timedelta64(1, "h")
    actions += mon.tick()["actions"]
kinds = [a["type"] for a in actions]
print("agent actions:", kinds)
assert "flag_sensor" in kinds and "publish_provisional_stage" in kinds
assert all(a.get("status") == "pending_approval" for a in actions if a["type"] in ("flag_sensor", "draft_alert"))

# 4. same frames, healthy gauge -> no sensor flag
world = FakeWorld(flood, 17.61)
mon = Monitor(reader, world, RulePlanner())
kinds = [a["type"] for k in range(3) for a in mon.tick()["actions"]]
assert "flag_sensor" not in kinds
print("all checks passed")
