"""Plate reader on real Hancock frames, the agent's fault handling, and the API (run: pytest).

Gauge values are the USGS record at the frame times; the plate reading uses no gauge data.
"""
import base64
import json
import os
import sys

import cv2 as cv
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "aws"))
from stagesight import service, storage  # noqa: E402
from stagesight.agent import Monitor, RulePlanner  # noqa: E402
from stagesight.reader import SiteReader  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
reader = SiteReader(os.path.join(ROOT, "sites", "hancock"))
flood = cv.imread(os.path.join(FIX, "hancock_flood_17.61ft.jpg"))


@pytest.mark.parametrize("name,gauge", [("hancock_flood_17.61ft.jpg", 17.61), ("hancock_recession_12.47ft.jpg", 12.47)])
def test_plate_reads_real_flood_frames(name, gauge):
    r = reader.read(cv.imread(os.path.join(FIX, name)))
    assert r["status"] == "ok" and abs(r["stage_ft"] - gauge) <= 0.5


def test_no_number_without_a_plate():
    assert reader.read(np.full((1920, 1080, 3), 128, np.uint8))["stage_ft"] is None   # blank frame
    assert reader.read(cv.resize(flood, (540, 960)))["status"] == "wrong_size"          # other camera


class FakeWorld:
    """The flood frame every hour and a constant gauge value."""

    def __init__(self, img, gauge_ft):
        self.img, self.g, self.now = img, gauge_ft, np.datetime64("2026-07-29T16:00:00")

    def frame(self, back=0):
        return (self.now - np.timedelta64(back, "h"), self.img)

    def gauge(self, hours=3):
        return [(str(self.now - np.timedelta64(15 * k, "m")), self.g) for k in range(int(hours * 4))][::-1]

    def gauge_now(self, max_age_min=30):
        return self.g

    def upstream(self, hours=12):
        return []


def _run(gauge_ft, ticks=3):
    world, actions = FakeWorld(flood, gauge_ft), []
    mon = Monitor(reader, world, RulePlanner())
    for _ in range(ticks):
        world.now += np.timedelta64(1, "h")
        actions += mon.tick()["actions"]
    return actions


def test_gauge_stuck_high_is_flagged_and_queued():
    actions = _run(17.61 + 1.5)
    kinds = [a["type"] for a in actions]
    assert "flag_sensor" in kinds and "publish_provisional_stage" in kinds
    assert all(a.get("status") == "pending_approval" for a in actions if a["type"] in ("flag_sensor", "draft_alert"))


def test_healthy_gauge_is_not_flagged():
    assert "flag_sensor" not in [a["type"] for a in _run(17.61)]


def test_each_frame_is_processed_once(tmp_path):
    store, world = storage.LocalStore(str(tmp_path)), FakeWorld(flood, 17.61)
    first = service.tick(store, reader, world, RulePlanner())
    again = service.tick(store, reader, world, RulePlanner())           # next 15-min tick, same hourly frame
    assert "skipped" not in first and again.get("skipped")
    assert len(storage.get_json(store, "timeseries.json")) == 1


def test_only_approved_items_are_sent():
    sent = []
    item = {"type": "draft_alert", "level": "flood", "message": "Stage 18.4 ft", "decided_by": "op", "decided_utc": "t"}
    assert "sent" not in service.notify({**item, "status": "rejected"}, "arn:x", lambda **k: sent.append(k))
    assert service.notify({**item, "status": "approved"}, "arn:x", lambda **k: sent.append(k))["sent"] == "sns"
    assert len(sent) == 1 and "flood alert" in sent[0]["Subject"]


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", f"local:{tmp_path}")
    monkeypatch.setenv("APPROVE_TOKEN", "secret")
    import handler
    handler._reader = None

    def call(method, path, body=b"", headers=None, b64=True):
        ev = {"requestContext": {"http": {"method": method}}, "rawPath": path, "headers": headers or {},
              "body": base64.b64encode(body).decode() if b64 else body.decode("utf-8", "replace"), "isBase64Encoded": b64}
        r = handler.api(ev, None)
        return r["statusCode"], r["body"]
    return call


def test_api_rejects_bad_requests(api):
    assert api("POST", "/approve", b'{"id":"x","decision":"approve"}')[0] == 401              # no token
    auth = {"Authorization": "Bearer secret"}
    assert api("POST", "/approve", b"\xff\xfe", auth)[0] == 400                                # not UTF-8
    assert api("POST", "/approve", b"[1]", auth)[0] == 400                                     # not an object
    assert api("POST", "/approve", '{"id":"x","decision":"approve","who":"操作"}'.encode(), auth, b64=False)[0] == 404
    assert api("POST", "/approve", b'{"id":"x"}', auth)[0] == 400                              # no silent reject
    assert api("POST", "/approve", b'{"id":"x","decision":"approve"}', {"Authorization": "Bearer \xe9"})[0] == 401
    assert api("POST", "/measure", b"")[0] == 400
    assert api("GET", "/data/../state.json")[0] == 404
    assert api("GET", "/replay/../../handler.py")[0] == 404
    assert json.loads(api("GET", "/data/events.json")[1]) == []                                 # empty, not 404
    assert api("POST", "/measure", cv.imencode(".jpg", cv.resize(flood, (540, 960)))[1].tobytes())[0] == 200
