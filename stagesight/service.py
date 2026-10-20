"""The running service: one tick reads the newest frame, runs the agent and publishes results.

Used by aws/handler.py on Lambda (store = S3) and by scripts/serve_local.py on a laptop
(store = a local folder). Keys written to the store:
  latest.json, timeseries.json, events.json, approvals.json, state.json, latest_overlay.jpg
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import cv2 as cv
import numpy as np
import requests

from stagesight import storage, usgs
from stagesight.agent import AgentState, Monitor
from stagesight.reader import SiteReader, public

MAX_SERIES = 3000


class LiveWorld:
    """Same interface as ReplayWorld, backed by the live NIMS camera and USGS gauges."""

    def __init__(self, site: dict, upstream_site: str | None = None):
        self.site, self.upstream_site = site, upstream_site
        self._urls = usgs.list_images(site["camera_id"], limit=6)       # newest first
        # the agent's clock is the newest frame's time, so camera and gauge are compared at
        # the same moment; USGS gauges report about hourly, so allow that delay (gauge_now)
        self.now = (usgs.image_time(self._urls[0]) if self._urls
                    else np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None), "s"))
        self._series: dict[str, tuple] = {}
        self._frames: dict[int, tuple | None] = {}

    def frame(self, back: int = 0):
        if back >= len(self._urls):
            return None
        if back not in self._frames:
            url = self._urls[back]
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            img = cv.imdecode(np.frombuffer(r.content, np.uint8), cv.IMREAD_COLOR)
            self._frames[back] = (usgs.image_time(url), img) if img is not None else None
        return self._frames[back]

    def _get(self, site_id: str):
        if site_id not in self._series:
            day = str(self.now.astype("datetime64[D]"))
            prev = str((self.now - np.timedelta64(2, "D")).astype("datetime64[D]"))
            try:
                self._series[site_id] = usgs.gage_height(site_id, prev, day)
            except (requests.RequestException, KeyError, IndexError, ValueError):
                self._series[site_id] = (np.array([], "datetime64[s]"), np.array([]))
        return self._series[site_id]

    @staticmethod
    def _window(t, v, now, hours):
        m = (t <= now) & (t > now - np.timedelta64(int(hours * 3600), "s"))
        return [(str(a), round(float(b), 2)) for a, b in zip(t[m], v[m])]

    def gauge(self, hours: float = 3):
        return self._window(*self._get(self.site["site_id"]), self.now, hours)

    def gauge_now(self, max_age_min: int = 90):
        """Gauge at the frame time: nearest reading within 20 min, else the latest within the
        normal reporting delay; None means the gauge really is silent."""
        r = self.gauge(max_age_min / 60)
        if not r:
            t, v = self._get(self.site["site_id"])
            near = (t >= self.now - np.timedelta64(20, "m")) & (t <= self.now + np.timedelta64(20, "m"))
            return round(float(v[near][np.argmin(np.abs(t[near] - self.now))]), 2) if near.any() else None
        return r[-1][1]

    def upstream(self, hours: float = 12):
        return self._window(*self._get(self.upstream_site), self.now, hours) if self.upstream_site else []


def load_state(store) -> AgentState:
    s = storage.get_json(store, "state.json")
    if not s:
        return AgentState()
    st = AgentState(**{k: v for k, v in s.items() if k != "open_issues"})
    st.open_issues = set(s.get("open_issues", []))
    return st


def save_state(store, st: AgentState) -> None:
    storage.put_json(store, "state.json", {**st.__dict__, "open_issues": sorted(st.open_issues)})


def tick(store, reader: SiteReader, world, planner) -> dict:
    """One monitoring step. Returns the published record."""
    t0 = time.perf_counter()
    # the schedule runs every 15 min but the camera posts hourly: each frame is processed once,
    # so the agent's frame counts (CONFIRM_FRAMES, camera_blind) count frames, not ticks
    prev = storage.get_json(store, "latest.json") or {}
    if prev.get("frame_time_utc") and str(prev["frame_time_utc"])[:19] == str(world.now)[:19]:
        return {**prev, "skipped": "frame already processed"}
    mon = Monitor(reader, world, planner, load_state(store))
    rec = mon.tick()
    cam = rec["camera"] or {}
    stage = cam.get("stage_ft")
    latest = {"time_utc": rec["time_utc"], "frame_time_utc": cam.get("time_utc"),
              "camera": cam, "gauge_ft": rec["gauge_ft"],
              "alert_level": reader.alert_level(stage if stage is not None else rec["gauge_ft"]),
              "events": [e["event"]["kind"] for e in rec["events"]],
              "action_stage_ft": reader.site.get("action_stage_ft"),
              "flood_stage_ft": reader.site.get("flood_stage_ft"),
              "site": {k: reader.site[k] for k in ("name", "site_id", "camera_id", "datum_ft")},
              "planner": planner.name, "tick_seconds": None}
    # while the agent's gauge flag (or silent-gauge issue) is open, the camera is the provisional stage
    latest["gauge_flagged"] = "disagree" in mon.state.open_issues
    if (latest["gauge_flagged"] or rec["gauge_ft"] is None) and stage is not None:
        latest["provisional_ft"] = stage
    # overlay of the frame just read
    fr = world.frame(0)
    if fr is not None:
        ok, jpg = cv.imencode(".jpg", reader.draw(fr[1], reader.read(fr[1]), rec["gauge_ft"]),
                              [cv.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            store.put("latest_overlay.jpg", jpg.tobytes(), "image/jpeg")
    latest["tick_seconds"] = round(time.perf_counter() - t0, 2)
    storage.put_json(store, "latest.json", latest)
    series = storage.get_json(store, "timeseries.json", [])
    series = (series + [{"t": rec["time_utc"], "camera_ft": stage, "gauge_ft": rec["gauge_ft"],
                         "status": cam.get("status"), "provisional_ft": latest.get("provisional_ft")}])[-MAX_SERIES:]
    storage.put_json(store, "timeseries.json", series)
    if rec["events"]:
        events = storage.get_json(store, "events.json", [])
        events += [{"time_utc": rec["time_utc"], **e} for e in rec["events"]]
        storage.put_json(store, "events.json", events[-300:])
        approvals = storage.get_json(store, "approvals.json", [])
        for a in rec["actions"]:
            if a.get("status") == "pending_approval":
                approvals.append({"id": uuid.uuid4().hex[:8], **a})
        storage.put_json(store, "approvals.json", approvals[-200:])
    save_state(store, mon.state)
    return latest


def decide(store, approval_id: str, decision: str, who: str = "operator") -> dict:
    """Record a human decision on a queued flag or alert. Nothing is sent from here."""
    approvals = storage.get_json(store, "approvals.json", [])
    for a in approvals:
        if a["id"] == approval_id and a["status"] == "pending_approval":
            a["status"] = "approved" if decision == "approve" else "rejected"
            a["decided_by"], a["decided_utc"] = who, datetime.now(timezone.utc).isoformat(timespec="seconds")
            storage.put_json(store, "approvals.json", approvals)
            return a
    return {"error": "not found or already decided"}


def notify(item: dict, topic_arn: str | None, publish=None) -> dict:
    """After an operator approves a flag or alert, send it to the subscribers of an SNS topic.

    This is the only path by which anything leaves the system, and only for approved items.
    `publish` is injectable for tests; by default it is boto3's SNS publish.
    """
    if not topic_arn or item.get("status") != "approved":
        return item
    kind = f"{item.get('level', '')} alert".strip() if item["type"] == "draft_alert" else f"sensor flag: {item.get('suspect')}"
    subject = f"StageSight, Hancock NY (USGS 01421500): {kind}"[:100]
    body = "\n".join([f"Approved by {item.get('decided_by')} at {item.get('decided_utc')} UTC.", "",
                      item.get("message") or item.get("reason") or "", "",
                      f"Evidence: {item['evidence']}" if item.get("evidence") else "",
                      "StageSight is a decision aid, not an official warning. Camera stages are provisional."])
    try:
        if publish is None:
            import boto3
            publish = boto3.client("sns").publish
        publish(TopicArn=topic_arn, Subject=subject, Message=body)
        return {**item, "sent": "sns"}
    except Exception as exc:  # noqa: BLE001  (a failed email must not lose the recorded decision)
        return {**item, "sent": f"failed: {type(exc).__name__}"}


def measure_bytes(reader: SiteReader, data: bytes) -> dict:
    """API: stage from an uploaded frame of the same camera."""
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        return {"error": "send a JPEG or PNG frame from this camera"}
    r = reader.read(img)
    return {**public(r), "alert_level": reader.alert_level(r["stage_ft"])}
