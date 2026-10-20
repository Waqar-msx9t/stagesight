"""USGS data access: gage height time series and gauge camera images.

Gage height: USGS Water Services instantaneous values (parameter 00065, feet).
If this legacy endpoint is retired, use the new API at https://api.waterdata.usgs.gov/.

Camera images: USGS National Imagery Management System (NIMS) API,
base URL https://api.waterdata.usgs.gov/nims/v0 (docs: https://api.waterdata.usgs.gov/nims/v0/docs).
A free API key from https://api.waterdata.usgs.gov/signup/ raises the rate limit (set API_USGS_PAT).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import requests

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
NIMS_BASE = "https://api.waterdata.usgs.gov/nims/v0"


def gage_height(site: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times as datetime64[s] UTC, gage height in feet)."""
    r = requests.get(IV_URL, params={"sites": site, "parameterCd": "00065",
                                     "startDT": start, "endDT": end, "format": "json"},
                     timeout=60)
    r.raise_for_status()
    values = r.json()["value"]["timeSeries"][0]["values"][0]["value"]
    t = np.array([np.datetime64(datetime.fromisoformat(v["dateTime"]).astimezone(timezone.utc)
                                .replace(tzinfo=None), "s") for v in values])
    h = np.array([float(v["value"]) for v in values])
    keep = h > -999          # USGS writes -999999 for ice, equipment and other no-data codes
    return t[keep], h[keep]


def stage_at(times, stages, when: np.datetime64, max_gap_min: int = 20) -> float | None:
    """Gage height interpolated to an image time; None if no reading within max_gap_min."""
    i = np.searchsorted(times, when)
    if i == 0 or i >= len(times):
        return None
    gap = min(abs(times[i] - when), abs(when - times[i - 1])) / np.timedelta64(1, "m")
    if gap > max_gap_min:
        return None
    t0, t1 = times[i - 1].astype(float), times[i].astype(float)
    w = (when.astype(float) - t0) / (t1 - t0)
    return float(stages[i - 1] + w * (stages[i] - stages[i - 1]))


def nims_get(path: str, **params) -> dict:
    """Thin wrapper for the NIMS API. Path and parameters come from the Swagger docs."""
    headers = {}
    if os.environ.get("API_USGS_PAT"):
        headers["X-Api-Key"] = os.environ["API_USGS_PAT"]
    r = requests.get(f"{NIMS_BASE}/{path.lstrip('/')}", params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def list_images(cam_id: str, after: str | None = None, before: str | None = None,
                size: str = "overlayDir", limit: int = 50000) -> list[str]:
    """Full image URLs for a camera, newest first. size: overlayDir (full), smallDir (720 px)."""
    base = nims_get("cameras", camId=cam_id)[0][size]
    params = {"camId": cam_id, "limit": limit}
    if after:
        params["after"] = after
    if before:
        params["before"] = before
    return [base + f for f in nims_get("listFiles", **params)]


def latest_image_url(cam_id: str) -> str:
    return list_images(cam_id, limit=1)[0]


def image_time(url: str) -> np.datetime64:
    """UTC time from a NIMS filename like CAM___2026-10-18T18-00-03Z.jpg."""
    d, t = url.rsplit("___", 1)[1].removesuffix(".jpg").removesuffix("Z").split("T")
    return np.datetime64(f"{d}T{t.replace('-', ':')}", "s")


def download(url: str, dest: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
