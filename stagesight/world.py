"""What the agent can observe: camera frames, the site gauge and an upstream gauge.

ReplayWorld steps through archived data hour by hour, so the agent can be tested on a real
flood with injected gauge faults. LiveWorld (aws/handler.py) exposes the same methods on live
USGS feeds.
"""
from __future__ import annotations

import csv

import cv2 as cv
import numpy as np


def load_series(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    t = np.array([np.datetime64(r["time_utc"], "s") for r in rows])
    v = np.array([float(r["stage_ft"]) if r["stage_ft"] not in ("", "nan") else np.nan for r in rows])
    return t, v


class ReplayWorld:
    """Archived frames (images.csv) plus gauge and upstream series, seen up to `now` only."""

    def __init__(self, images_csv: str, gauge: tuple[np.ndarray, np.ndarray],
                 upstream: tuple[np.ndarray, np.ndarray] | None = None,
                 camera_fault: dict | None = None):
        rows = list(csv.DictReader(open(images_csv)))
        self.frame_t = np.array([np.datetime64(r["time_utc"], "s") for r in rows])
        self.frame_p = [r["path"] for r in rows]
        order = np.argsort(self.frame_t)
        self.frame_t, self.frame_p = self.frame_t[order], [self.frame_p[i] for i in order]
        self.gauge_t, self.gauge_v = gauge
        self.up_t, self.up_v = upstream if upstream is not None else (np.array([], "datetime64[s]"), np.array([]))
        self.now = self.frame_t[0]
        self.camera_fault = camera_fault

    # ---- camera ----------------------------------------------------------------------------
    def frame(self, back: int = 0) -> tuple[np.datetime64, np.ndarray] | None:
        """The `back`-th most recent frame at or before now (0 = latest)."""
        i = int(np.searchsorted(self.frame_t, self.now, side="right")) - 1 - back
        if i < 0:
            return None
        img = cv.imread(self.frame_p[i])
        if img is None:
            return None
        f = self.camera_fault
        if f and np.datetime64(f["start"]) <= self.frame_t[i] < np.datetime64(f["end"]):
            img = corrupt_frame(img, f)
        return (self.frame_t[i], img)

    # ---- gauges ----------------------------------------------------------------------------
    @staticmethod
    def _window(t, v, now, hours):
        m = (t <= now) & (t > now - np.timedelta64(int(hours * 3600), "s")) & ~np.isnan(v)
        return [(str(a), round(float(b), 2)) for a, b in zip(t[m], v[m])]

    def gauge(self, hours: float = 3) -> list[tuple[str, float]]:
        return self._window(self.gauge_t, self.gauge_v, self.now, hours)

    def gauge_now(self, max_age_min: int = 30) -> float | None:
        recent = self._window(self.gauge_t, self.gauge_v, self.now, max_age_min / 60)
        return recent[-1][1] if recent else None

    def upstream(self, hours: float = 12) -> list[tuple[str, float]]:
        return self._window(self.up_t, self.up_v, self.now, hours)


def plate_band(plate: dict, ft_lo: float, ft_hi: float, datum: float, widen: float = 1.6) -> np.ndarray:
    """Image polygon covering the plate between two stages (for camera-fault simulation)."""
    a, b, c = plate["coef"]
    o, d, n = (np.array(plate[k]) for k in ("origin", "axis", "normal"))
    hw = plate["half_width"] * widen

    def t_of(stage):                       # invert units = (a t + b)/(c t + 1), units = datum - stage
        u = datum - stage
        return (u - b) / (a - c * u)
    t1, t2 = t_of(ft_hi), t_of(ft_lo)
    pts = [o + d * t1 - n * hw, o + d * t1 + n * hw, o + d * t2 + n * hw, o + d * t2 - n * hw]
    return np.array(pts, np.int32)


def corrupt_frame(img: np.ndarray, fault: dict) -> np.ndarray:
    """Camera-side faults on a real frame. occlusion: debris/leaf mat over part of the plate.
    spray: droplets and haze on the lens over the same band (marks lose their colour)."""
    out = img.copy()
    poly = np.array(fault["polygon"], np.int32)
    rng = np.random.default_rng(int(fault.get("seed", 0)))
    if fault["kind"] == "occlusion":
        cv.fillPoly(out, [poly], (32, 52, 70))                          # wet brown debris
        noise = rng.integers(0, 25, out.shape, dtype=np.uint8)
        mask = np.zeros(out.shape[:2], np.uint8)
        cv.fillPoly(mask, [poly], 255)
        out[mask > 0] = cv.add(out, noise)[mask > 0]
    elif fault["kind"] == "spray":
        x, y, w, h = cv.boundingRect(poly)
        haze = out.copy()
        for _ in range(400):
            cx, cy = int(rng.integers(x, x + w)), int(rng.integers(y, y + h))
            cv.circle(haze, (cx, cy), int(rng.integers(6, 22)), (215, 215, 210), -1, cv.LINE_AA)
        haze = cv.GaussianBlur(haze, (31, 31), 0)
        mask = np.zeros(out.shape[:2], np.uint8)
        cv.fillPoly(mask, [poly], 255)
        mask = cv.GaussianBlur(mask, (41, 41), 0).astype(np.float32)[..., None] / 255
        out = (out * (1 - 0.97 * mask) + haze * 0.97 * mask).astype(np.uint8)
    else:
        raise ValueError(f"unknown camera fault: {fault['kind']}")
    return out


def inject_fault(t: np.ndarray, v: np.ndarray, kind: str, start: str, hours: float,
                 size: float = 1.5) -> tuple[np.ndarray, dict]:
    """Copy of the gauge series with one sensor fault, plus its ground-truth window.

    drift    - reading creeps away from truth, reaching +size ft at the end of the window
    offset   - reading jumps by +size ft (e.g. a bumped or re-seated sensor)
    flatline - reading freezes at its value when the fault starts
    dropout  - no data (power or telemetry loss)
    """
    v = v.copy()
    t0 = np.datetime64(start, "s")
    t1 = t0 + np.timedelta64(int(hours * 3600), "s")
    m = (t >= t0) & (t < t1)
    if kind == "drift":
        frac = (t[m] - t0) / (t1 - t0)
        v[m] += size * frac.astype(float)
    elif kind == "offset":
        v[m] += size
    elif kind == "flatline":
        before = v[(t < t0) & ~np.isnan(v)]
        v[m] = before[-1] if len(before) else v[m][0]
    elif kind == "dropout":
        v[m] = np.nan
    else:
        raise ValueError(f"unknown fault kind: {kind}")
    return v, {"kind": kind, "start": str(t0), "end": str(t1), "size_ft": size}
