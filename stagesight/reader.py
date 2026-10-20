"""One call from a camera frame to a stage reading, used by the service, the agent and the API."""
from __future__ import annotations

import json
import os

import cv2 as cv
import numpy as np

from stagesight import plate

VISIBLE = (60, 200, 60)
HIDDEN = (200, 60, 200)


class SiteReader:
    """Loads a site folder (site.json + reference.jpg) made by scripts/setup_site.py."""

    def __init__(self, site_dir: str):
        self.site = json.load(open(os.path.join(site_dir, "site.json")))
        self.reference = cv.imread(os.path.join(site_dir, "reference.jpg"))
        self.plate = self.site["plate"]
        self.bars = plate.bar_masks(self.reference, self.plate)
        self.reg = plate.registration(self.reference, self.plate)

    def read(self, img: np.ndarray) -> dict:
        """Stage in ft (plate datum), bracket, status and the evidence behind it.

        Status is one of: ok, below_plate, unreadable, rejected, camera_moved, wrong_size.
        """
        if img.shape[:2] != self.reference.shape[:2]:
            # the plate geometry belongs to this camera's frame size; never guess on another
            return {"status": "wrong_size", "stage_ft": None, "mark_gap_ft": None, "shift_px": [0.0, 0.0],
                    "marks_visible": 0, "marks_total": len(self.bars), "mark_errors": None,
                    "expected_size": [self.reference.shape[1], self.reference.shape[0]], "_raw": {}}
        r = plate.read_bars(img, self.plate, self.bars, reg=self.reg)
        out = {"status": r["status"], "stage_ft": None, "mark_gap_ft": None,
               "shift_px": [round(float(v), 1) for v in r.get("shift", (0, 0))],
               "marks_visible": int(sum(r.get("visible", []))), "marks_total": len(self.bars),
               "mark_errors": r.get("errors"), "_raw": r}
        if r["status"] == "ok":
            datum = self.site["datum_ft"]
            out["stage_ft"] = round(datum - r["units"], 2)
            lo, hi = sorted(datum - u for u in r["bracket_units"])
            out["mark_gap_ft"] = [round(lo, 2), round(hi, 2)]
        elif r["status"] == "below_plate":
            # all but the lowest mark visible: the stage is below that mark
            low = self.bars[-1]["t_min"]
            out["below_ft"] = round(self.site["datum_ft"] - float(plate.to_units(self.plate, low)), 2)
        return out

    def alert_level(self, stage: float | None) -> str | None:
        if stage is None:
            return None
        flood, action = self.site.get("flood_stage_ft"), self.site.get("action_stage_ft")
        if flood is not None and stage >= flood:
            return "flood"
        if action is not None and stage >= action:
            return "action"
        return "normal"

    def draw(self, img: np.ndarray, result: dict, gauge_ft: float | None = None) -> np.ndarray:
        """Overlay for the dashboard and the video: marks seen (green) and hidden (magenta),
        the waterline, and the readings."""
        vis = img.copy()
        raw = result["_raw"]
        dx, dy = (int(round(v)) for v in raw.get("shift", (0, 0)))
        h, w = vis.shape[:2]
        # marks are drawn only when the reader could see the plate; on a night or fogged frame
        # every mark is "not visible", which must not look like "under water"
        marks = raw.get("visible", []) if result["status"] in ("ok", "below_plate") else []
        for b, seen in zip(self.bars, marks):
            vis[np.clip(b["ys"] + dy, 0, h - 1), np.clip(b["xs"] + dx, 0, w - 1)] = VISIBLE if seen else HIDDEN
        if result["status"] == "ok":
            o = np.array(self.plate["origin"]) + [dx, dy]
            d, n = np.array(self.plate["axis"]), np.array(self.plate["normal"])
            c = o + d * raw["t"]
            hw = self.plate["half_width"] * 2.2
            p0, p1 = (c - n * hw).astype(int), (c + n * hw).astype(int)
            cv.line(vis, tuple(p0), tuple(p1), (0, 0, 0), 9, cv.LINE_AA)
            cv.line(vis, tuple(p0), tuple(p1), (0, 220, 255), 5, cv.LINE_AA)
        lines = [f"camera: {result['stage_ft']:.2f} ft" if result["stage_ft"] is not None
                 else "camera: no reading (marks not visible)" if result["status"] == "unreadable"
                 else f"camera: {result['status'].replace('_', ' ')}"]
        if gauge_ft is not None:
            lines.append(f"gauge:  {gauge_ft:.2f} ft")
        scale = max(w, h) / 1400
        line_h = int(48 * scale)
        top = h - line_h * len(lines) - int(20 * scale)
        band = vis[top:h].astype(np.float32)
        vis[top:h] = (band * 0.35).astype(np.uint8)                 # dark banner for legibility
        for i, txt in enumerate(lines):
            org = (int(20 * scale), top + int(line_h * (i + 0.8)))
            cv.putText(vis, txt, org, cv.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255),
                       max(1, int(2.5 * scale)), cv.LINE_AA)
        return vis


def public(result: dict) -> dict:
    """The result without internal fields, safe to store or return from the API."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
