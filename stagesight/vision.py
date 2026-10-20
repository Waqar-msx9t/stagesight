"""Research mode for cameras without a staff plate (see stagesight/selfcal.py and the report,
Sec. VIII): ORB alignment and transect waterline detectors. The plate reader does not use it."""
from __future__ import annotations

import cv2 as cv
import numpy as np

from stagesight.plate import quality_check  # noqa: F401  (re-exported for the research scripts)


class Aligner:
    """Warps each frame onto a reference frame so fixed transects stay valid if the camera moves."""

    def __init__(self, reference: np.ndarray, n_features: int = 2000,
                 mask: np.ndarray | None = None):
        """mask (uint8, reference size) limits reference keypoints to static scenery, so moving
        water, foam and debris cannot pull the homography during a flood."""
        self.orb = cv.ORB_create(n_features)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.ref_shape = reference.shape[:2]
        ref_gray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, mask)

    def align(self, img: np.ndarray, min_inliers: int = 25) -> tuple[np.ndarray, dict]:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or self.ref_des is None or len(kp) < min_inliers:
            return img, {"aligned": False, "inliers": 0, "shift_px": None}
        matches = sorted(self.matcher.match(des, self.ref_des), key=lambda m: m.distance)[:500]
        if len(matches) < min_inliers:
            return img, {"aligned": False, "inliers": len(matches), "shift_px": None}
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src, dst, cv.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if H is None or inliers < min_inliers:
            return img, {"aligned": False, "inliers": inliers, "shift_px": None}
        h, w = self.ref_shape
        warped = cv.warpPerspective(img, H, (w, h))
        centre = np.float32([[[w / 2, h / 2]]])
        shift = float(np.linalg.norm(cv.perspectiveTransform(centre, H) - centre))
        return warped, {"aligned": True, "inliers": inliers, "shift_px": shift}


def _sample_line(arr: np.ndarray, p0, p1, n: int) -> np.ndarray:
    xs = np.linspace(p0[0], p1[0], n, dtype=np.float32)
    ys = np.linspace(p0[1], p1[1], n, dtype=np.float32)
    return cv.remap(arr, xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_LINEAR).ravel()


def waterline_on_transect(img: np.ndarray, p_top, p_bottom, n: int = 400,
                          margin: float = 0.05) -> dict:
    """Find where a transect drawn from dry bank (p_top) down into the channel (p_bottom)
    crosses the water surface. Returns the position as a fraction 0..1 along the transect
    (0 = p_top) plus a confidence score (edge peak relative to the median edge strength)."""
    gray = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    mean = cv.blur(gray, (9, 9))
    texture = np.sqrt(np.maximum(cv.blur(gray * gray, (9, 9)) - mean * mean, 0))
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)

    profiles = [_sample_line(a, p_top, p_bottom, n) for a in (gray, texture, hsv[..., 1])]
    edge = np.zeros(n, dtype=np.float32)
    for prof in profiles:
        prof = cv.GaussianBlur(prof.reshape(1, -1), (1, 9), 0).ravel()
        d = np.abs(np.gradient(prof))
        edge += d / (d.max() + 1e-6)
    lo, hi = int(n * margin), int(n * (1 - margin))
    idx = lo + int(np.argmax(edge[lo:hi]))
    conf = float(edge[idx] / (np.median(edge[lo:hi]) + 1e-6))
    return {"fraction": idx / (n - 1), "confidence": conf,
            "pixel": (float(p_top[0] + (p_bottom[0] - p_top[0]) * idx / (n - 1)),
                      float(p_top[1] + (p_bottom[1] - p_top[1]) * idx / (n - 1)))}


def waterline_water_run(img: np.ndarray, p_top, p_bottom, n: int = 400,
                        water_frac: float = 0.1) -> dict:
    """Waterline from the frame's own water colour.

    p_bottom must lie in water at every stage. The first `water_frac` of the samples (from the
    water end) give this frame's water colour in Lab, so lighting and turbidity changes cancel.
    Each sample gets a distance to that colour, and the waterline is the single split that best
    separates a water-like run (from p_bottom) from a not-water run (towards p_top).
    Returns fraction 0..1 from p_top, like waterline_on_transect.
    """
    lab = cv.cvtColor(cv.GaussianBlur(img, (5, 5), 0), cv.COLOR_BGR2Lab).astype(np.float32)
    prof = np.stack([_sample_line(lab[..., c], p_bottom, p_top, n) for c in range(3)], axis=1)
    k0 = max(5, int(n * water_frac))
    ref = np.median(prof[:k0], axis=0)
    spread = np.median(np.abs(prof[:k0] - ref), axis=0) * 1.4826 + 2.0
    d = np.sqrt((((prof - ref) / spread) ** 2).sum(axis=1))
    d = cv.GaussianBlur(d.reshape(1, -1).astype(np.float32), (1, 9), 0).ravel()
    tau = 3.0                                                    # "not water" beyond 3 robust sd
    cost = np.cumsum(np.r_[0, d]) + (np.cumsum(np.r_[0, (tau - d)[::-1]])[::-1])
    k = int(np.argmin(cost[k0:])) + k0                          # split index from the water end
    k = min(k, n - 1)
    inside = d[k0:k].mean() if k > k0 else 0.0
    outside = d[k:].mean() if k < n else 0.0
    conf = float((outside - inside) / (inside + 1.0))
    frac_from_top = 1.0 - k / (n - 1)
    return {"fraction": frac_from_top, "confidence": conf,
            "pixel": (float(p_bottom[0] + (p_top[0] - p_bottom[0]) * k / (n - 1)),
                      float(p_bottom[1] + (p_top[1] - p_bottom[1]) * k / (n - 1)))}


def waterline_two_sided(img: np.ndarray, p_top, p_bottom, n: int = 400,
                        ref_frac: float = 0.1, tau: float = 3.0) -> dict:
    """Waterline as the best split between a water-like run and a dry-like run.

    p_bottom lies in water at every stage and p_top stays dry at every stage, so both ends give
    this frame's own reference colours (Lab). Each sample's distance to each reference is capped
    at `tau` robust sd, so foam, debris or glare that matches neither side costs the same on
    both sides and cannot drag the split. Returns fraction 0..1 from p_top.
    """
    lab = cv.cvtColor(cv.GaussianBlur(img, (5, 5), 0), cv.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    mean = cv.blur(L, (15, 15))
    texture = np.sqrt(np.maximum(cv.blur(L * L, (15, 15)) - mean * mean, 0)) * 2.0
    chans = [lab[..., 0], lab[..., 1], lab[..., 2], texture]    # texture splits foam from concrete
    prof = np.stack([_sample_line(c, p_bottom, p_top, n) for c in chans], axis=1)
    k0 = max(5, int(n * ref_frac))

    def dist(segment):
        ref = np.median(segment, axis=0)
        spread = np.median(np.abs(segment - ref), axis=0) * 1.4826 + 2.0
        d = np.sqrt((((prof - ref) / spread) ** 2).sum(axis=1))
        return np.minimum(cv.GaussianBlur(d.reshape(1, -1).astype(np.float32), (1, 9), 0).ravel(), tau)

    d_wet, d_dry = dist(prof[:k0]), dist(prof[-k0:])
    if np.median(d_wet[-k0:]) < 0.5 * tau:                        # dry end looks like water:
        return {"fraction": 0.0, "confidence": 0.0, "submerged": True,   # line is under water
                "pixel": (float(p_top[0]), float(p_top[1]))}
    cost = np.cumsum(np.r_[0, d_wet]) + np.cumsum(np.r_[0, d_dry[::-1]])[::-1]
    k = int(np.argmin(cost[k0:n - k0])) + k0
    gain = (d_dry[k0:k].mean() if k > k0 else 0) + (d_wet[k:n - k0].mean() if k < n - k0 else 0)
    loss = (d_wet[k0:k].mean() if k > k0 else 0) + (d_dry[k:n - k0].mean() if k < n - k0 else 0)
    conf = float(gain / (loss + 0.5))
    return {"fraction": 1.0 - k / (n - 1), "confidence": conf,
            "pixel": (float(p_bottom[0] + (p_top[0] - p_bottom[0]) * k / (n - 1)),
                      float(p_bottom[1] + (p_top[1] - p_bottom[1]) * k / (n - 1)))}


DETECTORS = {"edge": waterline_on_transect, "water": waterline_water_run,
             "both": waterline_two_sided}


def measure(img: np.ndarray, transects: list, aligner: Aligner | None = None,
            min_confidence: float = 3.0, method: str = "edge") -> dict:
    """Full per-frame measurement: quality gate, alignment, then median waterline over transects."""
    q = quality_check(img)
    out = {"quality": q, "alignment": None, "transects": [], "fraction": None, "status": "rejected"}
    if not q["ok"]:
        return out
    if aligner is not None:
        img, info = aligner.align(img)
        out["alignment"] = info
        if not info["aligned"]:
            out["status"] = "camera_moved"
            return out
    results = [DETECTORS[method](img, t[0], t[1]) for t in transects]
    out["transects"] = results
    good = [r["fraction"] for r in results if r["confidence"] >= min_confidence]
    if not good:
        out["status"] = "low_confidence"
        return out
    out["fraction"] = float(np.median(good))
    out["spread"] = float(np.ptp(good)) if len(good) > 1 else 0.0
    out["status"] = "ok"
    return out


def draw_overlay(img: np.ndarray, transects: list, m: dict, stage_text: str = "") -> np.ndarray:
    """Overlay transects and detected waterline points for the dashboard and the video."""
    vis = img.copy()
    for t, r in zip(transects, m.get("transects", [])):
        cv.line(vis, tuple(map(int, t[0])), tuple(map(int, t[1])), (255, 200, 0), 2)
        cv.circle(vis, tuple(map(int, r["pixel"])), 7, (0, 0, 255), -1)
    label = f"{m['status']}  {stage_text}".strip()
    cv.putText(vis, label, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(vis, label, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return vis
