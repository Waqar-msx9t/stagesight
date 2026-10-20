"""Staff-plate mode: use the plate's own printed graduations as the ruler.

The plate's red bars are found with an HSV mask, the plate axis with a robust line fit, and the
bars' positions along the axis give the camera's 1D projective map from pixels to plate units.
Only the plate offset (the stage of one bar) is taken from the gauge, so a flood far above the
calibration period is read with the plate's printed scale, not an extrapolated curve.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from stagesight import vision


def red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 80))
    return cv.morphologyEx(m.astype(np.uint8) * 255, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_plate(reference: np.ndarray, bar_spacing_ft: float = 1.0, min_bars: int = 5) -> dict | None:
    """Locate the plate and fit pixel position along its axis -> plate units (ft from top bar).

    Returns None when no plate-like row of red bars is found.
    """
    red = red_mask(reference)
    ys, xs = np.nonzero(red)
    if len(xs) < 500:
        return None
    vx, vy, x0, y0 = cv.fitLine(np.c_[xs, ys].astype(np.float32), cv.DIST_HUBER, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy]) * (1 if vy > 0 else -1)              # axis points down the plate
    nrm = np.array([-d[1], d[0]])
    o = np.array([x0, y0])
    rel = np.c_[xs, ys] - o
    s = rel @ nrm
    # bars = red connected components; each bar's centroid projected on the axis
    n, _, stats, cents = cv.connectedComponentsWithStats(red)
    areas = stats[1:, cv.CC_STAT_AREA]
    keep = areas >= 0.6 * np.median(np.sort(areas)[-8:])        # long (foot) bars only
    tc = np.sort((cents[1:][keep] - o) @ d)
    peaks = []
    for t in tc:                                                 # merge pieces of one bar
        if peaks and t - peaks[-1][-1] < 25:
            peaks[-1].append(t)
        else:
            peaks.append([t])
    peaks = np.array([np.mean(p) for p in peaks])
    if len(peaks) < min_bars:
        return None
    # index the bars top to bottom, allowing for a missing bar where a gap is ~2x the local spacing
    idx = [0]
    for k in range(1, len(peaks)):
        gaps = np.diff(peaks[max(0, k - 3):k + 1])
        local = np.median(gaps[:-1]) if len(gaps) > 1 else gaps[-1]
        idx.append(idx[-1] + max(1, int(round((peaks[k] - peaks[k - 1]) / local))))
    units = np.array(idx, float) * bar_spacing_ft
    # 1D homography: units = (a t + b) / (c t + 1)
    A = np.c_[peaks, np.ones_like(peaks), -peaks * units]
    coef, *_ = np.linalg.lstsq(A, units, rcond=None)
    resid = (coef[0] * peaks + coef[1]) / (coef[2] * peaks + 1) - units
    half_w = np.percentile(np.abs(s), 90)
    return {"origin": o.tolist(), "axis": d.tolist(), "normal": nrm.tolist(),
            "t_top": float(peaks[0] - 40), "t_bottom": float(peaks[-1]), "half_width": float(half_w),
            "bars_t": peaks.tolist(), "bars_ft": units.tolist(), "coef": coef.tolist(),
            "fit_rmse_ft": float(np.sqrt((resid ** 2).mean()))}


def read_datum(reference: np.ndarray, plate: dict, model_path: str,
               min_votes: int = 2) -> dict | None:
    """Read the printed foot numbers with the OpenCV DNN text recogniser (CRNN, OpenCV Zoo) and
    vote for the stage of the top foot bar, so the plate needs no gauge calibration at all.

    The plate is rotated upright, red bars are whitened, and the black digits under each foot
    bar are cropped. Every clean read gives a candidate number + bar offset; the most common
    candidate with at least `min_votes` wins, so a few misreads cannot move the datum.
    """
    rec = cv.dnn.TextRecognitionModel(model_path)
    rec.setDecodeType("CTC-greedy")
    rec.setVocabulary(list("0123456789abcdefghijklmnopqrstuvwxyz"))
    rec.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5))
    o, d = np.array(plate["origin"]), np.array(plate["axis"])
    rot = cv.warpAffine(reference, cv.getRotationMatrix2D(tuple(map(float, o)),
                                                          -float(np.degrees(np.arctan2(d[0], d[1]))), 1.0),
                        (reference.shape[1], reference.shape[0]))
    hw = plate["half_width"]
    x0, x1 = int(max(o[0] - 1.2 * hw, 0)), int(min(o[0] + 3.2 * hw, reference.shape[1]))
    bt, bu = plate["bars_t"], plate["bars_ft"]
    reads = []
    for i, t in enumerate(bt):
        gap = (bt[i + 1] - t) / (bu[i + 1] - bu[i]) if i + 1 < len(bt) else bt[-1] - bt[-2]
        y0, y1 = int(max(o[1] + t + 0.05 * gap, 0)), int(o[1] + t + 0.8 * gap)
        crop = rot[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue
        hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
        crop[((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 70)] = 255
        g = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        dark = (g < 90).astype(np.uint8)
        nc, _, st, _ = cv.connectedComponentsWithStats(dark)
        big = [k for k in range(1, nc) if st[k, 4] > 0.02 * dark.size and st[k, 3] > 0.3 * crop.shape[0]]
        if not big:
            continue
        xa = min(st[k, 0] for k in big)
        xb = max(st[k, 0] + st[k, 2] for k in big)
        ya = min(st[k, 1] for k in big)
        yb = max(st[k, 1] + st[k, 3] for k in big)
        digit = g[max(ya - 8, 0):yb + 8, max(xa - 8, 0):xb + 8]
        digit = cv.copyMakeBorder(digit, 10, 10, 14, 14, cv.BORDER_CONSTANT, value=255)
        txt = rec.recognize(cv.cvtColor(digit, cv.COLOR_GRAY2BGR))
        reads.append({"bar": i, "text": txt})
    digits = [r for r in reads if r["text"].isdigit()]
    longest = max((len(r["text"]) for r in digits), default=0)
    # a dropped leading digit is the usual misread, so only the longest reads may vote
    votes = [int(r["text"]) + bu[r["bar"]] for r in digits if len(r["text"]) == longest]
    if not votes:
        return None
    vals, counts = np.unique(np.round(votes, 2), return_counts=True)
    best = int(counts.argmax())
    if counts[best] < min_votes:
        return None
    return {"datum_ft": float(vals[best]), "votes": int(counts[best]), "reads": reads}


def to_units(plate: dict, t) -> np.ndarray:
    a, b, c = plate["coef"]
    t = np.asarray(t, float)
    return (a * t + b) / (c * t + 1)


def transects(plate: dict, img_shape: tuple[int, int], below: float = 250.0,
              offsets=(-0.4, 0.0, 0.4)) -> list:
    """Lines along the plate face, [top of plate, below the plate bottom], across its width."""
    o, d, nrm = (np.array(plate[k]) for k in ("origin", "axis", "normal"))
    h, w = img_shape
    out = []
    for f in offsets:
        shift = nrm * f * plate["half_width"]
        top = o + d * plate["t_top"] + shift
        bot = o + d * (plate["t_bottom"] + below) + shift
        bot = np.clip(bot, 0, [w - 1, h - 1])
        out.append([top.tolist(), bot.tolist()])
    return out


def bar_masks(reference: np.ndarray, plate: dict, min_area: int = 120) -> list[dict]:
    """Every red mark on the plate in the reference frame (foot and half-foot bars), with its
    pixels and its extent along the plate axis."""
    red = red_mask(reference)
    o, d, nrm = (np.array(plate[k]) for k in ("origin", "axis", "normal"))
    n, labels, stats, cents = cv.connectedComponentsWithStats(red)
    bars = []
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < min_area:
            continue
        if abs((cents[i] - o) @ nrm) > 2.0 * plate["half_width"]:
            continue
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        ring = cv.dilate(comp, np.ones((15, 15), np.uint8)) - cv.dilate(comp, np.ones((5, 5), np.uint8))
        ry, rx = np.nonzero(ring & (red == 0))
        t = (np.c_[xs, ys] - o) @ d
        bars.append({"ys": ys, "xs": xs, "ry": ry, "rx": rx,
                     "t_min": float(t.min()), "t_max": float(t.max())})
    return sorted(bars, key=lambda b: b["t_min"])


def registration(reference: np.ndarray, plate: dict, dry_share: float = 0.3, pad: int = 60) -> dict:
    """Template for re-locating the plate in each frame: the top `dry_share` of the plate (above
    any flood on record) plus a margin, as a windowed grey crop."""
    o, d = np.array(plate["origin"]), np.array(plate["axis"])
    t0, t1 = plate["t_top"], plate["t_top"] + dry_share * (plate["t_bottom"] - plate["t_top"])
    pts = np.array([o + d * t0, o + d * t1])
    x0 = int(max(pts[:, 0].min() - plate["half_width"] * 2 - pad, 0))
    x1 = int(min(pts[:, 0].max() + plate["half_width"] * 2 + pad, reference.shape[1]))
    y0, y1 = int(max(pts[:, 1].min() - pad, 0)), int(min(pts[:, 1].max() + pad, reference.shape[0]))
    crop = cv.cvtColor(reference[y0:y1, x0:x1], cv.COLOR_BGR2GRAY).astype(np.float32)
    return {"box": (x0, y0, x1, y1), "ref": crop,
            "win": cv.createHanningWindow((x1 - x0, y1 - y0), cv.CV_32F)}


def plate_shift(img: np.ndarray, reg: dict) -> tuple[float, float, float]:
    """(dx, dy, response) that moves reference coordinates onto this frame (phase correlation)."""
    x0, y0, x1, y1 = reg["box"]
    cur = cv.cvtColor(img[y0:y1, x0:x1], cv.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), resp = cv.phaseCorrelate(reg["ref"], cur, reg["win"])
    return float(dx), float(dy), float(resp)


def read_bars(img: np.ndarray, plate: dict, bars: list[dict], min_contrast: float = 8.0,
              reg: dict | None = None, max_shift: float = 80.0) -> dict:
    """Waterline from which plate marks are still visible.

    A bar is visible when it is redder (Lab a*) than the plate right around it by
    `min_contrast`; this relative test holds in sun, shade and overcast, and water or foam
    over a bar gives no contrast.

    Bars are ordered top to bottom; the split that best separates visible (above) from hidden
    (below) is found, tolerating a few wrong bars (glare, a leaf). The waterline is placed midway
    between the lowest visible bar's bottom edge and the next bar's top edge.
    Returns units (plate ft below the top foot bar) or a status when the water is off the plate.
    """
    q = vision.quality_check(img)
    if not q["ok"]:
        return {"status": "rejected", "quality": q, "units": None}
    dx = dy = 0.0
    if reg is not None:
        dx, dy, _ = plate_shift(img, reg)
        if np.hypot(dx, dy) > max_shift:
            return {"status": "camera_moved", "shift": (dx, dy), "units": None}
    a_star = cv.cvtColor(img, cv.COLOR_BGR2Lab)[..., 1].astype(np.float32)
    h, w = a_star.shape
    ix, iy = int(round(dx)), int(round(dy))

    def at(ys, xs):
        return a_star[np.clip(ys + iy, 0, h - 1), np.clip(xs + ix, 0, w - 1)]

    contrast = np.array([np.median(at(b["ys"], b["xs"])) - np.median(at(b["ry"], b["rx"]))
                         for b in bars])
    vis = contrast >= min_contrast
    n = len(vis)
    # cost of split k: hidden bars above + visible bars below
    cost = [(~vis[:k]).sum() + vis[k:].sum() for k in range(n + 1)]
    k = int(np.argmin(cost))
    out = {"shift": (dx, dy), "visible": vis.tolist(), "contrast": contrast.round(1).tolist(), "split": k,
           "errors": int(cost[k]), "units": None}
    if k >= n - 1:
        # only the lowest bar hidden is ambiguous: wet, shaded or splashed at low water
        out["status"] = "below_plate"
    elif k == 0:
        out["status"] = "unreadable"      # no mark visible: dark, fog, spray or lens cover
    else:
        # midpoint of the gap between the lowest visible and the first hidden mark.
        # refine() (sub-mark edge search) was tested on the Jul 29 flood and did not beat this
        # (0.275 vs 0.26 ft RMSE), so it is kept for experiments only.
        t_lo, t_hi = bars[k - 1]["t_max"], bars[k]["t_min"]
        t = (t_lo + t_hi) / 2
        out.update(status="ok", t=float(t), units=float(to_units(plate, t)),
                   bracket_units=[float(to_units(plate, t_lo)), float(to_units(plate, t_hi))])
    return out


def _along_axis(img_ch: np.ndarray, plate: dict, shift, t0: float, t1: float,
                offsets: np.ndarray, n: int) -> np.ndarray:
    """Samples of one channel along lines parallel to the plate axis: (len(offsets), n)."""
    o = np.array(plate["origin"]) + np.array(shift)
    d, nrm = np.array(plate["axis"]), np.array(plate["normal"])
    ts = np.linspace(t0, t1, n, dtype=np.float32)
    rows = []
    for f in offsets:
        p = o + nrm * f
        xs = (p[0] + d[0] * ts).astype(np.float32).reshape(1, -1)
        ys = (p[1] + d[1] * ts).astype(np.float32).reshape(1, -1)
        rows.append(cv.remap(img_ch, xs, ys, cv.INTER_LINEAR).ravel())
    return np.array(rows)


def refine(img: np.ndarray, plate: dict, bars: list[dict], k: int, shift, t_lo: float, t_hi: float,
           pad: float = 15.0, n: int = 200) -> float:
    """Exact waterline inside the bracket between the lowest visible bar and the first hidden one.

    Plate pixels are white or red; water is neither. Thresholds come from this frame: the white
    plate around the visible bar and the hidden bar's area (which is under water). Along several
    lines across the plate width, the best split between plate above and water below is found;
    the median over lines is the waterline, kept inside the bracket.
    """
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
    L, A = lab[..., 0], lab[..., 1]
    ix, iy = int(round(shift[0])), int(round(shift[1]))
    h, w = L.shape

    def vals(ch, ys, xs):
        return ch[np.clip(ys + iy, 0, h - 1), np.clip(xs + ix, 0, w - 1)]

    vb, hb = bars[max(k - 2, 0)], bars[min(k + 1, len(bars) - 1)]   # surely dry / surely wet
    L_plate, L_water = np.median(vals(L, vb["ry"], vb["rx"])), np.median(vals(L, hb["ys"], hb["xs"]))
    A_red, A_water = np.median(vals(A, vb["ys"], vb["xs"])), np.median(vals(A, hb["ys"], hb["xs"]))
    l_thr, a_thr = (L_plate + L_water) / 2, (A_red + A_water) / 2
    offsets = np.linspace(-0.8, 0.8, 9) * plate["half_width"]
    t0, t1 = t_lo - pad, t_hi + pad
    Ls = _along_axis(L, plate, (ix, iy), t0, t1, offsets, n)
    As = _along_axis(A, plate, (ix, iy), t0, t1, offsets, n)
    is_plate = ((Ls > l_thr) | (As > a_thr)).astype(np.float32)
    cuts = []
    for row in is_plate:
        # split j: rows above should be plate, rows below water
        cost = np.cumsum(np.r_[0, 1 - row]) + np.cumsum(np.r_[0, row[::-1]])[::-1]
        cuts.append(int(np.argmin(cost)))
    t = t0 + (t1 - t0) * np.median(cuts) / (n - 1)
    return float(np.clip(t, t_lo, t_hi))


def read(img: np.ndarray, plate: dict, method: str = "both", min_confidence: float = 1.0) -> dict:
    """Waterline on the plate, in plate units (ft below the top bar). Higher water = smaller."""
    lines = transects(plate, img.shape[:2])
    m = vision.measure(img, lines, method=method, min_confidence=min_confidence)
    o, d = np.array(plate["origin"]), np.array(plate["axis"])
    m["units"] = None
    if m["status"] == "ok":
        ts = [float((np.array(r["pixel"]) - o) @ d) for r in m["transects"]
              if r["confidence"] >= min_confidence]
        m["units"] = float(to_units(plate, np.median(ts)))
    m["lines"] = lines
    return m
