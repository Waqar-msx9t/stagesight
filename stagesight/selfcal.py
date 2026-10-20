"""Self-calibration: find where the waterline moves in a camera view and place transects there.

Each frame is aligned to a reference frame and reduced to a few lighting-robust feature maps.
For every pixel, the correlation of each feature with USGS gage height over the training period
gives a stage-sensitivity map. Pixels the waterline sweeps across as the river rises and falls
correlate strongly; sky, far bank and open water do not. Transects are then laid across the
sensitive bands, so a new camera needs no hand-drawn lines.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from stagesight import vision

FEATURES = ("gray_z", "chroma_g", "chroma_b", "texture")


def frame_features(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Feature stack (C, h, w) that is less sensitive to overall lighting than raw intensity."""
    small = cv.resize(img, size, interpolation=cv.INTER_AREA).astype(np.float32)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    gray_z = (gray - gray.mean()) / (gray.std() + 1e-6)          # removes exposure changes
    s = small.sum(axis=2) + 1e-6
    chroma_g = small[..., 1] / s                                 # vegetation vs muddy water
    chroma_b = small[..., 0] / s
    mean = cv.blur(gray, (5, 5))
    texture = np.sqrt(np.maximum(cv.blur(gray * gray, (5, 5)) - mean * mean, 0))
    texture /= texture.mean() + 1e-6
    return np.stack([gray_z, chroma_g, chroma_b, texture])


def build_stack(paths: list[str], aligner: vision.Aligner | None,
                size: tuple[int, int] = (320, 180)) -> tuple[np.ndarray, np.ndarray]:
    """Aligned feature stack (N, C, h, w) as float16, and a mask of frames that passed
    quality and alignment."""
    feats, ok = [], []
    for p in paths:
        img = cv.imread(p)
        good = img is not None and vision.quality_check(img)["ok"]
        if good and aligner is not None:
            img, info = aligner.align(img)
            good = info["aligned"]
        ok.append(good)
        feats.append((frame_features(img, size) if good
                      else np.zeros((len(FEATURES), size[1], size[0]), np.float32)).astype(np.float16))
    return np.stack(feats), np.array(ok)


def sensitivity_map(stack: np.ndarray, stage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel |Pearson r| with stage, max over features. Returns (map h x w, best feature index)."""
    y = (stage - stage.mean()).astype(np.float32)
    r = []
    for c in range(stack.shape[1]):                              # one feature at a time keeps memory low
        x = stack[:, c].astype(np.float32)
        x -= x.mean(axis=0)
        num = np.tensordot(y, x, axes=(0, 0))
        den = np.sqrt((x ** 2).sum(axis=0) * (y ** 2).sum()) + 1e-6
        r.append(cv.GaussianBlur(np.abs(num / den), (5, 5), 0))
    r = np.stack(r)                                              # (C, h, w)
    return r.max(axis=0), r.argmax(axis=0)


def static_mask(stack: np.ndarray, full_shape: tuple[int, int], keep: float = 0.35) -> np.ndarray:
    """uint8 mask of the steadiest `keep` share of pixels (lowest frame-to-frame change of
    normalised grey level): walls, plates, banks and buildings, not water or sky."""
    g = stack[:, FEATURES.index("gray_z")].astype(np.float32)
    jitter = np.median(np.abs(np.diff(g, axis=0)), axis=0)
    mask = (jitter <= np.quantile(jitter, keep)).astype(np.uint8) * 255
    mask = cv.erode(mask, np.ones((3, 3), np.uint8))
    return cv.resize(mask, (full_shape[1], full_shape[0]), interpolation=cv.INTER_NEAREST)


def flood_stage_map(stack: np.ndarray, stage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel, the stage at which its appearance switches (the stage it goes under water).

    Frames are sorted by stage; for each pixel and feature the split of that sequence into a
    low-stage and a high-stage group with the largest between-group variance gives the switching
    stage. The feature with the cleanest split wins. Returns (flood stage in ft, strength 0..1),
    where strength is the explained variance of the split. Pixels the waterline crosses have
    high strength, and along any line from water to dry land their flood stage rises: an
    image-space rating curve learned from the archive alone.
    """
    order = np.argsort(stage)
    s = stage[order]
    n = len(s)
    k = np.arange(1, n)[:, None, None].astype(np.float32)       # size of the low-stage group
    best_r2 = np.zeros(stack.shape[2:], np.float32)
    best_stage = np.zeros(stack.shape[2:], np.float32)
    lo, hi = int(0.05 * n), int(0.95 * n)                       # ignore splits on a handful of frames
    for c in range(stack.shape[1]):
        x = stack[order, c].astype(np.float32)
        x = np.stack([cv.GaussianBlur(f, (3, 3), 0) for f in x])
        tot = x.sum(0)
        csum = np.cumsum(x, axis=0)[:-1]                        # sums of the first k frames
        between = (csum - k * tot / n) ** 2 * n / (k * (n - k))
        var = ((x - x.mean(0)) ** 2).sum(0) + 1e-6
        r2 = between[lo:hi] / var
        kbest = r2.argmax(0)
        r2max = np.take_along_axis(r2, kbest[None], 0)[0]
        better = r2max > best_r2
        best_r2[better] = r2max[better]
        best_stage[better] = ((s[kbest + lo] + s[kbest + lo + 1]) / 2)[better]
    return best_stage, best_r2


def _clip_ray(p: np.ndarray, d: np.ndarray, length: float, w: int, h: int) -> np.ndarray:
    """End point of a ray from p along unit vector d, stopped at `length` or the image border."""
    t = length
    for k, lim in ((0, w - 1), (1, h - 1)):
        if d[k] > 1e-9:
            t = min(t, (lim - p[k]) / d[k])
        elif d[k] < -1e-9:
            t = min(t, -p[k] / d[k])
    return p + max(t, 0.0) * d


def gradient_transects(fmap: np.ndarray, strength: np.ndarray, full_shape: tuple[int, int],
                       min_strength: float = 0.5, n_transects: int = 6, dry_len: float = 0.45,
                       wet_len: float = 0.08, min_area: int = 30) -> list[list[list[float]]]:
    """Transects that follow the flood-stage map uphill, ordered [dry end, water end].

    The strongest region is taken; its mean flood-stage gradient gives the direction the
    waterline moves (across a bank, or up a wall or staff plate). Transects are laid along that
    direction through points spread across the region. They reach `dry_len` of the image
    diagonal uphill, so floods far above the training range stay on the line, and `wet_len`
    into the water, which is enough for the water-colour reference.
    """
    h, w = fmap.shape
    sy, sx = full_shape[0] / h, full_shape[1] / w
    mask = (strength >= min_strength * strength.max()).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    comps = [i for i in range(1, n) if stats[i, cv.CC_STAT_AREA] >= min_area]
    if not comps:
        return []
    i = max(comps, key=lambda j: strength[labels == j].sum())
    region = labels == i
    f = cv.GaussianBlur(fmap, (7, 7), 0)
    gx, gy = cv.Sobel(f, cv.CV_32F, 1, 0, ksize=5), cv.Sobel(f, cv.CV_32F, 0, 1, ksize=5)
    wgt = strength[region]
    g = np.array([(gx[region] * wgt).sum(), (gy[region] * wgt).sum()])
    up = g / (np.linalg.norm(g) + 1e-9)                         # towards higher flood stage = dry
    across = np.array([-up[1], up[0]])
    ys, xs = np.nonzero(region)
    pts = np.c_[xs, ys].astype(np.float32)
    centre = (pts * wgt[:, None]).sum(0) / wgt.sum()
    a = (pts - centre) @ across
    diag = np.hypot(h, w)
    along = (pts - centre) @ up
    out = []
    for q in np.linspace(15, 85, n_transects):
        c = centre + np.percentile(a, q) * across
        c_wet = c + np.percentile(along, 5) * up                 # low edge of the band
        p_dry = _clip_ray(c_wet, up, dry_len * diag, w, h)
        p_wet = _clip_ray(c_wet, -up, wet_len * diag, w, h)
        out.append([[float(p_dry[0] * sx), float(p_dry[1] * sy)],
                    [float(p_wet[0] * sx), float(p_wet[1] * sy)]])
    return out


def auto_transects(smap: np.ndarray, full_shape: tuple[int, int], threshold: float = 0.75,
                   n_per_band: int = 3, max_bands: int = 2, land_map: np.ndarray | None = None,
                   land_len: float = 0.2, water_pad: float = 3.0,
                   min_area: int = 40) -> list[list[list[float]]]:
    """Lay transects across the most stage-sensitive bands.

    A band is a connected region above `threshold` times the map maximum. The waterline sweeps
    across a band as stage changes, so each transect runs along the band's short axis (PCA).
    With `land_map` (higher = more land-like, e.g. mean green chromaticity), each transect is
    oriented [land end, water end]: it reaches `water_pad` band half-widths into the channel
    (always water) and `land_len` of the image height up the bank, so stages far above the
    training range still fall on the line. Coordinates are in full-resolution pixels.
    """
    h, w = smap.shape
    sy, sx = full_shape[0] / h, full_shape[1] / w
    mask = (smap >= threshold * smap.max()).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    order = sorted(range(1, n), key=lambda i: -smap[labels == i].sum())
    transects = []
    for i in order[:max_bands]:
        if stats[i, cv.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.c_[xs, ys].astype(np.float32)
        weights = smap[ys, xs]
        centre = (pts * weights[:, None]).sum(0) / weights.sum()
        cov = np.cov((pts - centre).T, aweights=weights)
        evals, evecs = np.linalg.eigh(cov)                       # ascending
        minor, major = evecs[:, 0], evecs[:, 1]
        along = (pts - centre) @ major
        across = (pts - centre) @ minor
        half = max((np.percentile(across, 95) - np.percentile(across, 5)) / 2, 2.0)
        for q in np.linspace(20, 80, n_per_band):
            c = centre + np.percentile(along, q) * major
            if land_map is None:
                p0, p1 = c - 2.5 * half * minor, c + 2.5 * half * minor
            else:
                probe = [np.clip(c + s * 3 * half * minor, 0, [w - 1, h - 1]) for s in (1, -1)]
                land_dir = minor if (land_map[int(probe[0][1]), int(probe[0][0])] >=
                                     land_map[int(probe[1][1]), int(probe[1][0])]) else -minor
                p0 = c + land_dir * land_len * h                 # land end (p_top)
                p1 = c - land_dir * water_pad * half             # water end (p_bottom)
            p0 = np.clip(p0, 0, [w - 1, h - 1])
            p1 = np.clip(p1, 0, [w - 1, h - 1])
            transects.append([[float(p0[0] * sx), float(p0[1] * sy)],
                              [float(p1[0] * sx), float(p1[1] * sy)]])
    return transects


def heatmap(smap: np.ndarray, background: np.ndarray, transects: list | None = None) -> np.ndarray:
    """Overlay of the sensitivity map (and transects) on a frame, for the report and dashboard."""
    bg = background.copy()
    heat = cv.applyColorMap((np.clip(smap, 0, 1) * 255).astype(np.uint8), cv.COLORMAP_INFERNO)
    heat = cv.resize(heat, (bg.shape[1], bg.shape[0]), interpolation=cv.INTER_LINEAR)
    out = cv.addWeighted(bg, 0.45, heat, 0.55, 0)
    for t in transects or []:
        cv.line(out, tuple(map(int, t[0])), tuple(map(int, t[1])), (255, 255, 0), 3)
    return out
