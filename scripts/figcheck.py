"""Layout check for matplotlib figures: every text, legend and data mark is measured after drawing.

overlaps(fig) returns a list of problems:
  - two texts whose boxes intersect (titles, labels, tick labels, annotations, legend entries);
  - a text or legend box that covers a data point or crosses a plotted line or an arrow;
Figures are saved with bbox_inches="tight", so text beyond the axes is fine; tick labels of ticks
outside the view are never drawn and are skipped.
Photos (imshow) are ignored on purpose: labels on a photo are placed there deliberately.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text

PAD = 1.0          # px: text boxes may touch each other (tick labels sit close to axis labels)
CLEAR = 3.0        # px: a text or legend must keep this much space around data marks and arrows


def _tick_labels(fig) -> tuple[set, list]:
    """All tick-label Text objects, and the ones actually drawn (tick inside the view interval)."""
    every, drawn = set(), []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for lab in (tick.label1, tick.label2):
                    every.add(lab)
                    if lab.get_visible() and lab.get_text().strip() and lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                        drawn.append(lab)
    return every, drawn


def _texts(fig) -> list[Text]:
    every, drawn = _tick_labels(fig)
    out = list(drawn)
    for t in fig.findobj(Text):
        if t in every:
            continue
        if not t.get_visible() or not t.get_text().strip() or t.figure is None:
            continue
        if t.get_gid() == "allow-overlap":
            continue
        out.append(t)
    return out


def _sample(poly: np.ndarray) -> np.ndarray:
    if len(poly) < 2:
        return poly
    return np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) // 2))) for a, b in zip(poly[:-1], poly[1:])])


def _markers(fig) -> list[tuple[np.ndarray, float]]:
    """Centres and radii (px) of every plotted marker."""
    out = []
    px = fig.dpi / 72.0
    for ax in fig.axes:
        for ln in ax.get_lines():
            if ln.get_visible() and ln.get_marker() not in (None, "None", "", " ") and ln.get_gid() != "ref":
                xy = ln.get_transform().transform(ln.get_xydata())
                out.append((xy[np.isfinite(xy).all(1)], ln.get_markersize() * px / 2))
        for c in ax.collections:
            if isinstance(c, PathCollection) and c.get_visible() and len(c.get_offsets()):
                size = float(np.sqrt(np.max(c.get_sizes()))) if len(c.get_sizes()) else 6.0
                out.append((c.get_offset_transform().transform(c.get_offsets()), size * px / 2))
    return out


def _points(fig, r) -> list[tuple[str, np.ndarray, float]]:
    """Display-space samples (with a radius in px) of every data line, marker, arrow and frame."""
    pts = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_gid() == "allow-overlap":
                continue
            xy = ln.get_transform().transform(ln.get_xydata())      # units (dates) already converted
            xy = xy[np.isfinite(xy).all(1)]
            if ln.get_linestyle() not in ("None", "", " "):
                pts.append((f"line '{ln.get_label()}'" + (" [ref]" if ln.get_gid() == "ref" else ""), _sample(xy), 0.5))
        for spine in ax.spines.values():                # axis frames: text must not cross or touch them
            if spine.get_visible() and ax.axison:
                for poly in spine.get_path().to_polygons(spine.get_transform(), closed_only=False):
                    pts.append(("axis frame", _sample(poly), 0.5))
    for xy, rad in _markers(fig):
        pts.append(("marker", xy, rad))
    for p in fig.findobj(Rectangle):                 # drawn frames (diagram groups), not axes backgrounds
        if p.get_visible() and not p.get_fill() and p.get_linewidth() > 0 and p.axes is not None:
            for poly in p.get_path().to_polygons(p.get_transform()):
                pts.append(("frame", _sample(poly), 0.5))
    for p in fig.findobj(FancyArrowPatch):
        if p.get_visible() and p.get_gid() != "allow-overlap":
            # to_polygons honours the path codes (MOVETO, CLOSEPOLY), so heads are not joined to (0, 0)
            for poly in p.get_path().to_polygons(p.get_transform(), closed_only=False):
                pts.append(("arrow", _sample(poly), 0.5))
    return pts


def overlaps(fig, name: str = "") -> list[str]:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _texts(fig)
    boxes = []
    for t in texts:
        bb = t.get_window_extent(r)
        boxes.append((t, (bb.x0 + PAD, bb.y0 + PAD, bb.x1 - PAD, bb.y1 - PAD)))
    for lg in [a.get_legend() for a in fig.axes] + list(fig.legends):
        if lg is not None and lg.get_visible():
            bb = lg.get_window_extent(r)
            boxes.append((lg, (bb.x0 + PAD, bb.y0 + PAD, bb.x1 - PAD, bb.y1 - PAD)))
    issues = []

    def label(a):
        return "legend" if not isinstance(a, Text) else repr(a.get_text()[:40])

    def inside_legend(t, lg):
        return isinstance(t, Text) and not isinstance(lg, Text) and (t in lg.get_texts() or t is lg.get_title())

    for i, (a, A) in enumerate(boxes):
        for b, B in boxes[i + 1:]:
            if inside_legend(a, b) or inside_legend(b, a):
                continue
            if isinstance(a, Text) and isinstance(b, Text) and a.get_text() == b.get_text() and A == B:
                continue
            if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                issues.append(f"{name}: {label(a)} overlaps {label(b)}")
    tick_labels = set(_tick_labels(fig)[1])
    for what, xy, rad in _points(fig, r):
        if not len(xy):
            continue
        for a, A in boxes:
            if isinstance(a, Text) and a in tick_labels:
                continue
            if isinstance(a, Text) and a is getattr(a.axes, "title", None):
                continue
            # a label written along its own reference line (gid "on-line") may touch reference lines only
            if isinstance(a, Text) and a.get_gid() == "on-line" and what.endswith("[ref]"):
                continue
            c = (1.0 if what == "axis frame" else CLEAR + rad) + PAD
            hit = (xy[:, 0] > A[0] - c) & (xy[:, 0] < A[2] + c) & (xy[:, 1] > A[1] - c) & (xy[:, 1] < A[3] + c)
            if hit.any():
                issues.append(f"{name}: {label(a)} covers {what}")
    # reference lines (gid "ref", e.g. axvline flags) must not run through data markers
    refs = [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_gid() == "ref" and ln.get_visible()]
    for ln in refs:
        xy = _sample(ln.get_transform().transform(ln.get_xydata()))
        for centres, rad in _markers(fig):
            if not len(centres) or ln.get_zorder() < 2:            # drawn under the markers: fine
                continue
            d = np.min(np.hypot(*(xy[:, None, :] - centres[None, :, :]).transpose(2, 0, 1)), axis=0)
            if (d < rad * 0.9).any():
                issues.append(f"{name}: reference line '{ln.get_label()}' is drawn over a marker")
                break
    return issues
