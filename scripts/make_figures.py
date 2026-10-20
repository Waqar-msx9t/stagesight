"""Report figures, rebuilt from the saved results and the two fixture frames (no recomputation
of the evaluation).

Usage:  python scripts/make_figures.py      ->  docs/figures/*.pdf (vector) and *.png
Needs matplotlib and SciencePlots (dev only, not in the Lambda package).
"""
import csv
import json
import os
import sys
from datetime import datetime

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" styles)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from stagesight import plate as P  # noqa: E402
from stagesight.reader import SiteReader  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import figcheck  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "legend.handlelength": 1.6,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# one palette for every figure
CAM, GAUGE, BASE, RULES = "#0C5DA5", "#111111", "#9a9a9a", "#7f7f7f"
VIS, HID, WL = "#3cc83c", "#c83cc8", "#d99a00"   # marks seen / under water (as reader.draw); WL on white
WL_PHOTO = "#ffdc00"                                # waterline on photos, with a black outline
FLOOD, ACTION = "#c62828", "#ef6c00"
FULL_W, COL_W = 7.16, 3.5                  # IEEE two-column widths (in)


ISSUES: list[str] = []


def _save(fig, name: str) -> None:
    ISSUES.extend(figcheck.overlaps(fig, name))      # every text, legend and mark measured after drawing
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s[:19])


def _rows(path: str) -> list[dict]:
    with open(os.path.join(ROOT, path), newline="") as f:
        return list(csv.DictReader(f))


def _label(ax, s: str) -> None:
    ax.text(0.0, 1.015, f"({s})", transform=ax.transAxes, va="bottom", ha="left", fontsize=8, fontweight="bold")


def pipeline_figure() -> dict:
    """Fig. 1: setup on the reference frame (a-c), reading a flood frame (d-e)."""
    site = SiteReader(os.path.join(ROOT, "sites", "hancock"))
    ref = site.reference
    flood = cv.imread(os.path.join(ROOT, "tests", "fixtures", "hancock_flood_17.61ft.jpg"))
    pl, datum = site.plate, site.site["datum_ft"]
    o, d, nrm = (np.array(pl[k]) for k in ("origin", "axis", "normal"))
    h, w = ref.shape[:2]
    cx = int(o[0] + d[0] * (pl["t_top"] + pl["t_bottom"]) / 2)
    x0, x1 = max(cx - 215, 0), min(cx + 215, w)
    # start below the camera's burned-in caption (top ~75 px of every NIMS frame)
    y0, y1 = int(max(o[1] + pl["t_top"] - 120, 115)), int(min(o[1] + pl["t_bottom"] + 220, h))
    rgb = lambda im: cv.cvtColor(im, cv.COLOR_BGR2RGB)[y0:y1, x0:x1]   # noqa: E731

    fig = plt.figure(figsize=(FULL_W, 3.05))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.85, 0.85, 1.45, 0.85, 1.45], wspace=0.5)

    # (a) red marks + robust axis on the reference frame
    a = fig.add_subplot(gs[0])
    red = P.red_mask(ref)
    show = rgb(ref).copy()
    show[red[y0:y1, x0:x1] > 0] = (255, 40, 40)
    a.imshow((0.55 * show + 0.45 * 255 * (red[y0:y1, x0:x1, None] > 0)).astype(np.uint8))
    ts = np.array([pl["t_top"] - 60, pl["t_bottom"] + 120])
    a.plot(o[0] + d[0] * ts - x0, o[1] + d[1] * ts - y0, color="cyan", lw=1.0)
    for t in pl["bars_t"]:
        p = o + d * t
        a.plot(p[0] - x0, p[1] - y0, "o", ms=2.6, mfc="none", mec="cyan", mew=0.8)
    a.set_xlim(0, x1 - x0)
    a.set_ylim(y1 - y0, 0)                                  # keep the axis line inside the photo
    _label(a, "a")

    # (b) OCR on the upright plate
    b = fig.add_subplot(gs[1])
    datum_read = P.read_datum(ref, pl, os.path.join(ROOT, "models", "crnn_en.onnx"))
    bx0 = max(cx - 300, 0)
    b.imshow(cv.cvtColor(ref, cv.COLOR_BGR2RGB)[y0:y1, bx0:min(cx + 215, w)])
    for r in datum_read["reads"]:
        t = pl["bars_t"][r["bar"]]
        nxt = pl["bars_t"][r["bar"] + 1] if r["bar"] + 1 < len(pl["bars_t"]) else t + 60
        p = o + d * (t + 0.42 * (nxt - t)) + nrm * pl["half_width"] * 2.4
        ok = r["text"].isdigit() and len(r["text"]) == 2          # only the longest reads vote
        b.text(p[0] - bx0, p[1] - y0, f"“{r['text']}”", fontsize=7, va="center", ha="left",
               color="white", bbox={"boxstyle": "round,pad=0.15", "fc": CAM if ok else BASE, "ec": "none"})
    _label(b, "b")

    # (c) 1-D projective map from the foot marks
    c = fig.add_subplot(gs[2])
    bt, bu = np.array(pl["bars_t"]), np.array(pl["bars_ft"])
    tt = np.linspace(bt.min() - 40, bt.max() + 220, 200)
    c.plot(tt, datum - P.to_units(pl, tt), color=CAM, lw=1.0, label="fit, Eq. (1)")
    c.plot(bt, datum - bu, "o", ms=3, mfc="white", mec=GAUGE, mew=0.7, label="foot marks")
    c.set_xlabel("Position along plate axis $t$ (px)")
    c.set_ylabel("Stage (ft)")
    c.legend(loc="lower left", frameon=False)
    c.text(0.90, 0.95, f"fit error {pl['fit_rmse_ft']:.2f} ft", transform=c.transAxes, fontsize=7, ha="right", va="top")
    _label(c, "c")

    # (d) flood frame: visible / hidden marks and the waterline
    r = site.read(flood)
    raw = r["_raw"]
    dd = fig.add_subplot(gs[3])
    im = rgb(flood).copy()
    dx, dy = (int(round(v)) for v in raw["shift"])
    for bar, seen in zip(site.bars, raw["visible"]):
        ys, xs = bar["ys"] + dy - y0, bar["xs"] + dx - x0
        k = (ys >= 0) & (ys < im.shape[0]) & (xs >= 0) & (xs < im.shape[1])
        im[ys[k], xs[k]] = (60, 200, 60) if seen else (200, 60, 200)        # VIS / HID (RGB)
    dd.imshow(im)
    cpt = o + [dx, dy] + d * raw["t"]
    hw = pl["half_width"] * 2.3
    dd.plot([cpt[0] - nrm[0] * hw - x0, cpt[0] + nrm[0] * hw - x0],
            [cpt[1] - nrm[1] * hw - y0, cpt[1] + nrm[1] * hw - y0], color=WL_PHOTO, lw=1.8,
            path_effects=[pe.withStroke(linewidth=3.2, foreground="black")])
    _label(dd, "d")

    # (e) per-mark Lab a* contrast, on the stage axis
    e = fig.add_subplot(gs[4])
    stage_mark = [datum - float(P.to_units(pl, (bar["t_min"] + bar["t_max"]) / 2)) for bar in site.bars]
    con = np.array(raw["contrast"])
    cols = [VIS if v else HID for v in raw["visible"]]
    e.scatter(con, stage_mark, s=9, c=cols, lw=0, zorder=3)
    e.axvline(8, color=GAUGE, lw=0.6, ls="--", gid="ref", zorder=1)
    e.axhline(r["stage_ft"], color=WL, lw=1.4, gid="ref", zorder=1)
    e.axhline(17.61, color=GAUGE, lw=0.7, ls=":", gid="ref", zorder=1)
    e.text(11, r["stage_ft"] + 1.0, f"camera {r['stage_ft']:.2f}", fontsize=7, ha="left", va="bottom", color="#8a6200", gid="on-line")
    e.text(11, 17.61 - 0.12, "gauge 17.61", fontsize=7, ha="left", va="top", gid="on-line")
    e.text(11, min(stage_mark), "threshold", fontsize=7, va="bottom")
    e.set_xlabel(r"Mark contrast $\Delta a^*$ (mark $-$ ring)")
    e.set_ylabel("Stage (ft)")
    e.text(0.97, 0.55, "visible", color=VIS, transform=e.transAxes, ha="right", fontsize=7)
    e.text(0.24, 0.18, "hidden", color=HID, transform=e.transAxes, ha="left", fontsize=7)
    _label(e, "e")

    for ax in (a, b, dd):
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, "pipeline")
    return {"datum": datum_read["datum_ft"], "votes": datum_read["votes"], "flood_read": r["stage_ft"]}


def flood_figure() -> dict:
    """Fig. 3: the Jul 29 flood, camera vs gauge, and the error."""
    series = [r for r in _rows("results/hancock_plate/series.csv") if r["split"] == "test"]
    t = [_t(r["time_utc"]) for r in series]
    obs = np.array([float(r["stage_obs"]) for r in series])
    cam = np.array([float(r["pred_ocr"]) for r in series])
    lin = np.array([float(r["pred_linear"]) for r in series])
    gauge = [r for r in _rows("results/hancock_plate/gauge_jul28-31.csv") if "2026-07-28T12" <= r["time_utc"][:13] <= "2026-07-31T12"]
    gt, gv = [_t(r["time_utc"]) for r in gauge], [float(r["stage_ft"]) for r in gauge]

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.22})
    for lvl, col, ls, txt in ((18, FLOOD, "--", "NWS minor flood (18 ft)"), (14, ACTION, "-.", "NWS action (14 ft)")):
        a.axhline(lvl, color=col, lw=0.7, ls=ls, gid="ref")
        a.text(_t("2026-07-31T20:00:00"), lvl + 0.15, txt, color=col, fontsize=7, ha="right", va="bottom", gid="on-line")
    a.axhspan(4, 11, color="#000000", alpha=0.04, lw=0)
    a.axhline(11, color="#777777", lw=0.4, gid="ref")
    a.text(_t("2026-07-28T00:00:00"), 7.0, "below the plate (< 11 ft):\nno camera reading", fontsize=7, color="#555555")
    a.plot(gt, gv, color=GAUGE, lw=0.9, label="USGS gauge")
    a.plot(t, cam, "o", ms=3, color=CAM, label="StageSight camera")
    a.plot(t, lin, "x", ms=3, mew=0.7, color=BASE, label="linear pixel fit")
    a.set_ylabel("Stage (ft)")
    a.set_ylim(6.5, 19.2)
    a.set_xlim(_t("2026-07-27T20:00:00"), _t("2026-07-31T23:59:00"))           # room for the legend before the rise
    a.xaxis.set_major_locator(mdates.DayLocator())
    a.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    a.legend(loc="center left", bbox_to_anchor=(0.0, (16 - 6.5) / (19.2 - 6.5)), frameon=False, handletextpad=0.4,
             labelspacing=0.25)                                 # between the 14 and 18 ft lines
    _label(a, "a")

    err = cam - obs
    b.axhspan(-0.5, 0.5, color=CAM, alpha=0.08, lw=0)
    b.axhline(0, color=GAUGE, lw=0.5)
    b.plot(obs, err, "o", ms=3, color=CAM)
    b.set_xlabel("Gauge stage (ft)")
    b.set_ylabel("Camera $-$ gauge (ft)")
    b.set_ylim(-0.8, 0.8)
    rmse, within = float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err) <= 0.5))
    b.text(0.04, 0.04, f"RMSE {rmse:.2f} ft\n{int(round(within * len(err)))} of {len(err)} within ±0.5 ft",
           transform=b.transAxes, fontsize=7)
    _label(b, "b")
    _save(fig, "flood")
    return {"n": len(err), "rmse": rmse, "within": within}


def timeline_figure() -> None:
    """Fig. 5: the frozen-gauge replay: what each instrument said and when each planner acted."""
    from stagesight.world import inject_fault
    def load(p):
        return [json.loads(l) for l in open(os.path.join(ROOT, f"results/agent/{p}/flatline/trace.jsonl"))]
    tr = {p: load(p) for p in ("claude", "rules")}
    lo, hi = "2026-07-29T06", "2026-07-30T02"
    rows = [r for r in tr["claude"] if lo <= r["time_utc"][:13] <= hi]
    gauge = [r for r in _rows("results/hancock_plate/gauge_jul28-31.csv") if lo <= r["time_utc"][:13] <= hi]
    gt = np.array([np.datetime64(r["time_utc"][:19]) for r in gauge])
    gv = np.array([float(r["stage_ft"]) for r in gauge])
    fv, _ = inject_fault(gt, gv, "flatline", "2026-07-29T13:00:00", 10, 0.0)   # as in scripts/replay_agent.py
    tt = [_t(str(x)) for x in gt]
    fig, a = plt.subplots(figsize=(FULL_W, 1.9))
    a.axvspan(_t("2026-07-29T13:00:00"), _t("2026-07-29T23:00:00"), color="#000000", alpha=0.05, lw=0)
    a.text(_t("2026-07-29T22:50:00"), 18.95, "fault: gauge frozen", fontsize=7, color="#555555",
           ha="right", va="top")
    a.axhline(18, color=FLOOD, lw=0.7, ls="--", gid="ref")
    a.text(_t("2026-07-30T03:30:00"), 17.88, "NWS minor flood (18 ft)", color=FLOOD, fontsize=7, va="top", ha="right",
           gid="on-line")
    a.plot(tt, gv, color=GAUGE, lw=0.8, ls=":", label="true stage (USGS record, held out)")
    a.plot(tt, fv, color=GAUGE, lw=1.1, label="USGS gauge as reported (frozen)")
    cam = [r for r in rows if r["camera"]["stage_ft"] is not None]
    a.plot([_t(r["time_utc"]) for r in cam], [r["camera"]["stage_ft"] for r in cam], "o", ms=3, color=CAM,
           label="StageSight camera")
    fault0 = _t("2026-07-29T13:00:00")
    for p, col, name, ha in (("claude", CAM, "Claude", "right"), ("rules", "#555555", "Rules", "left")):
        t = next(_t(r["time_utc"]) for r in tr[p] if any(x["type"] == "flag_sensor" for x in r["actions"]))
        a.axvline(t, color=col, lw=1.0, alpha=0.85, gid="ref", zorder=1)
        a.text(t, 11.55, f" {name} flags gauge, +{(t - fault0).total_seconds() / 3600:.0f} h ", color=col,
               fontsize=7, ha=ha, va="bottom", fontweight="bold", gid="on-line")
    a.set_ylim(11.2, 19.2)
    a.set_ylabel("Stage (ft)")
    a.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    a.set_xlabel("Time (UTC)")
    a.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.08))
    _save(fig, "timeline")


def faults_figure() -> None:
    """Fig. 6: the two simulated camera faults on real frames, the reader's output and the agent's verdict.
    Needs the downloaded archive (data/hancock); skipped when it is absent."""
    from stagesight.world import corrupt_frame, plate_band
    idx = os.path.join(ROOT, "data", "hancock", "images.csv")
    if not os.path.exists(idx):
        print("faults figure skipped: data/hancock not downloaded")
        return
    site = SiteReader(os.path.join(ROOT, "sites", "hancock"))
    paths = {r["time_utc"][:13]: os.path.join(ROOT, r["path"].replace("\\", "/")) for r in _rows("data/hancock/images.csv")}
    gauge = {r["time_utc"][:13]: float(r["stage_obs"]) for r in _rows("results/hancock_plate/series.csv")}
    cases = [("2026-07-29T18", None, None, "clean", "agent: no flag"),            # same specs as scripts/replay_agent.py
             ("2026-07-29T18", "occlusion", (16.6, 18.8), "debris", "agent: camera"),
             ("2026-07-30T10", "spray", (12.4, 14.8), "lens spray", "agent: camera")]
    pl, datum = site.plate, site.site["datum_ft"]
    o, d = np.array(pl["origin"]), np.array(pl["axis"])
    cx = int(o[0] + d[0] * (pl["t_top"] + pl["t_bottom"]) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.05), gridspec_kw={"wspace": 0.06})
    for ax, (hour, kind, band, title, verdict), lab in zip(axes, cases, "abc"):
        img = cv.imread(paths[hour])
        if kind:
            img = corrupt_frame(img, {"kind": kind, "seed": 1, "polygon": plate_band(pl, *band, datum).tolist()})
        r = site.read(img)
        vis = site.draw(img, r)[:, :, ::-1]
        y0 = int(max(o[1] + pl["t_top"] - 60, 115))
        ax.imshow(vis[y0:y0 + 1150, max(cx - 260, 0):cx + 260])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"$\\bf{{({lab})}}$ {title}", fontsize=7, pad=2)
        ax.set_xlabel(f"camera {r['stage_ft']:.2f} ft\ngauge {gauge[hour]:.2f} ft", fontsize=7, labelpad=2)
        ax.text(0.5, -0.27, verdict, transform=ax.transAxes, ha="center", va="top", fontsize=7,
                color=CAM if kind else "#555555", fontweight="bold" if kind else "normal")
    _save(fig, "faults")


def coverage_figure() -> dict:
    """Fig. 4: reader status for every hourly frame over 90 days (day x hour of day)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/agent_90d/rules/none/trace.jsonl"))]
    days = sorted({r["time_utc"][:10] for r in rows})
    code = {"ok": 3, "below_plate": 1, "unreadable": 2}
    grid = np.zeros((24, len(days)))
    for r in rows:
        grid[int(r["time_utc"][11:13]), days.index(r["time_utc"][:10])] = code.get(r["camera"]["status"], 0)
    cmap = ListedColormap(["#ffffff", "#c9d3df", "#5c6370", CAM])
    fig, a = plt.subplots(figsize=(FULL_W, 1.3))
    a.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ticks = [i for i, dname in enumerate(days) if dname.endswith("-01") or dname.endswith("-15")]
    a.set_xticks(ticks, [datetime.fromisoformat(days[i]).strftime("%b %d") for i in ticks])
    a.set_yticks([0, 6, 12, 18], ["00", "06", "12", "18"])
    for day, txt in (("2026-07-29", "Jul 29–31 flood:\ntest frames (n = 30)"), ("2026-09-04", "Sep 4–5 high water:\nbaseline training (n = 15)")):
        a.annotate(txt, xy=(days.index(day) + 1, 12), xytext=(days.index(day) + 5, 15.5), fontsize=7,
                   color="#1d2330", va="center", arrowprops={"arrowstyle": "-", "color": "#1d2330", "lw": 0.6})
    a.tick_params(which="both", top=False, right=False)
    a.minorticks_off()
    a.set_ylabel("Hour (UTC)")
    n = {k: sum(r["camera"]["status"] == k for r in rows) for k in code}
    a.legend(handles=[Patch(color=CAM, label=f"stage read ({n['ok']})"),
                      Patch(color="#c9d3df", label=f"water below plate ({n['below_plate']})"),
                      Patch(color="#5c6370", label=f"night or obscured ({n['unreadable']})"),
                      Patch(fc="white", ec="#999999", lw=0.5, label="no frame")],
             loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False,
             title=f"{len(rows):,} hourly frames, Jul 21 – Oct 18, 2026", title_fontsize=7)
    _save(fig, "coverage")
    return {"frames": len(rows), **n}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = flood_figure()
    assert m["n"] == 30 and abs(m["rmse"] - 0.257) < 0.01, m    # must match results/hancock_plate/metrics.json
    p = pipeline_figure()
    assert p["datum"] == 20.0 and abs(p["flood_read"] - 17.77) < 0.01, p
    timeline_figure()
    faults_figure()
    print(m, p, coverage_figure())
    for issue in ISSUES:
        print(issue)
    print(f"layout check: {len(ISSUES)} problems")
