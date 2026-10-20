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
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" styles)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from stagesight import plate as P          # noqa: E402
from stagesight.reader import SiteReader   # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# one palette for every figure
CAM, GAUGE, BASE, RULES = "#0C5DA5", "#111111", "#9a9a9a", "#7f7f7f"
VIS, HID, WL = "#2ca02c", "#c2185b", "#f2b100"
FLOOD, ACTION = "#c62828", "#ef6c00"
FULL_W, COL_W = 7.16, 3.5                  # IEEE two-column widths (in)


def _save(fig, name: str) -> None:
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
    """Fig. 2: setup on a dry reference frame (a-c), reading a flood frame (d-e)."""
    site = SiteReader(os.path.join(ROOT, "sites", "hancock"))
    ref = site.reference
    flood = cv.imread(os.path.join(ROOT, "tests", "fixtures", "hancock_flood_17.61ft.jpg"))
    pl, datum = site.plate, site.site["datum_ft"]
    o, d, nrm = (np.array(pl[k]) for k in ("origin", "axis", "normal"))
    h, w = ref.shape[:2]
    cx = int(o[0] + d[0] * (pl["t_top"] + pl["t_bottom"]) / 2)
    x0, x1 = max(cx - 215, 0), min(cx + 215, w)
    y0, y1 = int(max(o[1] + pl["t_top"] - 120, 0)), int(min(o[1] + pl["t_bottom"] + 220, h))
    rgb = lambda im: cv.cvtColor(im, cv.COLOR_BGR2RGB)[y0:y1, x0:x1]   # noqa: E731

    fig = plt.figure(figsize=(FULL_W, 3.05))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.45, 1, 1.45], wspace=0.5)

    # (a) red marks + robust axis on the reference frame
    a = fig.add_subplot(gs[0])
    red = P.red_mask(ref)
    show = rgb(ref).copy()
    show[red[y0:y1, x0:x1] > 0] = (255, 40, 40)
    a.imshow((0.55 * show + 0.45 * 255 * (red[y0:y1, x0:x1, None] > 0)).astype(np.uint8))
    ts = np.array([pl["t_top"] - 60, pl["t_bottom"] + 120])
    a.plot(o[0] + d[0] * ts - x0, o[1] + d[1] * ts - y0, color="cyan", lw=1.0)
    for t, u in zip(pl["bars_t"], pl["bars_ft"]):
        p = o + d * t
        a.plot(p[0] - x0, p[1] - y0, "o", ms=2.6, mfc="none", mec="cyan", mew=0.8)
    _label(a, "a")

    # (b) OCR on the upright plate
    b = fig.add_subplot(gs[1])
    datum_read = P.read_datum(ref, pl, os.path.join(ROOT, "models", "crnn_en.onnx"))
    b.imshow(rgb(ref))
    for r in datum_read["reads"]:
        t = pl["bars_t"][r["bar"]]
        nxt = pl["bars_t"][r["bar"] + 1] if r["bar"] + 1 < len(pl["bars_t"]) else t + 60
        p = o + d * (t + 0.42 * (nxt - t)) + nrm * pl["half_width"] * 2.4
        ok = r["text"].isdigit() and len(r["text"]) == 2
        b.text(p[0] - x0, p[1] - y0, f"“{r['text']}”", fontsize=6.5, va="center", ha="left",
               color="white", bbox={"boxstyle": "round,pad=0.15", "fc": CAM if ok else BASE, "ec": "none"})
    _label(b, "b")

    # (c) 1-D projective map from the foot marks
    c = fig.add_subplot(gs[2])
    bt, bu = np.array(pl["bars_t"]), np.array(pl["bars_ft"])
    tt = np.linspace(bt.min() - 40, bt.max() + 220, 200)
    c.plot(tt, datum - P.to_units(pl, tt), color=CAM, lw=1.0, label=r"$u=\frac{at+b}{ct+1}$")
    c.plot(bt, datum - bu, "o", ms=3, mfc="white", mec=GAUGE, mew=0.7, label="foot marks")
    c.set_xlabel("Position along plate axis $t$ (px)")
    c.set_ylabel("Stage (ft)")
    c.legend(loc="upper right", frameon=False)
    c.text(0.04, 0.05, f"fit error {pl['fit_rmse_ft']:.2f} ft", transform=c.transAxes, fontsize=7)
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
        im[ys[k], xs[k]] = (44, 160, 44) if seen else (194, 24, 91)
    dd.imshow(im)
    cpt = o + [dx, dy] + d * raw["t"]
    hw = pl["half_width"] * 2.3
    dd.plot([cpt[0] - nrm[0] * hw - x0, cpt[0] + nrm[0] * hw - x0],
            [cpt[1] - nrm[1] * hw - y0, cpt[1] + nrm[1] * hw - y0], color=WL, lw=1.6)
    _label(dd, "d")

    # (e) per-mark Lab a* contrast, on the stage axis
    e = fig.add_subplot(gs[4])
    stage_mark = [datum - float(P.to_units(pl, (bar["t_min"] + bar["t_max"]) / 2)) for bar in site.bars]
    con = np.array(raw["contrast"])
    cols = [VIS if v else HID for v in raw["visible"]]
    e.scatter(con, stage_mark, s=9, c=cols, lw=0, zorder=3)
    e.axvline(8, color=GAUGE, lw=0.6, ls="--")
    e.axhline(r["stage_ft"], color=WL, lw=1.4)
    e.axhline(17.61, color=GAUGE, lw=0.7, ls=":")
    e.text(10, min(stage_mark), "threshold", fontsize=6.5, va="bottom")
    e.set_xlabel("Mark contrast, Lab $a^*$")
    e.set_ylabel("Stage (ft)")
    e.text(0.97, 0.55, "visible", color=VIS, transform=e.transAxes, ha="right", fontsize=7)
    e.text(0.12, 0.30, "hidden", color=HID, transform=e.transAxes, ha="left", fontsize=7)
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
    for lvl, col, txt in ((18, FLOOD, "NWS flood stage"), (14, ACTION, "NWS action stage")):
        a.axhline(lvl, color=col, lw=0.6, ls="--")
        a.text(gt[-1], lvl + 0.15, txt, color=col, fontsize=6.5, ha="right", va="bottom")
    a.axhspan(4, 11, color="#000000", alpha=0.04, lw=0)
    a.text(gt[2], 10.6, "below the plate: no reading", fontsize=6.5, color="#555555", va="top")
    a.plot(gt, gv, color=GAUGE, lw=0.9, label="USGS gauge (pressure sensor)")
    a.plot(t, cam, "o", ms=2.6, color=CAM, label="StageSight (no gauge data)")
    a.plot(t, lin, "x", ms=2.6, mew=0.7, color=BASE, label="Linear pixel fit (normal flows)")
    a.set_ylabel("Stage (ft)")
    a.set_ylim(6.5, 19.2)
    a.xaxis.set_major_locator(mdates.DayLocator())
    a.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    a.legend(loc="lower right", frameon=False)
    _label(a, "a")

    err = cam - obs
    b.axhspan(-0.5, 0.5, color=CAM, alpha=0.08, lw=0)
    b.axhline(0, color=GAUGE, lw=0.5)
    b.plot(obs, err, "o", ms=2.6, color=CAM)
    b.set_xlabel("Gauge stage (ft)")
    b.set_ylabel("Camera $-$ gauge (ft)")
    b.set_ylim(-1, 1)
    rmse, within = float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err) <= 0.5))
    b.text(0.04, 0.05, f"RMSE {rmse:.2f} ft, n = {len(err)}", transform=b.transAxes, fontsize=7)
    _label(b, "b")
    _save(fig, "flood")
    return {"n": len(err), "rmse": rmse, "within": within}


def agent_figure() -> None:
    """Fig. 5: hours to catch each gauge fault, rules vs Claude."""
    res = {p: json.load(open(os.path.join(ROOT, f"results/agent/{p}/all_scenarios.json"))) for p in ("rules", "claude")}
    gauge = [("offset", "Offset +1 ft"), ("dropout", "Silent 8 h$^*$"), ("flatline", "Frozen 10 h"), ("drift", "Drift to +1.5 ft")]
    fig, a = plt.subplots(figsize=(COL_W, 1.6))
    y = np.arange(len(gauge))
    for off, p, col, lab in ((0.19, "rules", RULES, "Rule baseline"), (-0.19, "claude", CAM, "Claude Opus 4.6")):
        dl = [res[p][s]["delay_h"] for s, _ in gauge]
        a.barh(y + off, dl, height=0.36, color=col, label=lab)
        for yi, v in zip(y, dl):
            a.text(v + 0.12, yi + off, f"{v:.0f} h", va="center", fontsize=6.5)
    a.set_yticks(y, [g[1] for g in gauge])
    a.tick_params(axis="y", which="both", length=0)
    a.invert_yaxis()
    a.set_xlabel("Hours from fault start to sensor flag")
    a.set_xlim(0, 9)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    a.text(0.0, -0.42, "$^*$time until the camera stage was published in place of the silent gauge",
           transform=a.transAxes, fontsize=6)
    _save(fig, "agent")


def timeline_figure() -> None:
    """Fig. 6: the frozen-gauge replay: what each instrument said and when each planner acted."""
    def load(p):
        return [json.loads(l) for l in open(os.path.join(ROOT, f"results/agent/{p}/flatline/trace.jsonl"))]
    tr = {p: load(p) for p in ("claude", "rules")}
    lo, hi = "2026-07-29T06", "2026-07-30T02"
    rows = [r for r in tr["claude"] if lo <= r["time_utc"][:13] <= hi]
    gauge = [r for r in _rows("results/hancock_plate/gauge_jul28-31.csv") if lo <= r["time_utc"][:13] <= hi]
    fig, a = plt.subplots(figsize=(FULL_W, 1.8))
    a.axvspan(_t("2026-07-29T13:00:00"), _t("2026-07-29T23:00:00"), color="#000000", alpha=0.05, lw=0)
    a.text(_t("2026-07-29T13:10:00"), 18.95, "injected fault: gauge frozen", fontsize=6.5, va="top", color="#555555")
    a.axhline(18, color=FLOOD, lw=0.6, ls="--")
    a.plot([_t(r["time_utc"]) for r in gauge], [float(r["stage_ft"]) for r in gauge], color=GAUGE, lw=0.7,
           ls=":", label="true stage (USGS, held out)")
    a.plot([_t(r["time_utc"]) for r in rows if r["gauge_ft"] is not None],
           [r["gauge_ft"] for r in rows if r["gauge_ft"] is not None], color=GAUGE, lw=1.0,
           drawstyle="steps-post", label="gauge as reported (frozen)")
    cam = [r for r in rows if r["camera"]["stage_ft"] is not None]
    a.plot([_t(r["time_utc"]) for r in cam], [r["camera"]["stage_ft"] for r in cam], "o", ms=2.6, color=CAM,
           label="camera (StageSight)")
    fault0 = _t("2026-07-29T13:00:00")
    for p, col, name, ylab in (("claude", CAM, "Claude", 12.6), ("rules", RULES, "Rules", 11.4)):
        t = next(_t(r["time_utc"]) for r in tr[p] if any(x["type"] == "flag_sensor" for x in r["actions"]))
        a.axvline(t, color=col, lw=0.9, ymax=0.62, ymin=0.0)
        a.text(t, ylab, f" {name} flags gauge (+{(t - fault0).total_seconds() / 3600:.0f} h)", color=col,
               fontsize=6.5, ha="left", va="bottom")
    a.text(_t("2026-07-29T15:10:00"), 14.25, "Claude: “gauge frozen at 17.21 ft while upstream gauge\n"
           "and clean camera frames show a recession”", fontsize=6.2, style="italic", color=CAM, va="center")
    a.set_ylim(11.2, 19.1)
    a.set_ylabel("Stage (ft)")
    a.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    a.set_xlabel("Time (UTC)")
    a.legend(loc="lower left", frameon=False)
    _save(fig, "timeline")


def faults_figure() -> None:
    """Fig. 7: the two simulated camera faults on real frames, with the reader's output.
    Needs the downloaded archive (data/hancock); skipped when it is absent."""
    from stagesight.world import corrupt_frame, plate_band
    idx = os.path.join(ROOT, "data", "hancock", "images.csv")
    if not os.path.exists(idx):
        print("faults figure skipped: data/hancock not downloaded")
        return
    site = SiteReader(os.path.join(ROOT, "sites", "hancock"))
    paths = {r["time_utc"][:13]: os.path.join(ROOT, r["path"].replace("\\", "/")) for r in _rows("data/hancock/images.csv")}
    gauge = {r["time_utc"][:13]: float(r["stage_obs"]) for r in _rows("results/hancock_plate/series.csv")}
    cases = [("2026-07-29T18", None, None, "clean frame"),            # same specs as scripts/replay_agent.py
             ("2026-07-29T18", "occlusion", (16.6, 18.8), "debris over the plate"),
             ("2026-07-30T10", "spray", (12.4, 14.8), "spray on the lens")]
    pl, datum = site.plate, site.site["datum_ft"]
    o, d = np.array(pl["origin"]), np.array(pl["axis"])
    cx = int(o[0] + d[0] * (pl["t_top"] + pl["t_bottom"]) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.05), gridspec_kw={"wspace": 0.06})
    for ax, (hour, kind, band, title), lab in zip(axes, cases, "abc"):
        img = cv.imread(paths[hour])
        if kind:
            img = corrupt_frame(img, {"kind": kind, "seed": 1, "polygon": plate_band(pl, *band, datum).tolist()})
        r = site.read(img)
        vis = site.draw(img, r)[:, :, ::-1]
        y0 = int(max(o[1] + pl["t_top"] - 60, 0))
        ax.imshow(vis[y0:y0 + 1150, max(cx - 260, 0):cx + 260])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"({lab}) {title}", fontsize=7, pad=2)
        ax.set_xlabel(f"camera {r['stage_ft']:.2f} ft\ngauge {gauge[hour]:.2f} ft", fontsize=6.5, labelpad=2)
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
    cmap = ListedColormap(["#ffffff", "#d9dee5", "#5c6370", CAM])
    fig, a = plt.subplots(figsize=(FULL_W, 1.3))
    a.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ticks = [i for i, dname in enumerate(days) if dname.endswith("-01") or dname.endswith("-15")]
    a.set_xticks(ticks, [datetime.fromisoformat(days[i]).strftime("%b %d") for i in ticks])
    a.set_yticks([0, 6, 12, 18, 23], ["00", "06", "12", "18", "23"])
    a.tick_params(which="both", top=False, right=False)
    a.minorticks_off()
    a.set_ylabel("Hour (UTC)")
    n = {k: sum(r["camera"]["status"] == k for r in rows) for k in code}
    a.legend(handles=[Patch(color=CAM, label=f"stage read ({n['ok']})"),
                      Patch(color="#d9dee5", label=f"water below plate ({n['below_plate']})"),
                      Patch(color="#5c6370", label=f"night, fog or spray ({n['unreadable']})"),
                      Patch(fc="white", ec="#999999", lw=0.5, label="no frame")],
             loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    _save(fig, "coverage")
    return {"frames": len(rows), **n}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = flood_figure()
    assert m["n"] == 30 and abs(m["rmse"] - 0.257) < 0.01, m    # must match results/hancock_plate/metrics.json
    p = pipeline_figure()
    assert p["datum"] == 20.0 and abs(p["flood_read"] - 17.77) < 0.01, p
    agent_figure()
    timeline_figure()
    faults_figure()
    print(m, p, coverage_figure())
