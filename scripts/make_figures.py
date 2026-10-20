"""Report figures from the saved results (no recomputation).

Usage:  python scripts/make_figures.py      ->  docs/fig_*.png
Needs matplotlib and SciencePlots (dev only, not in the Lambda package).
"""
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" styles)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs")
plt.style.use(["science", "no-latex", "grid"])
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "font.size": 9, "grid.alpha": 0.35})
BLUE, ORANGE, GREY, RED = "#0C5DA5", "#FF6F00", "#6b6b6b", "#c0392b"   # science-style colours


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s[:19])


def _rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def flood_figure() -> dict:
    series = [r for r in _rows(os.path.join(ROOT, "results/hancock_plate/series.csv")) if r["split"] == "test"]
    t = [_t(r["time_utc"]) for r in series]
    obs = np.array([float(r["stage_obs"]) for r in series])
    cam = np.array([float(r["pred_ocr"]) for r in series])
    lin = np.array([float(r["pred_linear"]) for r in series])
    gauge = [r for r in _rows(os.path.join(ROOT, "data/hancock/gauge.csv"))
             if "2026-07-28" <= r["time_utc"][:10] <= "2026-07-31"]
    gt = [_t(r["time_utc"]) for r in gauge]
    gv = [float(r["stage_ft"]) for r in gauge]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.3, 1]})
    a.axhline(18, color=RED, lw=0.8, ls="--")
    a.axhline(14, color=ORANGE, lw=0.8, ls="--")
    a.axhline(11, color=GREY, lw=0.6, ls=":")
    a.text(gt[0], 18.15, "NWS flood 18 ft", color=RED, fontsize=7)
    a.text(gt[0], 14.15, "NWS action 14 ft", color=ORANGE, fontsize=7)
    a.text(gt[0], 11.15, "plate bottom", color=GREY, fontsize=7)
    a.plot(gt, gv, color="k", lw=1.0, label="USGS gauge (pressure sensor)")
    a.plot(t, cam, "o", ms=3.2, color=BLUE, label="StageSight, plate + OCR datum (no gauge)")
    a.plot(t, lin, "x", ms=3.2, color=GREY, label="Linear pixel fit, trained on normal flows")
    a.set_ylabel("Stage (ft)")
    a.set_ylim(4, 19.5)
    a.xaxis.set_major_locator(mdates.DayLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    a.legend(loc="lower right", fontsize=6.5, frameon=True)
    a.set_title("(a) Jul 29 2026 flood, Hancock NY (USGS 01421500)", fontsize=8.5)

    err = cam - obs
    b.axhspan(-0.5, 0.5, color=BLUE, alpha=0.08, lw=0)
    b.axhline(0, color="k", lw=0.6)
    b.plot(obs, err, "o", ms=3.2, color=BLUE)
    b.set_xlabel("Gauge stage (ft)")
    b.set_ylabel("Camera - gauge (ft)")
    b.set_ylim(-1, 1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    within = float(np.mean(np.abs(err) <= 0.5))
    b.set_title(f"(b) n={len(err)}, RMSE {rmse:.2f} ft, {within:.0%} within 0.5 ft", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_flood.png"))
    plt.close(fig)
    return {"n": len(err), "rmse": rmse, "within": within}


def agent_figure() -> None:
    res = {p: json.load(open(os.path.join(ROOT, f"results/agent/{p}/all_scenarios.json"))) for p in ("rules", "claude")}
    gauge = [("offset", "Offset +1 ft"), ("dropout", "Silent 8 h"), ("flatline", "Frozen 10 h"), ("drift", "Drift to +1.5 ft")]
    cams = [("occlusion", "Debris on plate"), ("spray", "Spray on lens")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(gauge))
    for k, (p, c, lab) in enumerate((("rules", GREY, "Rule baseline"), ("claude", BLUE, "Claude Opus 4.6"))):
        d = [res[p][s]["delay_h"] for s, _ in gauge]
        a.barh(y + (0.2 if k == 0 else -0.2), d, height=0.38, color=c, label=lab)
        for yi, v in zip(y, d):
            a.text(v + 0.1, yi + (0.2 if k == 0 else -0.2), f"{v:.0f} h", va="center", fontsize=7)
    a.set_yticks(y, [g[1] for g in gauge])
    a.invert_yaxis()
    a.set_xlabel("Time to catch the gauge fault (h)")
    a.set_xlim(0, 9.5)
    a.grid(axis="y", visible=False)
    a.legend(fontsize=7, loc="upper right", frameon=True)
    a.set_title("(a) Gauge faults: both catch all four", fontsize=8.5)

    b.axis("off")
    b.set_title("(b) Camera faults, gauge healthy", fontsize=8.5)
    cell = [["", "Rules", "Claude"]]
    for s, lab in cams:
        row = [lab]
        for p in ("rules", "claude"):
            r = res[p][s]
            row.append("blamed camera" if r.get("correct") else "blamed healthy\ngauge")
        cell.append(row)
    tab = b.table(cellText=cell, loc="center", cellLoc="center", colWidths=[0.38, 0.31, 0.31])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 2.4)
    for (i, j), c in tab.get_celld().items():
        c.set_linewidth(0.4)
        if i > 0 and j > 0:
            ok = "camera" in c.get_text().get_text()
            c.set_facecolor("#e3f0fb" if ok else "#fbe3e1")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_agent.png"))
    plt.close(fig)


def coverage_figure() -> dict:
    st = json.load(open(os.path.join(ROOT, "results/hancock_plate/metrics.json")))["status"]
    total = sum(st.values())
    parts = [("ok", "stage read", BLUE), ("below_plate", "water below plate (< 11 ft)", "#9fb7cf"),
             ("unreadable", "night / fog / spray", GREY), ("skipped", "skipped", "#cccccc")]
    fig, a = plt.subplots(figsize=(7.2, 1.15))
    left = 0
    for k, lab, c in parts:
        v = st.get(k, 0)
        a.barh(0, v, left=left, color=c, height=0.6)
        if v / total > 0.05:
            a.text(left + v / 2, 0, f"{lab}\n{v} ({v / total:.1%})", ha="center", va="center", fontsize=6.5,
                   color="white" if c in (BLUE, GREY) else "k")
        left += v
    a.annotate(f"stage read: {st['ok']} ({st['ok'] / total:.1%})", xy=(st["ok"] / 2, 0.3), xytext=(60, 0.62),
               fontsize=6.5, arrowprops={"arrowstyle": "-", "lw": 0.5})
    a.set_xlim(0, total)
    a.set_ylim(-0.45, 0.8)
    a.set_yticks([])
    a.grid(False)
    a.set_xlabel(f"Hourly frames, Jul 21 to Oct 19 2026 (n = {total})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_coverage.png"))
    plt.close(fig)
    return {"total": total, **st}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = flood_figure()
    assert m["n"] == 30 and abs(m["rmse"] - 0.257) < 0.01, m    # must match results/hancock_plate/metrics.json
    agent_figure()
    print(m, coverage_figure())
