"""Architecture diagram (report Fig. 2), drawn with matplotlib + SciencePlots like the other figures.

Usage:  python scripts/make_architecture.py   ->  docs/figures/architecture.{pdf,png}
Icons: AWS Architecture Icons (docs/icons). The layout is checked by scripts/figcheck.py.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(__file__))
import figcheck  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ICONS = os.path.join(ROOT, "docs", "icons")
OUT = os.path.join(ROOT, "docs", "figures")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"],
                     "mathtext.fontset": "stix", "savefig.dpi": 400, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
INK, SUB, AWS, REGION, PUB, BADGE = "#232F3E", "#4b5563", "#232F3E", "#00A4A6", "#7D8998", "#ED7100"
S = 0.46                                     # icon half-size (data units)


def icon(ax, name: str, x: float, y: float) -> None:
    ax.imshow(plt.imread(os.path.join(ICONS, name)), extent=(x - S, x + S, y - S, y + S), zorder=3)


def ec2_icon(ax, x: float, y: float) -> None:
    """EC2 in the AWS icon style (orange tile, white chip); the icon set in docs/icons has no EC2 file."""
    ax.add_patch(FancyBboxPatch((x - S, y - S), 2 * S, 2 * S, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=BADGE, ec="none", zorder=3))
    c = 0.2
    ax.add_patch(Rectangle((x - c, y - c), 2 * c, 2 * c, fill=False, ec="white", lw=1.1, zorder=4))
    for t in (-0.1, 0.0, 0.1):
        for a, b in (((x + t, y + c), (x + t, y + c + 0.1)), ((x + t, y - c), (x + t, y - c - 0.1)),
                     ((x - c, y + t), (x - c - 0.1, y + t)), ((x + c, y + t), (x + c + 0.1, y + t))):
            ax.plot(*zip(a, b), color="white", lw=0.9, zorder=4, solid_capstyle="butt")


def caption(ax, x: float, y: float, title: str, sub: str, ha: str = "center") -> None:
    ax.text(x, y, title, ha=ha, va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(x, y - 0.38, sub, ha=ha, va="top", fontsize=8, color=SUB, linespacing=1.15)


def group(ax, x0, y0, x1, y1, color, label, dashed=True, where="top") -> None:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=0.9,
                           ls=(0, (4, 3)) if dashed else "-", zorder=1))
    ty = y1 - 0.12 if where == "top" else y0 + 0.12
    ax.text(x0 + 0.12, ty, label, ha="left", va="top" if where == "top" else "bottom", fontsize=8.2,
            fontweight="bold", color=color)


def arrow(ax, pts, both=False, dashed=False) -> None:
    """Polyline arrow through the points; the head (or heads) at the ends."""
    style = dict(color=INK, lw=0.9, ls=(0, (3, 2)) if dashed else "-", zorder=2)
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        last, first = k == len(pts) - 2, k == 0
        head = "<|-|>" if both and len(pts) == 2 else "-|>" if last else "<|-" if both and first else "-"
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=head, mutation_scale=8, shrinkA=0, shrinkB=0, **style))


def badge(ax, x: float, y: float, n: int) -> None:
    ax.text(x, y, str(n), ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4,
            bbox={"boxstyle": "circle,pad=0.22", "fc": BADGE, "ec": "none"})


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.75))          # placed at text width in the report
    ax.set_xlim(0, 16.4)
    ax.set_ylim(-0.2, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    group(ax, 0.05, 0.3, 3.25, 6.55, PUB, "Public data")
    group(ax, 3.5, -0.15, 14.05, 8.95, AWS, "AWS Cloud", dashed=False)
    group(ax, 3.7, 0.0, 13.8, 5.75, REGION, "Region us-east-2")
    group(ax, 8.45, 6.15, 13.8, 8.7, REGION, "Region us-east-1 (global profile)")

    group(ax, 3.7, 6.15, 7.3, 8.35, BADGE, "One-off, per new site")
    ec2_icon(ax, 4.3, 7.15)
    ax.text(4.88, 7.65, "EC2 Graviton4", ha="left", va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(4.88, 7.27, "COOL 3.1 batch:\narchive in 4.1 s", ha="left", va="top", fontsize=8, color=SUB,
            linespacing=1.15)
    arrow(ax, [(2.0, 5.62), (2.0, 7.15), (3.8, 7.15)], dashed=True)          # archive, once
    ax.text(2.1, 7.25, "archive", ha="left", va="bottom", fontsize=8, color=SUB)

    icon(ax, "multimedia.png", 1.65, 5.15)
    caption(ax, 1.65, 4.55, "USGS NIMS camera", "new frame hourly")
    icon(ax, "internet-alt2.png", 1.65, 2.0)
    caption(ax, 1.65, 1.4, "USGS gauges, NWS", "stage, flood levels")

    icon(ax, "eventbridge.png", 5.7, 4.6)
    caption(ax, 5.7, 4.0, "Amazon EventBridge", "rule, every 15 min")
    icon(ax, "lambda.png", 5.7, 2.0)
    caption(ax, 5.7, 1.4, "AWS Lambda: reader", "OpenCV 5 reader, agent loop\nand tools (Graviton arm64)")
    icon(ax, "s3.png", 9.35, 2.0)
    caption(ax, 9.35, 1.4, "Amazon S3", "results, approval queue")
    icon(ax, "lambda.png", 12.25, 2.0)
    caption(ax, 12.25, 1.4, "AWS Lambda: API", "dashboard, replay,\nmeasure, approve")
    icon(ax, "bedrock.png", 9.3, 7.35)
    caption(ax, 10.0, 7.85, "Amazon Bedrock", "Claude Opus 4.6,\ncalled only on events", ha="left")
    icon(ax, "user.png", 15.65, 2.0)
    caption(ax, 15.65, 1.4, "Operator", "approves flags\nand alerts")

    arrow(ax, [(5.7, 3.1), (5.7, 2.5)])                                    # 1 schedule
    badge(ax, 6.1, 2.8, 1)
    arrow(ax, [(2.2, 5.15), (3.9, 5.15), (3.9, 2.3), (5.2, 2.3)])           # 2 frames
    arrow(ax, [(2.2, 1.95), (5.2, 1.95)])                                    # 2 gauges
    badge(ax, 2.85, 2.35, 2)
    arrow(ax, [(6.2, 2.3), (7.65, 2.3), (7.65, 7.5), (8.8, 7.5)])            # 3 event + tool results
    arrow(ax, [(8.8, 7.15), (8.2, 7.15), (8.2, 2.0), (6.2, 2.0)], dashed=True)    # tool calls back
    badge(ax, 8.05, 7.95, 3)
    ax.text(7.42, 4.4, "event, tool results", rotation=90, ha="center", va="center", fontsize=8, color=SUB)
    ax.text(8.42, 4.4, "tool calls, actions", rotation=90, ha="left", va="center", fontsize=8, color=SUB)
    arrow(ax, [(6.2, 1.72), (8.85, 1.72)])                                    # 4 write
    badge(ax, 7.9, 1.36, 4)
    arrow(ax, [(9.85, 2.0), (11.75, 2.0)], both=True)                          # 5 serve / decisions
    badge(ax, 10.8, 2.4, 5)
    arrow(ax, [(12.75, 2.0), (15.15, 2.0)], both=True)                          # 6 HTTPS
    badge(ax, 13.3, 2.4, 6)
    ax.text(14.2, 2.3, "HTTPS", ha="left", va="bottom", fontsize=8, color=SUB)
    icon(ax, "sns.png", 12.25, 4.4)                                            # 7 e-mail after approval
    ax.text(11.6, 4.62, "Amazon SNS", ha="right", va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(11.6, 4.18, "e-mail after approval", ha="right", va="center", fontsize=8, color=SUB)
    arrow(ax, [(12.25, 2.5), (12.25, 3.9)])
    arrow(ax, [(12.25, 4.9), (12.25, 5.35), (15.65, 5.35), (15.65, 2.55)])
    badge(ax, 12.65, 3.2, 7)

    issues = figcheck.overlaps(fig, "architecture")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"architecture.{ext}"))
    for i in issues:
        print(i)
    print(f"layout check: {len(issues)} problems")


if __name__ == "__main__":
    main()
