"""Turn the COOL runs in results/cool/ into the report's Table II macros (docs/paper/cool_stats.tex).

Usage:  python scripts/cool_stats.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COOL = os.path.join(ROOT, "results", "cool")


def load(instance: str, name: str) -> dict:
    return json.load(open(os.path.join(COOL, instance, "out", f"{name}.json")))


def main() -> None:
    c1, p1 = load("c8g.xlarge", "cool"), load("c8g.xlarge", "pip")
    c4, p4 = load("c8g.4xlarge", "cool"), load("c8g.4xlarge", "pip")
    sc, sp = load("c8g.4xlarge", "scale_cool"), load("c8g.4xlarge", "scale_pip")
    prices = json.load(open(os.path.join(COOL, "prices.json")))
    for a, b in ((c1, p1), (c4, p4), (sc, sp)):
        assert a["stages"] == b["stages"], "the two builds read different stages"
        assert "/opt/cool/" in a["cv2_file"] and "/opt/cool/" not in b["cv2_file"], "wrong OpenCV build loaded"
    best = max(sc["runs"], key=lambda r: r["frames_per_s"])
    pbest = next(r for r in sp["runs"] if r["workers"] == best["workers"])
    usd_h = prices["ec2_usd_per_hour"]["c8g.4xlarge"] + prices["cool_software_usd_per_hour"]["c8g.4xlarge"]
    per_1000 = usd_h * (1000 / best["frames_per_s"]) / 3600
    x = lambda a, b: f"{a / b:.2f}" + r"$\times$"    # noqa: E731
    m = {"COOLfps": f"{c1['frames_per_s']:.1f}", "PIPfps": f"{p1['frames_per_s']:.1f}",
         "COOLspeed": x(c1["frames_per_s"], p1["frames_per_s"]),
         "COOLfpsXL": f"{c4['frames_per_s']:.1f}", "PIPfpsXL": f"{p4['frames_per_s']:.1f}",
         "COOLspeedXL": x(c4["frames_per_s"], p4["frames_per_s"]),
         "COOLbatch": f"{best['frames_per_s']:.0f}", "PIPbatch": f"{pbest['frames_per_s']:.0f}",
         "COOLbatchx": x(best["frames_per_s"], pbest["frames_per_s"]), "BatchWorkers": str(best["workers"]),
         "BatchFrames": f"{sc['frames']:,}", "BatchSeconds": f"{best['seconds']:.1f}",
         "BatchOne": f"{sc['runs'][0]['frames_per_s']:.0f}",
         "CostPerK": f"{per_1000:.4f}", "CostArchive": f"{per_1000 * sc['frames'] / 1000:.4f}",
         "PIPocr": f"{p1['ocr_datum_s']:.2f}", "COOLocrXL": f"{c4['ocr_datum_s']:.2f}", "PIPocrXL": f"{p4['ocr_datum_s']:.2f}"}
    for key, op in (("cc", "connectedComponentsWithStats"), ("pc", "phaseCorrelate_plate"),
                    ("mo", "morphologyEx_open"), ("id", "imdecode_1080x1920")):
        m[f"COOL{key}"], m[f"PIP{key}"] = f"{c1['op_ms'][op]:.2f}", f"{p1['op_ms'][op]:.2f}"
        m[f"COOL{key}x"] = x(p1["op_ms"][op], c1["op_ms"][op])
    tex = "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in m.items())
    open(os.path.join(ROOT, "docs", "paper", "cool_stats.tex"), "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
