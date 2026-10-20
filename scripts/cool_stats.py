"""Turn results/cool/out/{cool,pip}.json into the report's Table IV macros (docs/paper/cool_stats.tex).

Usage:  python scripts/cool_stats.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "cool", "out")


def main() -> None:
    c, p = (json.load(open(os.path.join(OUT, f"{n}.json"))) for n in ("cool", "pip"))
    assert c["stages"] == p["stages"], "the two builds read different stages"
    assert "/opt/cool/" in c["cv2_file"] and "/opt/cool/" not in p["cv2_file"], "wrong OpenCV build loaded"
    m = {"COOLfps": f"{c['frames_per_s']:.1f}", "PIPfps": f"{p['frames_per_s']:.1f}",
         "COOLspeed": f"{c['frames_per_s'] / p['frames_per_s']:.2f}$\\times$"}
    for key, op in (("cc", "connectedComponentsWithStats"), ("pc", "phaseCorrelate_plate"),
                    ("mo", "morphologyEx_open"), ("id", "imdecode_1080x1920")):
        m[f"COOL{key}"], m[f"PIP{key}"] = f"{c['op_ms'][op]:.2f}", f"{p['op_ms'][op]:.2f}"
        m[f"COOL{key}x"] = f"{p['op_ms'][op] / c['op_ms'][op]:.2f}$\\times$"
    m["PIPocr"] = f"{p['ocr_datum_s']:.2f}"
    tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())
    open(os.path.join(ROOT, "docs", "paper", "cool_stats.tex"), "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
