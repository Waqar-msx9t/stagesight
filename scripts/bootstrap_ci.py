"""95% bootstrap interval of StageSight's test RMSE (report Table III), added to metrics.json.

Usage:  python scripts/bootstrap_ci.py [--dir results/hancock_plate]
Reads series.csv written by plate_test.py; resamples the test frames with replacement.
"""
import argparse
import csv
import json
import os

import numpy as np


def bootstrap_rmse(err: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of the RMSE of err over n resamples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n, len(err)))
    r = np.sqrt((err[idx] ** 2).mean(axis=1))
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/hancock_plate")
    a = ap.parse_args()
    with open(os.path.join(a.dir, "series.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["pred_ocr"] != "nan"]
    err = np.array([float(r["pred_ocr"]) - float(r["stage_obs"]) for r in rows])
    lo, hi = bootstrap_rmse(err)
    assert lo <= float(np.sqrt((err ** 2).mean())) <= hi
    p = os.path.join(a.dir, "metrics.json")
    m = json.load(open(p))
    m["plate_ocr_no_gauge_rmse_ci95"] = {"low": round(lo, 3), "high": round(hi, 3), "n": len(err), "resamples": 10000, "seed": 0}
    json.dump(m, open(p, "w"), indent=2)
    print(f"RMSE 95% bootstrap interval over {len(err)} test frames: {lo:.3f} to {hi:.3f} ft")


if __name__ == "__main__":
    main()
