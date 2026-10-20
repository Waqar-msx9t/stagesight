"""Self-calibrate a camera on normal flows, then test it on a flood it never saw.

1. Build (or load) the aligned feature stack for every frame.
2. Stage-sensitivity map from training frames only -> automatic transects.
3. Measure the waterline on every frame with those transects.
4. Calibrate on training frames (polynomial and camera-geometry forms).
5. Baseline: ridge regression from the most sensitive pixels straight to stage (the learned,
   CNN-style approach), trained on the same frames.
6. Report errors on the test period, split into in-range and out-of-range (above training max).

Usage:
  python scripts/flood_stress_test.py --test-start 2026-10-01 --out results/stress
"""
import argparse
import csv
import json
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import calibrate, selfcal, usgs, vision  # noqa: E402


def load_inputs(a):
    if os.path.exists(a.cache):
        z = np.load(a.cache)
        return z["stack"], z["ok"], z["stage"], z["times"].astype("datetime64[s]"), list(z["paths"])
    g = list(csv.DictReader(open(a.gauge)))
    gt = np.array([np.datetime64(r["time_utc"], "s") for r in g])
    gh = np.array([float(r["stage_ft"]) for r in g])
    rows = list(csv.DictReader(open(a.index)))
    paths = [r["path"] for r in rows]
    times = np.array([np.datetime64(r["time_utc"], "s") for r in rows])
    stage = np.array([usgs.stage_at(gt, gh, t) or np.nan for t in times], float)
    ref = cv.imread(a.reference)
    h, w = ref.shape[:2]
    size = (320, round(320 * h / w)) if w >= h else (round(320 * w / h), 320)
    stack, ok = selfcal.build_stack(paths, vision.Aligner(ref), size=size)
    np.savez(a.cache, stack=stack, ok=ok, stage=stage, times=times.astype(str), paths=np.array(paths))
    return stack, ok, stage, times, paths


def ridge_baseline(stack, smap, train, k=300, lam=1e2):
    """Learned baseline: ridge regression on the k most sensitive pixels of every feature."""
    flat = stack.reshape(len(stack), stack.shape[1], -1)
    idx = np.argsort(-smap.ravel())[:k]
    X = flat[:, :, idx].reshape(len(stack), -1).astype(np.float32)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    X = np.c_[(X - mu) / sd, np.ones(len(X))]
    return X, lambda y: np.linalg.solve(X[train].T @ X[train] + lam * np.eye(X.shape[1]),
                                        X[train].T @ y[train])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/images.csv")
    ap.add_argument("--gauge", default="data/gauge.csv")
    ap.add_argument("--reference", default="data/reference.jpg")
    ap.add_argument("--cache", default="results/stack.npz")
    ap.add_argument("--test-start", default="2026-10-01", help="UTC start of the test window")
    ap.add_argument("--test-end", default="2100-01-01", help="UTC end of the test window")
    ap.add_argument("--transects", default="gradient", choices=["gradient", "band"])
    ap.add_argument("--threshold", type=float, default=0.75, help="fraction of the map maximum")
    ap.add_argument("--method", default="both", choices=["both", "water", "edge"])
    ap.add_argument("--min-confidence", type=float, default=1.0)
    ap.add_argument("--n-per-band", type=int, default=8)
    ap.add_argument("--no-align", dest="align", action="store_false", help="fixed camera")
    ap.add_argument("--out", default="results/stress")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    stack, ok, stage, times, paths = load_inputs(a)
    has = ok & ~np.isnan(stage)
    in_window = (times >= np.datetime64(a.test_start)) & (times < np.datetime64(a.test_end))
    train, test = has & ~in_window, has & in_window
    print(f"train {train.sum()} frames ({np.nanmin(stage[train]):.2f}-{np.nanmax(stage[train]):.2f} ft), "
          f"test {test.sum()} ({np.nanmin(stage[test]):.2f}-{np.nanmax(stage[test]):.2f} ft)")

    smap, _ = selfcal.sensitivity_map(stack[train], stage[train])
    ref = cv.imread(a.reference)
    if a.transects == "gradient":
        fmap, strength = selfcal.flood_stage_map(stack[train], stage[train])
        transects = selfcal.gradient_transects(fmap, strength, ref.shape[:2],
                                               n_transects=a.n_per_band)
        smap = strength
        span = np.ptp(stage[train])
        fvis = np.clip((fmap - np.nanmin(stage[train])) / span, 0, 1) * (strength > 0.3)
        cv.imwrite(os.path.join(a.out, "flood_stage_map.jpg"), selfcal.heatmap(fvis, ref, transects))
    else:
        land = stack[train][:, selfcal.FEATURES.index("chroma_g")].astype(np.float32).mean(axis=0)
        transects = selfcal.auto_transects(smap, ref.shape[:2], threshold=a.threshold,
                                           land_map=cv.GaussianBlur(land, (9, 9), 0),
                                           n_per_band=a.n_per_band)
    print(f"max sensitivity {smap.max():.2f}, {len(transects)} transects")
    cv.imwrite(os.path.join(a.out, "sensitivity_map.jpg"),
               selfcal.heatmap(smap / (smap.max() + 1e-9), ref, transects))
    json.dump({"transects": transects}, open(os.path.join(a.out, "transects.json"), "w"), indent=1)

    aligner = vision.Aligner(ref) if a.align else None
    frac = np.full(len(paths), np.nan)
    status = []
    for i, p in enumerate(paths):
        if not has[i]:
            status.append("skipped")
            continue
        m = vision.measure(cv.imread(p), transects, aligner, method=a.method,
                           min_confidence=a.min_confidence)
        status.append(m["status"])
        if m["status"] == "ok":
            frac[i] = m["fraction"]
    status = np.array(status)
    usable = ~np.isnan(frac)

    tr, te = train & usable, test & usable
    hi = stage[tr].max()
    sets = {"test_all": te, "test_in_range": te & (stage <= hi), "test_above_train_max": te & (stage > hi)}
    preds = {}
    for name, kw in {"poly2": {"kind": "poly", "degree": 2}, "linear": {"kind": "poly", "degree": 1},
                     "projective": {"kind": "projective"}}.items():
        model = calibrate.fit(frac[tr], stage[tr], **kw)
        preds[name] = np.where(usable, calibrate.predict(model, np.nan_to_num(frac)), np.nan)
        calibrate.save(model, os.path.join(a.out, f"calibration_{name}.json"))
    X, solve = ridge_baseline(stack, smap, train)
    preds["ridge_pixels_baseline"] = X @ solve(np.nan_to_num(stage))

    report = {"test_window": [a.test_start, a.test_end], "train_frames": int(tr.sum()), "train_max_ft": float(hi),
              "frames_by_status": {s: int((status == s).sum()) for s in np.unique(status)},
              "methods": {}}
    for name, p in preds.items():
        report["methods"][name] = {k: calibrate.metrics(stage[m], p[m]) if m.sum() else None
                                   for k, m in sets.items()}
        # the baseline needs no waterline, so also score it on every test frame
        if name == "ridge_pixels_baseline":
            m = test
            report["methods"][name]["test_all_frames"] = calibrate.metrics(stage[m], p[m])
    json.dump(report, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(a.out, "series.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "stage_obs", "status", "fraction", "split"] + list(preds))
        for i in range(len(paths)):
            w.writerow([str(times[i]), stage[i], status[i], frac[i],
                        "train" if train[i] else "test" if test[i] else ""] + [preds[k][i] for k in preds])
    print(json.dumps({k: {s: (v[s]["rmse"] if v[s] else None) for s in v}
                      for k, v in report["methods"].items()}, indent=1))
    print("status:", report["frames_by_status"])


if __name__ == "__main__":
    main()
