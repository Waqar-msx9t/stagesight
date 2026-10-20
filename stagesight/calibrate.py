"""Calibrate waterline position to USGS gage height and evaluate on held-out frames."""
from __future__ import annotations

import json

import numpy as np


def _fit_projective(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1D projective map s = (a*u + b) / (c*u + 1), solved linearly from s = a*u + b - c*u*s.

    A pinhole camera images a straight line on the bank onto a straight line in the frame through
    a 1D homography, so this 3-parameter form follows the camera geometry instead of a free curve.
    """
    A = np.c_[x, np.ones_like(x), -x * y]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _eval(model: dict, x: np.ndarray) -> np.ndarray:
    coef = np.asarray(model["coef"])
    if model.get("kind") == "projective":
        a, b, c = coef
        return (a * x + b) / (c * x + 1)
    return np.polyval(coef, x)


def fit(fractions, stages, degree: int = 2, n_iter: int = 3, clip_sigma: float = 2.5,
        kind: str = "poly") -> dict:
    """Fit stage vs waterline fraction with iterative outlier clipping.

    kind="poly" fits a polynomial of `degree`; kind="projective" fits the camera-geometry form.
    """
    x, y = np.asarray(fractions, float), np.asarray(stages, float)
    keep = np.ones_like(x, bool)
    model = {"kind": kind, "degree": degree}
    for _ in range(n_iter):
        coef = (_fit_projective(x[keep], y[keep]) if kind == "projective"
                else np.polyfit(x[keep], y[keep], degree))
        model["coef"] = coef.tolist()
        resid = y - _eval(model, x)
        s = resid[keep].std() + 1e-9
        keep = np.abs(resid) < clip_sigma * s
    model.update({"n_used": int(keep.sum()),
                  "x_range": [float(x[keep].min()), float(x[keep].max())]})
    return model


def predict(model: dict, fractions):
    return _eval(model, np.asarray(fractions, float))


def metrics(obs, pred, tolerance: float = 0.5) -> dict:
    """Errors in the gauge's own unit (USGS gage height is in feet)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    err = pred - obs
    ss_res = float((err ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum()) + 1e-12
    return {"n": int(obs.size), "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "r2": 1 - ss_res / ss_tot,
            f"within_{tolerance}": float((np.abs(err) <= tolerance).mean())}


def save(model: dict, path: str):
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
