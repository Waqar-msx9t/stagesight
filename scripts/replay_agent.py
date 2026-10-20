"""Replay the agent through archived data, optionally with an injected gauge fault, and score it.

Usage:
  python scripts/replay_agent.py --scenario drift --planner rules
  python scripts/replay_agent.py --all --planner rules            # every scenario
  python scripts/replay_agent.py --scenario none --start 2026-07-21 --end 2026-10-19   # false alarms

Outputs results/agent/<planner>/<scenario>/trace.jsonl and summary.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight.agent import ClaudePlanner, Monitor, RulePlanner  # noqa: E402
from stagesight.reader import SiteReader  # noqa: E402
from stagesight.world import ReplayWorld, inject_fault, load_series, plate_band  # noqa: E402

# faults during the Jul 29, 2026 flood at Hancock (water on the plate Jul 29 10:00 - Jul 31 00:00 UTC)
SCENARIOS = {
    "none": None,
    "drift": ("drift", "2026-07-29T12:00:00", 12, 1.5),
    "offset": ("offset", "2026-07-30T10:00:00", 12, 1.0),
    "flatline": ("flatline", "2026-07-29T13:00:00", 10, 0.0),
    "dropout": ("dropout", "2026-07-29T16:00:00", 8, 0.0),
    # camera-side faults: the gauge is healthy, the camera is wrong (stage band covered, ft)
    "occlusion": ("camera:occlusion", "2026-07-29T18:00:00", 6, (16.6, 18.8)),
    "spray": ("camera:spray", "2026-07-30T10:00:00", 7, (12.4, 14.8)),
}


def score(records: list[dict], fault: dict | None) -> dict:
    flags = [(a["time_utc"], a) for r in records for a in r["actions"] if a["type"] == "flag_sensor"]
    prov = [a["time_utc"] for r in records for a in r["actions"] if a["type"] == "publish_provisional_stage"]
    alerts = [(a["time_utc"], a["level"]) for r in records for a in r["actions"] if a["type"] == "draft_alert"]
    levels = {lvl for _, lvl in alerts}
    out = {"frames": len(records), "events": sum(len(r["events"]) for r in records),
           "flood_alert": "flood" in levels, "action_alert": "action" in levels,
           "flags": [(t, a["suspect"]) for t, a in flags], "alerts": alerts, "provisional": len(prov)}
    if fault is None:
        out["false_flags"] = len(flags)
        return out
    t0, t1 = np.datetime64(fault["start"]), np.datetime64(fault["end"])
    grace = t1 + np.timedelta64(2, "h")
    if fault["kind"].startswith("camera:"):
        win = [(t, a) for t, a in flags if t0 <= np.datetime64(t) <= grace]
        out["blamed_gauge"] = sum(a["suspect"] == "site_gauge" for _, a in win)
        out["blamed_camera"] = sum(a["suspect"] == "camera" for _, a in win)
        out["published_bad_camera"] = len([t for t in prov if t0 <= np.datetime64(t) <= grace])
        out["false_flags"] = len([t for t, _ in flags if not (t0 <= np.datetime64(t) <= grace)])
        out["correct"] = out["blamed_gauge"] == 0 and out["published_bad_camera"] == 0
        out["fault"] = fault
        return out
    inside = [t for t, a in flags if t0 <= np.datetime64(t) <= grace and a["suspect"] == "site_gauge"]
    out["false_flags"] = len([t for t, _ in flags if not (t0 <= np.datetime64(t) <= grace)])
    if fault["kind"] == "dropout":
        filled = [t for t in prov if t0 <= np.datetime64(t) <= grace]
        out["detected"] = bool(filled)
        out["delay_h"] = (float((np.datetime64(filled[0]) - t0) / np.timedelta64(1, "h")) if filled else None)
    else:
        out["detected"] = bool(inside)
        out["delay_h"] = (float((np.datetime64(inside[0]) - t0) / np.timedelta64(1, "h")) if inside else None)
    out["fault"] = fault
    return out


def run(a, name: str) -> dict:
    gt, gv = load_series(a.gauge)
    fault, cam_fault = None, None
    reader = SiteReader(a.site)
    if SCENARIOS[name]:
        kind, start, hours, size = SCENARIOS[name]
        if kind.startswith("camera:"):
            end = str(np.datetime64(start) + np.timedelta64(hours, "h"))
            cam_fault = {"kind": kind.split(":")[1], "start": start, "end": end, "seed": 1,
                         "polygon": plate_band(reader.plate, *size, reader.site["datum_ft"]).tolist()}
            fault = {"kind": kind, "start": start, "end": end, "band_ft": size}
        else:
            gv, fault = inject_fault(gt, gv, kind, start, hours, size)
    world = ReplayWorld(a.images, (gt, gv), load_series(a.upstream) if a.upstream else None, cam_fault)
    planner = ClaudePlanner() if a.planner == "claude" else RulePlanner()
    mon = Monitor(reader, world, planner)
    t_start, t_end = np.datetime64(a.start), np.datetime64(a.end)
    frames = world.frame_t[(world.frame_t >= t_start) & (world.frame_t < t_end)]
    out_dir = os.path.join(a.out, a.planner, name)
    os.makedirs(out_dir, exist_ok=True)
    records = []
    with open(os.path.join(out_dir, "trace.jsonl"), "w") as f:
        for t in frames:
            world.now = t
            rec = mon.tick()
            records.append(rec)
            f.write(json.dumps(rec, default=str) + "\n")
            for e in rec["events"]:
                acts = ", ".join(x["type"] for x in rec["actions"])
                print(f"  {rec['time_utc']}  {e['event']['kind']:<13} -> {acts}")
    s = score(records, fault)
    json.dump(s, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="sites/hancock")
    ap.add_argument("--images", default="data/hancock/images.csv")
    ap.add_argument("--gauge", default="data/hancock/gauge.csv")
    ap.add_argument("--upstream", default="data/hancock/upstream_01417500.csv")
    ap.add_argument("--start", default="2026-07-28T12:00:00")
    ap.add_argument("--end", default="2026-08-01T00:00:00")
    ap.add_argument("--scenario", default="drift", choices=list(SCENARIOS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--planner", default="rules", choices=["rules", "claude"])
    ap.add_argument("--out", default="results/agent")
    a = ap.parse_args()
    names = list(SCENARIOS) if a.all else [a.scenario]
    table = {}
    for n in names:
        print(f"== {n}")
        table[n] = run(a, n)
        s = table[n]
        if "correct" in s:
            print(f"   flood_alert={s['flood_alert']} correct={s['correct']} blamed_gauge={s['blamed_gauge']} blamed_camera={s['blamed_camera']} "
                  f"published_bad_camera={s['published_bad_camera']} false_flags={s['false_flags']}")
        else:
            print(f"   flood_alert={s['flood_alert']} detected={s.get('detected')} delay_h={s.get('delay_h')} false_flags={s['false_flags']} "
                  f"events={s['events']} alerts={len(s['alerts'])}")
    if a.all:
        json.dump(table, open(os.path.join(a.out, a.planner, "all_scenarios.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
