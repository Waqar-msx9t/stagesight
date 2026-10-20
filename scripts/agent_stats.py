"""Summarise the repeated agent replays into the report's Table III.

Reads results/agent/rules/ and results/agent/claude_<model>_r<k>/ (scripts/repeat_agent_runs.sh)
and writes docs/paper/agent_table.tex and results/agent/repeat_summary.json.

Usage:  python scripts/agent_stats.py
"""
import glob
import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENT = os.path.join(ROOT, "results", "agent")
GAUGE = [("offset", "gauge offset +1 ft"), ("dropout", "gauge silent 8 h"),
         ("flatline", "gauge frozen 10 h"), ("drift", "gauge drift to +1.5 ft")]
CAMERA = [("occlusion", "debris over plate"), ("spray", "spray on lens")]
MODELS = [("opus", "Opus 4.6"), ("sonnet", "Sonnet 4.6")]


def camera_ok(s: dict) -> bool:
    """The camera is blamed, the healthy gauge is not, the bad camera stage is never published."""
    return s.get("blamed_camera", 0) > 0 and s.get("blamed_gauge", 0) == 0 and s.get("published_bad_camera", 0) == 0


def load(pattern: str) -> list[dict]:
    runs = []
    for d in sorted(glob.glob(os.path.join(AGENT, pattern))):
        f = os.path.join(d, "all_scenarios.json")
        if os.path.exists(f):
            runs.append(json.load(open(f)))
    return runs


def cell_gauge(runs: list[dict], name: str) -> tuple[str, dict]:
    hits = [r[name] for r in runs if r[name].get("detected")]
    delays = [h["delay_h"] for h in hits]
    med = statistics.median(delays) if delays else None
    txt = f"{len(hits)}/{len(runs)}" + (f", {med:.0f} h" if med is not None else "")
    if delays and max(delays) - min(delays) >= 1:
        txt += f" ({min(delays):.0f}--{max(delays):.0f})"
    return txt, {"caught": len(hits), "runs": len(runs), "delays_h": delays}


def main() -> None:
    rules = json.load(open(os.path.join(AGENT, "rules", "all_scenarios.json")))
    models = {m: load(f"claude_{m}_r*") for m, _ in MODELS}
    n = min(len(v) for v in models.values())
    models = {m: v[:n] for m, v in models.items()}          # same number of runs per model
    summary = {"runs_per_model": n, "rows": {}}
    rows = []
    for name, label in GAUGE:
        r = rules[name]
        cells = [f"1/1, {r['delay_h']:.0f} h" if r.get("detected") else "0/1"]
        for m, _ in MODELS:
            txt, stats = cell_gauge(models[m], name)
            cells.append(txt)
            summary["rows"].setdefault(name, {})[m] = stats
        rows.append((label, cells))
    for name, label in CAMERA:
        cells = ["0/1" if not camera_ok(rules[name]) else "1/1"]
        for m, _ in MODELS:
            ok = sum(camera_ok(r[name]) for r in models[m])
            cells.append(f"{ok}/{n}")
            summary["rows"].setdefault(name, {})[m] = {"handled": ok, "runs": n}
        rows.append((label, cells))
    false_flags = [sum(r[s]["false_flags"] for s in r) for r in [rules]]
    ff = {m: sum(sum(r[s]["false_flags"] for s in r) for r in models[m]) for m, _ in MODELS}
    alerts = {m: sum(all(r[s]["flood_alert"] for s in r) for r in models[m]) for m, _ in MODELS}
    fallbacks = {m: sum(sum(r[s].get("fallbacks", 0) for s in r) for r in models[m]) for m, _ in MODELS}
    tokens = {m: [sum(r[s].get("tokens_in", 0) for s in r) for r in models[m]] for m, _ in MODELS}
    tokens_out = {m: [sum(r[s].get("tokens_out", 0) for s in r) for r in models[m]] for m, _ in MODELS}
    rows.append(("flood alert in every scenario", ["1/1" if all(rules[s]["flood_alert"] for s in rules) else "0/1"]
                 + [f"{alerts[m]}/{n}" for m, _ in MODELS]))
    rows.append(("false sensor flags (all runs)", [str(false_flags[0])] + [str(ff[m]) for m, _ in MODELS]))
    summary.update(false_flags=ff, flood_alert_runs=alerts, fallbacks=fallbacks,
                   tokens_in_per_run=tokens, tokens_out_per_run=tokens_out)

    head = " & ".join(["Fault", "Rules"] + [lab for _, lab in MODELS])
    body = "\n".join(f"{lab} & " + " & ".join(c) + r"\\" + ("\n\\midrule" if lab == CAMERA[-1][1] else "")
                     for lab, c in rows)
    tex = rf"""\begin{{table}}[t]
\caption{{Injected faults on the 29 July flood: runs that caught the fault and median delay (range)}}
\label{{tab:agent}}
\centering\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
{head}\\
\midrule
{body}
\bottomrule
\multicolumn{{4}}{{@{{}}p{{\columnwidth}}@{{}}}}{{\scriptsize Rules are deterministic (one run); each Claude model ran
every scenario {n} times. Delay: hours from the fault start to the gauge flag (silent gauge: to the
published camera stage). Camera faults: the camera blamed, the healthy gauge not blamed, the bad
camera stage never published.}}
\end{{tabular}}
\end{{table}}
"""
    open(os.path.join(ROOT, "docs", "paper", "agent_table.tex"), "w").write(tex)
    tin, tout = statistics.median(tokens["opus"] or [0]), statistics.median(tokens_out["opus"] or [0])
    macros = (rf"\newcommand{{\AgentRuns}}{{{n}}}" "\n"
              rf"\newcommand{{\AgentTokens}}{{about {tin / 7 / 1000:.0f}k input and {tout / 7 / 1000:.0f}k output tokens per scenario}}" "\n")
    open(os.path.join(ROOT, "docs", "paper", "agent_stats.tex"), "w").write(macros + open(
        os.path.join(ROOT, "docs", "paper", "agent_prose.tex")).read())
    json.dump(summary, open(os.path.join(AGENT, "repeat_summary.json"), "w"), indent=1)
    print(tex)
    print(json.dumps({k: summary[k] for k in ("false_flags", "flood_alert_runs", "fallbacks",
                                             "tokens_in_per_run", "tokens_out_per_run")}, indent=1))


if __name__ == "__main__":
    main()
