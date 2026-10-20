"""The StageSight agent: OpenCV readings decide when to think and what to do next.

Every frame is read with OpenCV (cheap). Claude is only called when the reading raises an
event: camera and gauge disagree, the gauge goes silent, the camera goes blind during high
water, or the stage rises past an NWS level. It then gathers more evidence with tools and ends
with an action. Actions that reach people (sensor flags, flood alerts) go to an approval queue;
nothing is sent automatically.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from botocore.exceptions import BotoCoreError

from stagesight.reader import SiteReader, public

TOL_FT = 0.5            # camera vs gauge disagreement threshold (plate readings are ~0.26 ft RMSE)
CONFIRM_FRAMES = 2      # consecutive frames before a disagreement becomes an event
LEVELS = ["normal", "action", "flood"]

TOOL_SPECS = [
    {"name": "reread_earlier_frames",
     "description": "Read the stage with OpenCV from the n previous hourly camera frames (n from 1 "
                    "to 4) to check whether the camera reading is stable.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}},
                      "required": ["n"], "additionalProperties": False}},
    {"name": "get_site_gauge",
     "description": "USGS pressure-sensor readings (ft) at this site over the last `hours` hours.",
     "input_schema": {"type": "object", "properties": {"hours": {"type": "number"}},
                      "required": ["hours"], "additionalProperties": False}},
    {"name": "get_upstream_gauge",
     "description": "Readings (ft) at the upstream USGS gauge over the last `hours` hours. A rise "
                    "upstream usually reaches this site a few hours later.",
     "input_schema": {"type": "object", "properties": {"hours": {"type": "number"}},
                      "required": ["hours"], "additionalProperties": False}},
    {"name": "flag_sensor",
     "description": "Put a sensor-fault flag in the human approval queue, naming the instrument you "
                    "believe is wrong and the evidence.",
     "input_schema": {"type": "object", "properties": {
         "suspect": {"type": "string", "enum": ["site_gauge", "camera"]},
         "reason": {"type": "string"}, "evidence": {"type": "string"}},
         "required": ["suspect", "reason", "evidence"], "additionalProperties": False}},
    {"name": "draft_alert",
     "description": "Draft a flood alert for a human to approve. It is never sent automatically.",
     "input_schema": {"type": "object", "properties": {
         "level": {"type": "string", "enum": ["action", "flood"]}, "message": {"type": "string"}},
         "required": ["level", "message"], "additionalProperties": False}},
    {"name": "publish_provisional_stage",
     "description": "Publish the camera stage as the provisional site stage while the gauge is "
                    "missing or flagged. Labelled provisional on the dashboard.",
     "input_schema": {"type": "object", "properties": {
         "stage_ft": {"type": "number"}, "reason": {"type": "string"}},
         "required": ["stage_ft", "reason"], "additionalProperties": False}},
    {"name": "no_action",
     "description": "Close the event without action and say why.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}},
                      "required": ["reason"], "additionalProperties": False}},
]
FINAL_TOOLS = {"flag_sensor", "draft_alert", "publish_provisional_stage", "no_action"}

SYSTEM = """You are StageSight's monitoring agent for one USGS river gauge that also has a camera
pointed at a staff plate. OpenCV reads the plate every hour by checking which painted marks are
still visible above the water; its readings are accurate to about 0.26 ft RMSE when the water is
on the plate. The pressure-sensor gauge is normally the reference, but sensors drift, freeze,
jump or go silent, often during floods. The camera can be wrong too: debris, a leaf mat, spray or
droplets on the lens hide marks above the water and make the camera read too high. Either
instrument can be the faulty one; decide from the evidence which is more plausible.

Site facts an operator would know:
- Camera readings move in steps of about 0.5 ft (the spacing of the painted marks), so several
  identical readings in a row are normal while the river changes by less than a step.
- Changes at the upstream gauge reach this site about 3 hours later (cross-correlation of 90 days
  of records); a flood peak can arrive later than that. Compare trends with that lag in mind.

You are called only when the camera reading raised an event. Gather the evidence you need with
the read-only tools, then finish with one or more actions:
- flag_sensor when the evidence shows an instrument is wrong (say which and why);
- publish_provisional_stage when the gauge is missing or flagged and the camera is reliable;
- draft_alert when the stage reaches the NWS action or flood level (a person approves it);
- no_action when the event does not hold up.
Prefer independent evidence: earlier frames, the gauge history, the upstream gauge trend. Do not
flag the gauge on a single frame. Keep each reason to one or two sentences with the numbers."""


@dataclass
class AgentState:
    recent: list = field(default_factory=list)          # last readings (dicts)
    disagree_count: int = 0
    blind_count: int = 0
    open_issues: set = field(default_factory=set)       # events already handled and still true
    last_level: str = "normal"
    agree_streak: int = 0
    history: list = field(default_factory=list)         # the agent's own recent actions


class Toolbox:
    """Executes tool calls against the world; records final actions."""

    def __init__(self, reader: SiteReader, world):
        self.reader, self.world = reader, world
        self.actions: list[dict] = []
        self._cache: dict[str, dict] = {}

    def read(self, back: int) -> dict | None:
        fr = self.world.frame(back)
        if fr is None:
            return None
        t, img = fr
        key = str(t)
        if key not in self._cache:
            r = public(self.reader.read(img))
            r["time_utc"] = key
            self._cache[key] = r
        return self._cache[key]

    def run(self, name: str, args: dict) -> dict:
        if name == "reread_earlier_frames":
            n = max(1, min(int(args.get("n", 2)), 4))
            out = []
            for k in range(1, n + 1):
                r = self.read(k)
                if r:
                    out.append({k2: r.get(k2) for k2 in ("time_utc", "status", "stage_ft", "below_ft",
                                                          "marks_visible", "marks_total", "mark_errors")})
            return {"frames": out}
        if name == "get_site_gauge":
            r = self.world.gauge(min(float(args.get("hours", 3)), 24))
            step = max(1, len(r) // 24)                                 # at most ~24 readings
            return {"readings": (r[::step] + r[-1:]) if r and r[::step][-1] != r[-1] else r[::step]}
        if name == "get_upstream_gauge":
            r = self.world.upstream(min(float(args.get("hours", 12)), 48))
            return {"readings": r[::max(1, len(r) // 12)][-12:]}
        if name in FINAL_TOOLS:
            act = {"type": name, **args, "time_utc": str(self.world.now)}
            if name in ("flag_sensor", "draft_alert"):
                act["status"] = "pending_approval"
            self.actions.append(act)
            return {"ok": True, "queued": name in ("flag_sensor", "draft_alert")}
        return {"error": f"unknown tool {name}"}


class RulePlanner:
    """Deterministic baseline with the same tools: used offline, in tests, and as fallback."""

    name = "rules"

    def handle(self, event: dict, tools: Toolbox) -> list[dict]:
        trace = []

        def call(name, args):
            res = tools.run(name, args)
            trace.append({"tool": name, "input": args, "output": res})
            return res

        kind = event["kind"]
        if kind == "disagree":
            earlier = call("reread_earlier_frames", {"n": 3})["frames"]
            gauge = call("get_site_gauge", {"hours": 5})["readings"]
            # pair each camera frame with the gauge at the same time (within 20 min)
            gaps = [event["camera_ft"] - event["gauge_ft"]]
            for f in earlier:
                if f["stage_ft"] is None:
                    continue
                tf = np.datetime64(f["time_utc"])
                near = sorted((abs(np.datetime64(t) - tf), g) for t, g in gauge)
                if near and near[0][0] <= np.timedelta64(20, "m"):
                    gaps.append(f["stage_ft"] - near[0][1])
            same_side = [g for g in gaps if abs(g) > TOL_FT and np.sign(g) == np.sign(gaps[0])]
            if len(gaps) >= 2 and len(same_side) >= 2 and len(same_side) >= 0.66 * len(gaps):
                call("flag_sensor", {"suspect": "site_gauge",
                                     "reason": f"camera minus gauge = {', '.join(f'{g:+.2f}' for g in gaps)} ft over "
                                               f"{len(gaps)} frames; consistent offset, camera within 0.3 ft of "
                                               f"itself frame to frame",
                                     "evidence": json.dumps({"gaps_ft": [round(g, 2) for g in gaps],
                                                             "gauge_last": gauge[-3:]})})
                call("publish_provisional_stage", {"stage_ft": event["camera_ft"],
                                                   "reason": "gauge flagged; camera agrees with itself"})
            else:
                call("no_action", {"reason": f"camera-gauge gaps {[round(g, 2) for g in gaps]} not consistent "
                                             f"enough to flag the gauge"})
        elif kind == "gauge_silent":
            call("get_upstream_gauge", {"hours": 6})
            call("publish_provisional_stage", {"stage_ft": event["camera_ft"],
                                               "reason": "no gauge reading in the last 30 min"})
        elif kind == "camera_blind":
            call("flag_sensor", {"suspect": "camera", "reason": f"{event['frames']} frames unreadable "
                                 f"while gauge is {event['gauge_ft']} ft", "evidence": event["statuses"]})
        elif kind == "level_rise":
            up = call("get_upstream_gauge", {"hours": 12})["readings"]
            trend = (up[-1][1] - up[0][1]) if len(up) >= 2 else 0.0
            call("draft_alert", {"level": event["level"],
                                 "message": f"Stage {event['stage_ft']:.2f} ft reached the NWS {event['level']} "
                                            f"level (camera and gauge). Upstream change over 12 h: {trend:+.2f} ft."})
        return trace


class ClaudePlanner:
    """Claude decides which evidence to gather and which actions to take (manual tool loop)."""

    name = "claude"
    BUDGET_S = 100      # per event: 100 s + one 80 s call stays inside the 300 s Lambda timeout

    def __init__(self, backend: str | None = None, model: str | None = None, max_steps: int = 8):
        backend = backend or os.environ.get("AGENT_BACKEND", "bedrock")
        if backend == "bedrock":
            # Bedrock runtime with a cross-region inference profile; AGENT_MODEL selects the model.
            from anthropic import AnthropicBedrock
            self.client = AnthropicBedrock(   # new accounts have low request quotas: retry patiently,
                aws_region=os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1")),
                max_retries=1, timeout=40)    # worst case per call 80 s; see BUDGET_S below
            self.model = model or os.environ.get("AGENT_MODEL", "global.anthropic.claude-opus-4-6-v1")
        else:
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model or os.environ.get("AGENT_MODEL", "claude-opus-5-5")
        self.max_steps = max_steps
        self.fallback = RulePlanner()
        self.usage = [0, 0]          # input, output tokens (cost tracking)

    def handle(self, event: dict, tools: Toolbox) -> list[dict]:
        trace = []
        messages = [{"role": "user", "content": "Event from the OpenCV monitor:\n" + json.dumps(event, indent=1)}]
        import anthropic
        t0 = time.monotonic()
        for _ in range(self.max_steps):
            if time.monotonic() - t0 > self.BUDGET_S:
                # the 300 s Lambda must still write its results: hand the event to the rules
                trace.append({"note": "model time budget used up; rule planner used"})
                return trace + self.fallback.handle(event, tools)
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=8000, system=SYSTEM, tools=TOOL_SPECS,
                    thinking={"type": "adaptive"}, output_config={"effort": "medium"}, messages=messages)
            except (anthropic.APIError, BotoCoreError) as exc:
                # the monitor must keep working when the model is unreachable: log it, use the rules
                trace.append({"note": f"model unavailable ({type(exc).__name__}); rule planner used"})
                return trace + self.fallback.handle(event, tools)
            self.usage[0] += resp.usage.input_tokens
            self.usage[1] += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                trace.append({"note": "model declined; rule planner used", "category":
                              getattr(resp.stop_details, "category", None)})
                return trace + self.fallback.handle(event, tools)
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            if text:
                trace.append({"thought": text})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not calls:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for c in calls:
                out = tools.run(c.name, dict(c.input))
                trace.append({"tool": c.name, "input": dict(c.input), "output": out})
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
        if not tools.actions:
            trace.append({"note": "no final action from the model; rule planner used"})
            trace += self.fallback.handle(event, tools)
        return trace


class Monitor:
    """One tick per camera frame: read, detect events, let the planner act, keep the state."""

    def __init__(self, reader: SiteReader, world, planner, state: AgentState | None = None):
        self.reader, self.world, self.planner = reader, world, planner
        self.state = state or AgentState()

    def _level(self, stage):
        return self.reader.alert_level(stage) or "normal"

    def detect(self, r: dict, gauge: float | None) -> list[dict]:
        st, ev = self.state, []
        cam = r["stage_ft"]
        # disagreement (also when the camera says the water is below the plate but the gauge is on it)
        gap = None
        if cam is not None and gauge is not None:
            gap = cam - gauge
        elif r["status"] == "below_plate" and gauge is not None and gauge > r["below_ft"] + TOL_FT:
            gap = r["below_ft"] - gauge
        if gap is not None and abs(gap) > TOL_FT:
            st.disagree_count += 1
            st.agree_streak = 0
            if st.disagree_count >= CONFIRM_FRAMES and "disagree" not in st.open_issues:
                ev.append({"kind": "disagree", "camera_ft": cam if cam is not None else r["below_ft"],
                           "camera_status": r["status"], "gauge_ft": gauge, "frames": st.disagree_count,
                           "camera_marks_visible": r.get("marks_visible"),
                           "camera_mark_errors": r.get("mark_errors")})
        elif gap is not None:
            st.disagree_count = 0
            st.agree_streak += 1
            if st.agree_streak >= 3:
                st.open_issues.discard("disagree")
        # silent gauge
        if gauge is None and cam is not None and "gauge_silent" not in st.open_issues:
            ev.append({"kind": "gauge_silent", "camera_ft": cam})
        if gauge is not None:
            st.open_issues.discard("gauge_silent")
        # blind camera during high water (daylight frames only reach here as unreadable/rejected)
        action = self.reader.site.get("action_stage_ft") or 1e9
        if r["status"] in ("unreadable", "camera_moved") and gauge is not None and gauge >= action:
            st.blind_count += 1
            if st.blind_count >= 3 and "camera_blind" not in st.open_issues:
                ev.append({"kind": "camera_blind", "frames": st.blind_count, "gauge_ft": gauge,
                           "statuses": [x["status"] for x in st.recent[-3:]]})
        else:
            st.blind_count = 0
        # level rise (camera when readable, else gauge)
        stage = cam if cam is not None else gauge
        lvl = self._level(stage)
        if LEVELS.index(lvl) > LEVELS.index(st.last_level):
            ev.append({"kind": "level_rise", "level": lvl, "stage_ft": stage})
        if stage is not None:
            st.last_level = lvl
        return ev

    def tick(self) -> dict:
        tools = Toolbox(self.reader, self.world)
        r = tools.read(0)
        gauge = self.world.gauge_now()
        rec = {"time_utc": str(self.world.now), "camera": r, "gauge_ft": gauge, "events": [], "actions": []}
        if r is None:
            return rec
        self.state.recent = (self.state.recent + [r])[-6:]
        for ev in self.detect(r, gauge):
            # memory: what the agent already concluded (it must not forget a flag it raised)
            ev["open_issues"] = sorted(self.state.open_issues)
            ev["recent_agent_actions"] = self.state.history[-6:]
            n_before = len(tools.actions)
            trace = self.planner.handle(ev, tools)
            new = tools.actions[n_before:]
            if ev["kind"] == "level_rise" and not any(a["type"] == "draft_alert" for a in new):
                # safety net: an NWS level shown by an instrument always reaches a human; the
                # agent's doubts go in the message, the human decides
                doubt = next((a.get("reason") for a in new if a.get("reason")), "no reason given")
                tools.actions.append({"type": "draft_alert", "level": ev["level"], "status": "pending_approval",
                                      "time_utc": str(self.world.now), "safety_net": True,
                                      "message": f"Stage {ev['stage_ft']:.2f} ft reached the NWS {ev['level']} "
                                                 f"level on one instrument. The agent disputes it: {doubt}"})
                trace.append({"note": "safety net: alert drafted for human review despite agent doubt"})
                new = tools.actions[n_before:]
            self.state.history = (self.state.history + [
                {"time_utc": str(self.world.now), "event": ev["kind"], "type": a["type"],
                 **{k: a[k] for k in ("suspect", "level", "stage_ft") if k in a}} for a in new])[-20:]
            acted = any(a["type"] != "no_action" for a in new)
            # an event closed with no_action stays open for re-checking on the next frame
            if ev["kind"] != "level_rise" and acted:
                self.state.open_issues.add(ev["kind"])
            rec["events"].append({"event": ev, "planner": self.planner.name, "trace": trace})
        rec["actions"] = tools.actions
        return rec
