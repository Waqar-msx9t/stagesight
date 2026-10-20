"""AWS Lambda entry points (zip package on arm64 / Graviton).

scheduled(): EventBridge every 15 min. Reads the newest camera frame with OpenCV, runs the agent
             (Claude on Bedrock, only when the reading raises an event) and writes the results to S3.
api():       Lambda Function URL.
             GET  /latest      latest.json            GET /approvals   approval queue
             POST /measure     body = JPEG/PNG frame from this camera -> stage reading
             POST /approve     {"id": "...", "decision": "approve"|"reject"}

Environment: STORE=s3:<bucket>, SITE_DIR (default /var/task/sites/hancock), UPSTREAM_SITE,
             AGENT_BACKEND=bedrock|rules, AGENT_MODEL (default global.anthropic.claude-opus-4-6-v1).
"""
import base64
import json
import os

from stagesight import service, storage
from stagesight.agent import ClaudePlanner, RulePlanner
from stagesight.reader import SiteReader

SITE_DIR = os.environ.get("SITE_DIR", os.path.join(os.path.dirname(__file__), "sites", "hancock"))
_reader = None


def _ctx():
    global _reader
    if _reader is None:
        _reader = SiteReader(SITE_DIR)
    return _reader, storage.from_env()


def _planner():
    return RulePlanner() if os.environ.get("AGENT_BACKEND") == "rules" else ClaudePlanner()


def scheduled(event, context):
    if isinstance(event, dict) and event.get("selftest"):
        # one tiny Bedrock call: checks the role and model access from inside Lambda
        p = ClaudePlanner()
        r = p.client.messages.create(model=p.model, max_tokens=20,
                                     messages=[{"role": "user", "content": "Reply with the word OK."}])
        return {"model": p.model, "reply": "".join(b.text for b in r.content if b.type == "text")}
    reader, store = _ctx()
    world = service.LiveWorld(reader.site, os.environ.get("UPSTREAM_SITE"))
    return service.tick(store, reader, world, _planner())


_HERE = os.path.dirname(os.path.abspath(__file__))
WEB = next(p for p in (os.path.join(_HERE, "web", "index.html"),               # Lambda package
                       os.path.join(_HERE, "..", "web", "index.html"))         # repo checkout
           if os.path.exists(p))


def _resp(code, body, ctype="application/json"):
    headers = {"Content-Type": ctype, "Cache-Control": "no-cache"}
    if isinstance(body, bytes) and not ctype.startswith(("text/", "application/json")):
        return {"statusCode": code, "headers": headers, "isBase64Encoded": True,
                "body": base64.b64encode(body).decode()}
    if isinstance(body, bytes):
        body = body.decode()
    return {"statusCode": code, "headers": headers, "body": body if isinstance(body, str) else json.dumps(body)}


def api(event, context):
    reader, store = _ctx()
    http = event.get("requestContext", {}).get("http", {})
    method, path = http.get("method", "GET"), event.get("rawPath", "/latest")
    raw = event.get("body") or ""
    data = base64.b64decode(raw) if event.get("isBase64Encoded") else raw.encode("latin1")
    # the dashboard is served from here too, so the S3 bucket stays private
    if method == "GET" and path in ("/", "/index.html"):
        return _resp(200, open(WEB, encoding="utf-8").read(), "text/html; charset=utf-8")
    if method == "GET" and path == "/config.js":
        return _resp(200, 'window.STAGESIGHT={data:"data/",api:""};', "text/javascript")
    if method == "GET" and path.startswith("/data/"):
        key = path[len("/data/"):]
        if key not in ("latest.json", "timeseries.json", "events.json", "approvals.json", "latest_overlay.jpg"):
            return _resp(404, {"error": "unknown key"})
        body = store.get(key)
        if body is None:
            return _resp(404, {"error": f"{key} not written yet"})
        return _resp(200, body, "image/jpeg" if key.endswith(".jpg") else "application/json")
    if method == "GET" and path == "/latest":
        return _resp(200, (store.get("latest.json") or b"{}").decode())
    if method == "GET" and path == "/approvals":
        return _resp(200, (store.get("approvals.json") or b"[]").decode())
    if method == "POST" and path == "/measure":
        out = service.measure_bytes(reader, data)
        return _resp(400 if "error" in out else 200, out)
    if method == "POST" and path == "/approve":
        try:
            req = json.loads(data or b"{}")
        except json.JSONDecodeError:
            return _resp(400, {"error": "body must be JSON"})
        out = service.decide(store, str(req.get("id")), str(req.get("decision")), str(req.get("who", "operator")))
        return _resp(404 if "error" in out else 200, out)
    return _resp(404, {"error": f"no route {method} {path}"})
