"""Run the whole StageSight service on a laptop: ticks, agent, dashboard and API.

  python scripts/serve_local.py --mode replay --fault flatline     # replay the Jul 29 flood
  python scripts/serve_local.py --mode live                        # live USGS camera, every 15 min
Then open http://localhost:8000

The same code runs on AWS (aws/handler.py); only the store (folder vs S3) and the clock differ.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from stagesight import service, storage  # noqa: E402
from stagesight.agent import ClaudePlanner, RulePlanner  # noqa: E402
from stagesight.reader import SiteReader  # noqa: E402
from stagesight.world import ReplayWorld, inject_fault, load_series  # noqa: E402
from scripts.replay_agent import SCENARIOS  # noqa: E402


def ticker(a, reader, store, planner):
    if a.mode == "live":
        while True:
            try:
                world = service.LiveWorld(reader.site, a.upstream_site)
                rec = service.tick(store, reader, world, planner)
                print("tick", rec["time_utc"], rec["camera"].get("status"), rec["camera"].get("stage_ft"))
            except Exception as exc:  # keep serving the dashboard if one tick fails
                print("tick failed:", exc)
            time.sleep(a.interval)
    gt, gv = load_series(a.gauge)
    if a.fault != "none":
        kind, start, hours, size = SCENARIOS[a.fault]
        gv, _ = inject_fault(gt, gv, kind, start, hours, size)
    world = ReplayWorld(a.images, (gt, gv), load_series(a.upstream))
    frames = world.frame_t[(world.frame_t >= np.datetime64(a.start)) & (world.frame_t < np.datetime64(a.end))]
    for t in frames:
        world.now = t
        rec = service.tick(store, reader, world, planner)
        cam = rec["camera"]
        print(f"replay {rec['time_utc']}  camera={cam.get('stage_ft')} ({cam.get('status')})  "
              f"gauge={rec['gauge_ft']}  events={rec['events']}")
        time.sleep(a.interval)
    print("replay finished")


def make_handler(reader, store, web_dir):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=web_dir, **kw)

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/config.js":
                return self._send(200, b'window.STAGESIGHT={data:"data/",api:"api"};', "text/javascript")
            if path.startswith("/data/"):
                key = path[len("/data/"):]
                body = store.get(key)
                ctype = "image/jpeg" if key.endswith(".jpg") else "application/json"
                return self._send(200, body, ctype) if body is not None else self._send(404, {"error": key})
            return super().do_GET()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            if self.path == "/api/measure":
                out = service.measure_bytes(reader, body)
                return self._send(400 if "error" in out else 200, out)
            if self.path == "/api/approve":
                req = json.loads(body or b"{}")
                out = service.decide(store, str(req.get("id")), str(req.get("decision")), "local operator")
                return self._send(404 if "error" in out else 200, out)
            return self._send(404, {"error": self.path})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="replay", choices=["replay", "live"])
    ap.add_argument("--fault", default="none", choices=list(SCENARIOS))
    ap.add_argument("--planner", default="rules", choices=["rules", "claude"])
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between ticks (live: 900)")
    ap.add_argument("--site", default="sites/hancock")
    ap.add_argument("--images", default="data/hancock/images.csv")
    ap.add_argument("--gauge", default="data/hancock/gauge.csv")
    ap.add_argument("--upstream", default="data/hancock/upstream_01417500.csv")
    ap.add_argument("--upstream-site", default="01417500")
    ap.add_argument("--start", default="2026-07-28T12:00:00")
    ap.add_argument("--end", default="2026-08-01T00:00:00")
    ap.add_argument("--store", default="var/bucket")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.mode == "live" and a.interval < 60:
        a.interval = 900
    shutil.rmtree(a.store, ignore_errors=True)
    store = storage.LocalStore(a.store)
    reader = SiteReader(a.site)
    planner = ClaudePlanner() if a.planner == "claude" else RulePlanner()
    threading.Thread(target=ticker, args=(a, reader, store, planner), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(reader, store, os.path.join(ROOT, "web")))
    print(f"StageSight running at http://localhost:{a.port}  ({a.mode}, fault={a.fault}, planner={a.planner})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
