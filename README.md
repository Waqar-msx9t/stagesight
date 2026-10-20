# StageSight: a river camera that checks its own gauge

OpenCV 5 reads the river stage off the staff plate in a USGS gauge camera, **with no gauge data**.
An agent (Claude on Amazon Bedrock) compares that reading with the gauge sensor, decides which
instrument is wrong, and queues sensor flags and flood alerts for a person to approve. It runs live
on AWS (Lambda on Graviton, EventBridge, S3, Bedrock) and, unchanged, on a laptop CPU.

- **Live dashboard:** https://zkdbpwl26gqzom5j3osjaoi4dy0htqoq.lambda-url.us-east-2.on.aws/
  When the river is low or it is night, it opens a replay of the 29 July 2026 flood with the agent's
  recorded decisions; pick a fault in the scenario menu. `?mode=live` shows the live camera.
- **Technical report:** [docs/StageSight_report.pdf](docs/StageSight_report.pdf) (5 pages, [LaTeX source](docs/paper/stagesight.tex))

![Pipeline](docs/figures/pipeline.png)

## Results

Hancock, NY (USGS 01421500), flood of 29 July 2026 (NWS minor flood, 18.08 ft peak). 30 daylight
frames with water on the plate, 27 of them above every frame used to train the baselines:

| Method | Gauge data | RMSE (ft) | Within 0.5 ft |
|---|---|---|---|
| **StageSight** (plate marks + OCR datum) | none | **0.26** (95% bootstrap 0.20 to 0.31) | **29 of 30** |
| Plate scale + offset fitted to the gauge | training | 0.37 | 25 of 30 |
| Linear pixel fit, trained on normal flows | training | 6.31 | 7 of 30 |

![Flood](docs/figures/flood.png)

Agent replays of the same flood with injected faults are in report Table III, with repeated runs of
Claude Opus 4.6 and Sonnet 4.6 against a rule baseline (`results/agent/repeat_summary.json`). In
both camera-fault cases (debris over the plate, spray on the lens) the rules blame the healthy gauge;
Claude blames the camera. Over 90 days of normal operation (1,863 frames) the rule planner raised
0 false sensor flags. The reader gives a number for 2.4% of all hourly frames: it reads floods; below
the plate or at night it reports `below_plate` or `unreadable` instead of guessing.

## How it works
**Reader** (`stagesight/plate.py`, `stagesight/reader.py`)
- *Setup, once per camera, no gauge data* (`scripts/setup_site.py`): HSV red mask,
  `cv.connectedComponentsWithStats` and `cv.fitLine` find the plate and its marks; the foot marks give
  a 1-D projective map from pixels to feet (fit error 0.05 ft); the printed numbers, read with
  `cv.dnn.TextRecognitionModel` (OpenCV Zoo CRNN), give the datum (20 ft in 6 of 7 setup frames).
- *Every frame*: quality gate; `cv.phaseCorrelate` re-registers the plate; each mark is visible or
  hidden by its Lab a* contrast against its surround; the waterline is the best visible/hidden split.

**Agent** (`stagesight/agent.py`): code raises an event (camera and gauge disagree, gauge silent,
camera blind in high water, NWS level reached); only then does Claude run a tool loop:
`reread_earlier_frames`, `get_site_gauge`, `get_upstream_gauge` (USGS 01417500), then
`flag_sensor` or `draft_alert` (approval queue), `publish_provisional_stage` or `no_action`.
Agent memory, a safety net that always surfaces flood alerts, and a rule fallback on any model error.

**AWS** ([diagram](docs/figures/architecture.png)): EventBridge (15 min) → Lambda reader (arm64,
OpenCV 5, agent loop and tools; each hourly frame processed once) → Bedrock on events → private S3 →
Lambda API with a Function URL (dashboard, replay, `POST /measure`, `POST /approve` with an operator
token). Median tick 6.3 s, API 47 ms.

## Check it in 10 minutes (fresh clone, no downloads)
```bash
uv venv --python 3.12 && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt pytest
pytest -q                                                 # plate reader on 2 real flood frames, agent, API
python scripts/serve_local.py --mode live          # dashboard + flood replay at http://localhost:8000
```

## Reproduce every number
```bash
python scripts/download_site.py --cam NY_East_Branch_Delaware_River_at_Hancock_Staff --site 01421500 \
    --start 2026-07-21 --end 2026-10-20 --upstream 01417500 --out data/hancock
python scripts/flood_stress_test.py --index data/hancock/images.csv --gauge data/hancock/gauge.csv \
    --reference sites/hancock/reference.jpg --cache results/hancock_stack.npz \
    --test-start 2026-07-21 --test-end 2026-08-01 --no-align --out results/hancock_v2      # frame cache
python scripts/plate_test.py --cache results/hancock_stack.npz --reference sites/hancock/reference.jpg \
    --test-start 2026-07-21 --test-end 2026-08-01 --out results/hancock_plate             # Table II
python scripts/replay_agent.py --all --planner rules                               # rule baseline
bash scripts/repeat_agent_runs.sh && python scripts/agent_stats.py                 # Table III (Bedrock)
python scripts/replay_agent.py --scenario none --start 2026-07-21 --end 2026-10-19 --out results/agent_90d
uv pip install -r requirements-dev.txt && python scripts/make_figures.py         # docs/figures
python scripts/build_replay.py                                                     # web/replay
cd docs/paper && latexmk -pdf stagesight.tex                                              # the report
```
`results/` holds the metrics, traces and summaries behind every number in the report.

## Deploy to AWS
```bash
python aws/build_package.py                  # zip with Linux arm64 wheels and the replay, no Docker
AGENT_BACKEND=bedrock bash aws/deploy.sh     # bucket, IAM role, 2 functions, schedule, Function URL
```
The role can read and write only the project bucket and invoke Anthropic models on Bedrock. Failed
ticks are not retried. The first deploy prints the operator token that live approvals need; it is
stored in the API function's environment.

## Repository map
| Path | What |
|---|---|
| `stagesight/plate.py`, `reader.py` | plate reader (report Sec. IV) |
| `stagesight/agent.py`, `service.py`, `world.py`, `storage.py`, `usgs.py` | agent, live tick, replay world, S3 or local store, USGS and NIMS clients |
| `stagesight/selfcal.py`, `vision.py`, `calibrate.py`, `scripts/flood_stress_test.py` | research mode for cameras without a plate (report Sec. VIII) and the frame cache |
| `aws/`, `web/` | Lambda handlers, build and deploy, dashboard and flood replay |
| `scripts/` | data download, setup, evaluation, repeated agent runs, figures |
| `results/`, `docs/` | evidence behind every number; report, figures, Devpost text |

## Limitations
Needs a plate with red marks in view and daylight; one reading resolves 0.1 to 0.8 ft, set by the
mark spacing; below the plate (about 11 ft) the camera cannot check the gauge. The Hancock camera
started in April 2026, so the 29 July flood is its only flood, and the camera faults in the agent
evaluation are synthetic. StageSight is a decision aid, not an official warning system.

MIT licensed; third-party material in [NOTICE](NOTICE).
