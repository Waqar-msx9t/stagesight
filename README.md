# StageSight

A USGS gauge camera reads the river stage off its staff plate with OpenCV 5, and an agent uses
that reading to check the gauge's own sensor: it flags a drifting, frozen or offset sensor,
fills in when the gauge goes silent, and drafts flood alerts for a person to approve.
Runs on AWS (Lambda on Graviton, Bedrock, S3) and, unchanged, on a laptop.

**Result, Hancock NY (USGS 01421500), Jul 29 2026 flood (NWS minor flood):** 0.26 ft RMSE
against the USGS gauge, 97% of readings within 0.5 ft, on 30 daylight frames with the water on
the plate. **No gauge data is used to read the stage.** A pixel-regression baseline fitted on
normal flows missed by 6.3 ft. Agent replay with injected gauge faults: offset, dropout,
flatline and drift all caught (1 to 8 h), and 0 false sensor flags over 90 days (1,869 frames).

Technical report: [docs/REPORT.pdf](docs/REPORT.pdf) ([source](docs/REPORT.md)). Architecture:
[docs/architecture.png](docs/architecture.png). Figures: `python scripts/make_figures.py`
(needs `matplotlib` and `SciencePlots`, dev only).

## How the camera reads the plate (OpenCV 5)
1. **Setup, once per camera, no gauge data** (`scripts/setup_site.py`):
   - Red plate marks: HSV mask and connected components; the plate axis from `cv.fitLine`.
   - The foot marks' positions give the camera's 1D projective map from pixels to feet
     (3-parameter homography; fit error 0.05 ft; missing marks detected).
   - The datum is read from the printed foot numbers with the OpenCV 5 DNN
     `TextRecognitionModel` (CRNN, OpenCV Zoo) and a majority vote; only two-digit reads may vote.
2. **Every frame** (`stagesight/plate.py`, `stagesight/reader.py`):
   - Quality gate (blur, darkness).
   - Re-locate the plate with `cv.phaseCorrelate` on its always-dry top; the camera shifts a few px between days.
   - Each mark is visible or hidden by its Lab a* contrast against the plate around it. This
     holds in sun, shade and overcast; water and foam give no contrast.
   - Waterline = best split between visible marks above and hidden marks below; stage = datum − plate reading.
   - Status `below_plate`, `unreadable` (night, fog, spray), `rejected` or `camera_moved`
     instead of a wrong number.

A second, self-calibrated mode for cameras without a plate (`stagesight/selfcal.py`) learns a
per-pixel **flood-stage map** from the archive and the gauge: 0.29 ft MAE in range on the same
wall, but it does not extrapolate to floods. It fails on far-bank bridge cameras (Hermann MO,
De Soto KS), which are reported as failure cases.

## The agent (`stagesight/agent.py`)
- OpenCV runs every frame. **Claude is only called on an event**: camera and gauge disagree for
  2 frames, the gauge goes silent, the camera goes blind in high water, or the stage passes an
  NWS level.
- Tools: re-read earlier frames, the gauge history, the upstream gauge (USGS 01417500, Harvard NY).
- Actions: `flag_sensor` and `draft_alert` go to an **approval queue** (never sent
  automatically); `publish_provisional_stage` and `no_action`.
- Planners: `ClaudePlanner` (Claude Opus 4.6 on Bedrock, manual tool loop) and `RulePlanner`
  (same tools, deterministic). The rule planner is the offline baseline, the test oracle and
  the fallback if the model declines.
- Every decision and tool call goes to `events.json` and is shown on the dashboard.

## Run it on a laptop
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt
python tests/test_synthetic.py && python tests/test_plate.py
python scripts/serve_local.py --mode replay --fault flatline     # replay the flood with a frozen gauge
python scripts/serve_local.py --mode live                        # live USGS camera, every 15 min
# open http://localhost:8000
```

## Reproduce the numbers
```bash
python scripts/download_site.py --cam NY_East_Branch_Delaware_River_at_Hancock_Staff --site 01421500 --days 90 --out data/hancock
python scripts/flood_stress_test.py --index data/hancock/images.csv --gauge data/hancock/gauge.csv \
    --reference data/hancock/reference.jpg --cache results/hancock_stack.npz \
    --test-start 2026-07-21 --test-end 2026-08-01 --no-align --out results/hancock_v2      # builds the frame cache
python scripts/plate_test.py --cache results/hancock_stack.npz --reference data/hancock/reference.jpg \
    --test-start 2026-07-21 --test-end 2026-08-01 --out results/hancock_plate             # plate accuracy
python scripts/replay_agent.py --all --planner rules                                       # injected faults
python scripts/replay_agent.py --scenario none --start 2026-07-21 --end 2026-10-19 --out results/agent_90d
```

## AWS
- One zip package for arm64, built without Docker (`aws/build_package.py` cross-installs the
  `aarch64-manylinux_2_28` wheels with `uv`), running as two Lambda functions on Graviton:
  - `handler.scheduled`, triggered by an EventBridge rule every 15 minutes;
  - `handler.api`, behind a Function URL: the dashboard (`/`), `GET /latest`, `GET /approvals`,
    `POST /measure` (your own frame), `POST /approve`.
- The agent calls Claude on Amazon Bedrock (`AGENT_MODEL`, default `global.anthropic.claude-opus-4-6-v1`,
  the most capable Claude enabled for this new account).
- A private S3 bucket holds the results; the API function serves them, so the bucket is never public.
- IAM role limited to that bucket and `bedrock:InvokeModel*`.
- Environment: `STORE=s3:<bucket>`, `AGENT_BACKEND=bedrock|rules`, `AGENT_MODEL`, `BEDROCK_REGION`, `UPSTREAM_SITE=01417500`.

Deploy (AWS CLI v2 signed in with `aws login`, `uv` installed):
```bash
python aws/build_package.py                  # build/stagesight_lambda.zip
AGENT_BACKEND=bedrock bash aws/deploy.sh     # bucket, role, 2 functions, schedule, Function URL
```
`aws/Dockerfile` is an unused alternative (container image) for accounts that prefer ECR.

## Data
Public USGS data only: camera frames from the NIMS API, gage height (parameter 00065) from USGS
Water Services. NWS action and flood stages for HKCN6 from the NWPS API (14 ft and 18 ft).

## Limitations
- Plate mode needs a plate with red foot and half-foot marks, in view and lit. Night infrared
  frames are "unreadable". Readings step in 0.5 ft (the mark spacing).
- The camera at Hancock started on Apr 14, 2026, so the Jul 29 flood is its only flood: the
  accuracy rests on 30 frames.
- Faults below the plate (stage under ~11 ft) cannot be checked by the camera.
- StageSight is a decision aid, not an official warning system.
