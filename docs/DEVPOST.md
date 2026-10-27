# Devpost submission text

Paste each field into the Devpost "Enter a submission" form. Every number matches
`docs/StageSight_report.pdf` and the files in `results/`.

## Project name
StageSight: a river camera that checks its own gauge

## Elevator pitch (≤ 200 characters)
OpenCV 5 reads flood stage off public USGS staff-plate cameras with no gauge data; a Claude agent on AWS decides whether the camera or the gauge is wrong, and a person approves.

## Thumbnail and gallery (in this order)
1. `docs/figures/faults.png` cropped to 3:2 (clean, debris, spray: camera vs gauge and the agent's verdict)
2. `docs/figures/pipeline.png` (the OpenCV 5 pipeline on a real flood frame)
3. `docs/figures/flood.png` (0.26 ft RMSE through the flood)
4. `docs/figures/architecture.png`
5. `docs/figures/dashboard.png` (the live dashboard replaying the frozen-gauge case)
6. `docs/figures/timeline.png`

## About the project

**In short**
- OpenCV 5 reads river stage off a USGS staff-plate camera with **no gauge data**: 0.26 ft RMSE
  through the 29 July 2026 flood (29 of 30 frames within 0.5 ft), where a fitted pixel regression
  is off by 6.3 ft.
- When camera and gauge disagree, a Claude agent on Amazon Bedrock re-runs the OpenCV reader and
  decides which instrument is wrong. In both simulated camera faults the fixed rules blamed the
  healthy gauge; Claude blamed the camera in 3 of 3 runs.
- It runs live on AWS Lambda every 15 minutes, and nothing leaves it until a person approves.

### Inspiration
I study floods as a PhD student in civil engineering. A river gauge reports stage from one
pressure sensor. When that sensor fouls, freezes or goes silent, often in the storm it exists
for, the record shows no sign of it. The USGS already runs cameras at its gauges, many aimed
straight at a staff plate, but people look at those images; nobody measures them. I wanted the
camera to become a second instrument that can check the first.

### What it does
- Every 15 minutes StageSight takes the newest frame from the USGS camera at Hancock, NY, and
  **OpenCV 5 reads the river stage off the staff plate with no gauge data**: the plate's marks give
  the scale and the printed foot numbers, read by OpenCV's DNN text recognizer, give the datum.
- It compares that reading with the USGS gauge. When they disagree, the gauge goes silent or the
  river reaches an NWS flood level, a **Claude agent on Amazon Bedrock** gathers evidence (re-reads
  earlier frames, checks the gauge history and an upstream gauge) and decides which instrument to trust.
- Sensor flags and flood alert drafts go to an **approval queue**. Nothing is sent without a person.
- The live dashboard opens a **replay of the 29 July 2026 flood** when the river is low or it is
  night, so you can watch the agent handle a frozen gauge, a drifting gauge, debris on the plate or
  spray on the lens. You can also upload a frame and get a reading back from Lambda.

### How I built it
- **OpenCV 5 reader:** HSV red mask, `connectedComponentsWithStats` and `fitLine` find the plate
  and its marks; a 1-D projective map turns pixels into feet; `phaseCorrelate` re-registers the
  plate every frame; Lab a* contrast decides which marks are under water; the OpenCV Zoo CRNN, run
  through `cv.dnn.TextRecognitionModel`, reads the datum. A quality gate and explicit statuses
  (`below_plate`, `unreadable`, `camera_moved`) mean it returns a reason instead of a guess.
- **Agent:** code raises events; only events call Claude (Opus 4.6 on Bedrock) in a tool loop
  with three evidence tools and four actions. The tools run inside Lambda; Bedrock only chooses
  them. Agent memory, a safety net that always surfaces flood alerts, and a rule fallback on any
  model error came out of testing.
- **AWS:** two arm64 Lambda functions on Graviton: a reader that also runs the agent, and an API
  that serves the dashboard, the flood replay and `/measure` behind a Function URL. An EventBridge rule
  every 15 minutes, a private S3 bucket, Bedrock, and an SNS topic that e-mails approved alerts. The
  IAM role can write logs, use only that bucket, invoke Anthropic models and publish to that topic.
  The zip is built without Docker by cross-installing the Linux arm64 OpenCV 5 wheels. A tick takes a
  median of 6.3 s (n = 23 CloudWatch runs).
- **EC2 Graviton4 + COOL:** a one-off batch re-reads a camera's archive when a site is set up.
- The same code runs on a laptop CPU, with a replay that injects faults into the real flood.
- Tests (pytest) and lint run in GitHub Actions.

### Results
- **29 July 2026 flood at Hancock, NY (NWS minor flood, 18.08 ft peak): 0.26 ft RMSE against the
  USGS gauge, 29 of 30 readings within 0.5 ft, with no gauge data.** A linear pixel fit trained on
  a smaller September high water has 6.3 ft RMSE. The camera follows both flood peaks.
- **Second test event, 4–5 Sep 2026 high water** (frames the reader never used): 0.34 ft RMSE,
  15 of 15 within 0.5 ft. The water stayed inside one gap between two marks, so this is one
  reading repeated; the archive has no daylight frame spoiled by rain, fog or glare.
- **Injected faults, repeated runs** (Claude Opus 4.6 and Sonnet 4.6 against a rule baseline):
  Opus 4.6 caught all four gauge faults in 3 of 3 runs (frozen gauge in 2 h vs 4 h for the rules,
  drift in 7 h vs 8 h) and blamed the camera in both camera faults in 3 of 3 runs. It is not
  error-free: during the frozen gauge it also flagged the healthy camera at the second peak in 3 of 3
  runs, and there the flood alert came from the safety net. Sonnet 4.6 missed the slow drift in 2 of 3
  runs. The report (Table IV) shows all of it.
- **Camera faults** (debris over the plate, spray on the lens) with a healthy gauge: the rule
  baseline blamed the healthy gauge and published the bad camera stage; Claude blamed the camera.
  In the debris case it noticed the visible marks drop from 11 to 7 while both gauges said the
  river was falling.
- **90 days of normal operation** (1,863 frames): 0 false sensor flags from the rule planner and 0 from Claude Opus 4.6 (one run, 17k input / 2.9k output tokens); both drafted only the action and flood alerts at the real flood. Only two events reached the planner in those 90 days, so this is a small sample, not a measured false-alarm rate.
- **COOL on Graviton4:** with 16 worker processes a c8g.4xlarge re-reads the whole 90-day archive
  (1,869 frames) in 4.1 s, about $0.0008 at on-demand prices, scaling almost linearly from 33 frames/s
  with one worker. COOL gives the same status and stage as pip OpenCV in every frame; it is 4% faster
  in one process and 4.9x faster in connectedComponentsWithStats, while the batch is NumPy-bound, so
  the two builds tie there. COOL's DNN text recognizer aborted on c8g.xlarge and ran on c8g.4xlarge;
  a minimal reproduction is in the repo (report Sec. VI, Table II).

### Challenges I ran into
- My first approach, a self-calibrating waterline detector for any camera, reached 0.29 ft MAE
  in range on a close wall but failed on far-bank bridge cameras and could not extrapolate to
  floods. Reading the plate itself solved the flood problem; the far-bank cameras stay in the
  report as failure cases.
- The OCR dropped the leading "1" of "19"; now only the longest reads in a frame may vote.
- Claude's first run missed the drift, and its second run drafted no flood alert during a real
  flood because it had forgotten its own gauge flag. I fixed it in code (agent memory and a safety
  net), not in the prompt. In the third run it flagged the frozen gauge for a partly wrong reason;
  the report says so.
- A brand-new AWS account had only the 4.6-generation Claude models enabled and low request
  quotas, so the agent retries patiently and falls back to the rules when Bedrock is unavailable.

### Accomplishments that I'm proud of
- An absolute stage reading from an existing public camera, with no survey, no target and no
  gauge data, that holds through a flood above every frame used in setup.
- In both camera-fault replays the agent blamed the right instrument, where fixed thresholds
  blamed the healthy gauge.
- Stated limits: the reader returns a reason instead of a number when it cannot see, and the
  report lists the agent's failures.

### What I learned
When the event you care about lies outside the training range, a geometric model of the plate
extrapolates and a fitted regression does not. The agent helped where two instruments
disagreed; the safety net and human approval covered its mistakes.

### What's next for StageSight
Read night infrared frames from the printed numbers, and support more plate designs and the other
NIMS plate cameras.

### Responsible use
StageSight is a decision aid, not an official warning system. It uses public USGS and NWS data
only, sends nothing until a person approves it (then by SNS e-mail; live approvals need an operator
token), and labels a camera stage provisional whenever it stands in for the gauge.

### Special awards
- **Best Use of COOL:** the archive batch runs on EC2 Graviton4 with COOL 3.1 (cv2 loaded from
  /opt/cool, KleidiCV 0.7.0 in the build info), measured against stock OpenCV on the same instances:
  identical output on 1,869 frames, 454 frames/s with 16 workers, $0.0004 per 1,000 frames, and a
  DNN crash isolated to one instance size with a minimal reproduction (report Table II,
  `results/cool/`; `scripts/cool_run.py` launches, measures and terminates the instance).
- **Agentic Vision:** the OpenCV reading decides what happens next: code turns it into events, and
  `reread_earlier_frames` re-runs the OpenCV reader and returns mark counts. In the debris case the
  drop from 11 to 7 visible marks is what made Claude blame the camera instead of the gauge.
  Evidence: report Sec. V and VII-C, Table IV, Figs. 5 to 7, and the replay on the live site.

## Built with
opencv, opencv-dnn, cool, onnx, python, numpy, aws-lambda, aws-graviton, amazon-ec2, amazon-bedrock,
claude, amazon-s3, amazon-sns, amazon-eventbridge, amazon-cloudwatch, aws-iam, lambda-function-url,
usgs-nims, chart.js, pytest, github-actions

## Team
Muhammad Waqar Saleem (solo): computer vision, agent, AWS deployment, evaluation, report.

## Try it out links
- Live dashboard and flood replay: https://zkdbpwl26gqzom5j3osjaoi4dy0htqoq.lambda-url.us-east-2.on.aws/
- Code: https://github.com/Waqar-msx9t/stagesight

## How to test (paste into "Testing instructions" if the form has it; else end of About)
1. Open the live dashboard. At night or low water it opens the replay of the 29 July 2026 flood;
   `?mode=live` shows the latest real camera frame read on AWS Lambda.
2. In the replay, pick a scenario (for example "Gauge frozen for 10 h" or "Debris covers the plate")
   and press the chips under the time bar to jump to each agent action. The Agent loop panel shows
   the tools Claude called; the decision log has its full reasoning.
3. Approve or Reject in the replay is local only and sends nothing. Live approvals need the
   operator token, so judges see the live queue read-only.
4. "Try the reader": download the sample flood frame from the same panel, or any full 1080x1920 frame
   of the Hancock staff camera from USGS NIMS, and upload it; Lambda returns the stage it read.
5. Offline: `pytest -q` in a fresh clone runs the reader on two real flood frames (README, "Check it
   in 10 minutes").

## Video demo link
https://youtu.be/YnP3MDY5Bgs

## Files to upload
- `docs/StageSight_report.pdf` (technical report)
- `docs/figures/architecture.png` (architecture diagram)
