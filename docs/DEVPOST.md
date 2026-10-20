# Devpost submission text

Paste each field into the Devpost "Enter a submission" form. Every number matches
`docs/StageSight_report.pdf` and the files in `results/`.

## Project name
StageSight: a river camera that checks its own gauge

## Elevator pitch (≤ 200 characters)
OpenCV 5 reads flood stage off public USGS staff-plate cameras with no gauge data; a Claude agent on AWS decides whether the camera or the gauge is wrong, and a person approves.

## Thumbnail and gallery (in this order)
1. `docs/figures/faults.png` cropped to 16:9 (clean, debris, spray: camera vs gauge and the agent's verdict)
2. `docs/figures/pipeline.png` (the OpenCV 5 pipeline on a real flood frame)
3. `docs/figures/flood.png` (0.26 ft RMSE through the flood)
4. `docs/figures/architecture.png`
5. A screenshot of the live dashboard in replay mode (frozen-gauge scenario, approval queue visible)
6. `docs/figures/timeline.png`

## About the project

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
- **AWS:** two arm64 Lambda functions on Graviton (reader and agent; API, dashboard and replay
  behind a Function URL), an EventBridge rule every 15 minutes, a private S3 bucket, an IAM role
  limited to that bucket and Anthropic models, and Bedrock. The zip is built without Docker by
  cross-installing the Linux arm64 OpenCV 5 wheels. A tick takes a median of 6.3 s (n = 23 CloudWatch runs).
- The same code runs on a laptop CPU, with a replay that injects faults into the real flood.
- Tests (pytest) and lint run in GitHub Actions.

### Results
- **29 July 2026 flood at Hancock, NY (NWS minor flood, 18.08 ft peak): 0.26 ft RMSE against the
  USGS gauge, 29 of 30 readings within 0.5 ft, with no gauge data.** A linear pixel fit trained on
  normal flows has 6.3 ft RMSE. The camera follows both flood peaks.
- **Injected faults, repeated runs** (Claude Opus 4.6 and Sonnet 4.6 against a rule baseline):
  [fill from report Table III: catch rates and median delays].
- **Camera faults** (debris over the plate, spray on the lens) with a healthy gauge: the rule
  baseline blamed the healthy gauge and published the bad camera stage; Claude blamed the camera.
  In the debris case it noticed the visible marks drop from 11 to 7 while both gauges said the
  river was falling.
- **90 days of normal operation** (rule planner, 1,863 frames): 0 false sensor flags.

### Challenges I ran into
- My first approach, a self-calibrating waterline detector for any camera, reached 0.29 ft MAE
  in range on a close wall but failed on far-bank bridge cameras and could not extrapolate to
  floods. Reading the plate itself solved the flood problem; the far-bank cameras stay in the
  report as failure cases.
- The OCR dropped the leading "1" of "19"; now only two-digit reads may vote.
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
- Honest limits: the reader returns a reason instead of a number when it cannot see, and the
  report lists the agent's failures.

### What I learned
When the event you care about lies outside the training range, a geometric model of the plate
extrapolates and a fitted regression does not. The agent helped where two instruments
disagreed; the safety net and human approval covered its mistakes.

### What's next for StageSight
Read night infrared frames from the printed numbers, support more plate designs and the other
NIMS plate cameras, and send approved alerts through Amazon SNS.

### Responsible use
StageSight is a decision aid, not an official warning system. It uses public USGS and NWS data
only, sends nothing without a person's approval (live approvals need an operator token), and
labels a camera stage provisional whenever it stands in for the gauge.

### Special awards
- **Agentic Vision:** the OpenCV reading decides what happens next: code turns it into events,
  and the agent's evidence gathering (re-reading frames, mark counts) changes which instrument is
  flagged. Evidence: report Sec. V and VII-C, Table III, Fig. 5 to 7, and the replay on the live site.

## Built with
opencv, opencv-dnn, onnx, python, numpy, aws-lambda, aws-graviton, amazon-bedrock, claude,
amazon-s3, amazon-eventbridge, aws-iam, lambda-function-url, usgs-nims, chart.js, pytest,
github-actions

## Team
Muhammad Waqar Saleem (solo): computer vision, agent, AWS deployment, evaluation, report.

## Try it out links
- Live dashboard and flood replay: https://zkdbpwl26gqzom5j3osjaoi4dy0htqoq.lambda-url.us-east-2.on.aws/
- Code: https://github.com/Waqar-msx9t/stagesight (make it public before submitting)

## Video demo link
[unlisted YouTube link, 5 min or less]

## Files to upload
- `docs/StageSight_report.pdf` (technical report)
- `docs/figures/architecture.png` (architecture diagram)
