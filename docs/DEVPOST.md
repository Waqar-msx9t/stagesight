# Devpost submission text

Paste each field into the Devpost "Enter a submission" form. Numbers match `docs/StageSight_report.pdf`.

## Project name
StageSight: a river camera that checks its own gauge

## Elevator pitch (≤ 200 characters)
OpenCV 5 reads flood stage off USGS staff-plate cameras with no gauge data (0.26 ft RMSE), and a Claude agent on AWS decides whether the camera or the gauge sensor is wrong.

## About the project

### Inspiration
I study floods as a PhD student in civil engineering. A river gauge reports stage from one
pressure sensor, and when that sensor drifts, freezes or goes silent, usually in the storm it
exists for, the record does not say so. The USGS already runs cameras at its gauges, many
looking straight at a staff plate, but people look at those images; nobody measures them. I
wanted the camera to become a second instrument that can check the first.

### What it does
- Every 15 minutes StageSight takes the newest frame from the USGS camera at Hancock, NY, and
  **OpenCV 5 reads the river stage off the staff plate**, with no gauge data: the plate marks
  give the scale and the printed foot numbers, read by OpenCV's DNN text recognizer, give the datum.
- It compares that reading with the USGS gauge. When they disagree, the gauge goes silent or the
  river reaches an NWS flood level, a **Claude agent on Amazon Bedrock** gathers evidence
  (re-reads earlier frames, checks the gauge history and an upstream gauge) and decides which
  instrument to trust.
- Sensor flags and flood alert drafts go to an **approval queue** on the dashboard. Nothing is
  sent automatically.
- Anyone can POST their own frame from this camera to the live endpoint and get a reading back.

### How I built it
- **OpenCV 5 plate reader:** HSV red mask, `connectedComponentsWithStats` and `fitLine` find the
  plate and its marks; a 1D projective map turns pixels into feet; `phaseCorrelate` re-registers
  the plate every frame; Lab a* contrast decides which marks are under water; the CRNN model from
  the OpenCV Zoo, run through `cv.dnn.TextRecognitionModel`, reads the datum. A quality gate and
  explicit statuses (`below_plate`, `unreadable`, `camera_moved`) mean it never guesses.
- **Agent:** code raises events; only events call Claude (Opus 4.6 on Bedrock) in a tool loop
  with three evidence tools and four actions. Agent memory, a safety net that always puts a flood
  alert in front of a person, and a rule-based fallback came out of testing.
- **AWS:** two arm64 Lambda functions on Graviton (scheduled reader + agent; API and dashboard
  behind a Function URL), EventBridge every 15 minutes, a private S3 bucket, a narrow IAM role,
  and Bedrock. The 78 MB zip is built without Docker by cross-installing the Linux arm64 OpenCV 5 wheels.
- The same code runs on a laptop CPU, with a replay mode that injects faults into the real flood.

### Results
- **Jul 29 2026 flood at Hancock, NY (NWS minor flood, 18.08 ft): 0.26 ft RMSE against the USGS
  gauge, 97% of readings within 0.5 ft, n = 30, with no gauge data used.** A linear pixel fit
  trained on normal flows missed by 6.3 ft. The camera follows both flood peaks.
- **Injected gauge faults** on the same flood (offset, 8 h silence, frozen sensor, slow drift):
  caught by the rule baseline and by Claude; Claude caught the frozen gauge in 2 h vs 4 h and the
  drift in 7 h vs 8 h.
- **Injected camera faults** (debris over the plate, spray on the lens) with a healthy gauge:
  the rule baseline blamed the healthy gauge both times and published the bad camera stage;
  **Claude blamed the camera both times.** In the debris case it noticed the visible marks drop
  from 11 to 7 while both gauges said the river was falling.
- **90 days of normal operation** (rule planner, 1,863 replayed frames): 0 false sensor flags.
- **On AWS:** a 15-minute tick takes a median of 6.3 s on Lambda (Graviton); Claude is called only on events.

### Challenges I ran into
- My first approach, a self-calibrating waterline detector for any camera, worked on a close
  wall (0.29 ft in range) but failed on far-bank bridge cameras and could not extrapolate to
  floods. Reading the plate itself solved the flood problem; the far-bank cameras stay in the
  report as failure cases.
- The OCR dropped the leading "1" of "19"; only two-digit reads may vote now.
- Claude's first run missed the drift, and its second run drafted no flood alert during a real
  flood because it had forgotten its own gauge flag. I fixed the system (memory, safety net),
  not just the prompt, and the report documents all three runs.
- A brand-new AWS account had only Claude Opus 4.6 enabled and very low request quotas, so the
  agent retries patiently and falls back to the rules when Bedrock is unavailable.

### Accomplishments that I'm proud of
- An absolute stage reading from an existing public camera, with no survey, no target and no
  gauge data, that holds through a flood a learned model never saw.
- An agent that adds something rules cannot: deciding which instrument is wrong.
- Honest limits: the reader returns a reason instead of a number when it cannot see.

### What I learned
Geometry beats learning when the training range does not include the event you care about. And
an LLM agent earns its place where judgement between conflicting evidence is needed; the safety
net and the human approval are what make it safe to use.

### What's next for StageSight
Read night infrared frames from the printed numbers, support more plate designs and the other
NIMS plate cameras, and repeat the agent runs to report catch rates.

### Responsible use
StageSight is a decision aid, not an official warning system. It uses public USGS and NWS data
only, sends nothing without a person's approval, and labels every camera-based stage as provisional.

## Built with
opencv, opencv-dnn, python, numpy, aws-lambda, aws-graviton, amazon-bedrock, claude, amazon-s3,
amazon-eventbridge, usgs-nims, chart.js

## Try it out links
- Live dashboard: https://zkdbpwl26gqzom5j3osjaoi4dy0htqoq.lambda-url.us-east-2.on.aws/
- Code: https://github.com/Waqar-msx9t/stagesight (private; add the judges as collaborators or attach a zip)

## Video demo link
[unlisted YouTube link, ≤ 5 min]

## Files to upload
- `docs/StageSight_report.pdf` (technical report, 5 pages)
- `docs/figures/architecture.png` (architecture diagram)
- `docs/figures/pipeline.png` (good gallery image)
