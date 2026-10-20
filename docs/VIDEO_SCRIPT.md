# StageSight demo video script

**Limit:** 5:00 or less (Devpost rules; the video must show the team, the working app, the
architecture and the main results). **Target: 4:40**, 20 s under the limit.
Upload to YouTube as Unlisted and add corrected English captions in YouTube Studio.

Every number below is in `README.md`, `docs/DEVPOST.md` or the report (`docs/paper/*.tex`).
Do not add a number while recording that is not in those files.

Dashboard: `https://zkdbpwl26gqzom5j3osjaoi4dy0htqoq.lambda-url.us-east-2.on.aws/` (called `DASH` below).
Panel names below are the panel titles on the page; the visual design may change, so find
panels by name, not by position or color.

## Timed script

| Time | On screen | Narration (spoken) | Caption |
|---|---|---|---|
| 0:00–0:10 | `DASH?mode=replay&s=flatline`, scrubbed to about 13:00 ET (17:00 UTC) so the **Camera vs gauge** chart shows the gauge line flat inside the grey fault band while the camera points rise. Slow zoom on the flat line. | River gauges fail during floods, exactly when warnings need them. We built StageSight so the camera at the gauge can check it. | River gauges fail when floods need them most |
| 0:10–0:22 | Webcam, full screen or picture-in-picture over the dashboard header. | I am Muhammad Waqar Saleem, a PhD student in civil engineering at the University of Missouri. I built StageSight alone, with OpenCV 5 on AWS. | Team: Muhammad Waqar Saleem (solo), University of Missouri |
| 0:22–0:45 | `DASH` (default view). Point at the four numbered steps in the header (OpenCV reads → Claude checks → a person approves → SNS e-mail), the **Live / Replay** toggle, then the **status strip**, then the **Now** panel: the stage meter (camera and gauge water columns against the action and flood lines) and the tiles. The page opens in a dark console theme; the ◐ button switches to light. | This is the live dashboard, served by AWS Lambda. Every 15 minutes it reads the newest frame from the USGS camera at Hancock, New York. When the river is low or it is night, it opens a replay of the 29 July 2026 flood, with real USGS data and the agent's recorded decisions. | Live on AWS Lambda · USGS 01421500, Hancock NY |
| 0:45–1:25 | `docs/figures/pipeline.png` full screen (open the file locally). Move the cursor along the steps as they are named. | OpenCV 5 reads the stage off the staff plate, with no gauge data. Setup runs once per camera. A red color mask, connected components and fitLine find the plate and its foot marks, which give the scale. OpenCV's DNN text recognizer reads the printed numbers, which give the datum. On every frame, a quality gate rejects bad images, phaseCorrelate re-registers the plate, and color contrast tells which marks are under water. | OpenCV 5: connectedComponentsWithStats · fitLine · dnn.TextRecognitionModel · phaseCorrelate |
| 1:25–1:37 | Back to the dashboard, **camera frame** panel, zoom on the overlay. | Green marks are visible, magenta marks are under water, and the yellow line is the waterline the reader found. | Green: visible · Magenta: under water · Yellow: waterline |
| 1:37–2:02 | `docs/figures/flood.png` full screen. | On the 29 July flood, which peaked at 18.08 feet, the reader had 0.26 feet RMSE against the USGS gauge. 29 of 30 readings were within half a foot. A linear pixel fit trained on a smaller September high water was off by 6.3 feet. At night or below the plate, the reader says so instead of guessing. | 0.26 ft RMSE · 29 of 30 within 0.5 ft · no gauge data used |
| 2:02–2:16 | `DASH?mode=replay&s=flatline`. Press play from the start; let the **Agent decision log** fill. | Code watches for events: camera and gauge disagree, the gauge goes silent, or the river reaches a flood level. Only then do we call Claude on Amazon Bedrock. | Claude on Bedrock runs only on events |
| 2:16–2:46 | Same replay. Click the agent-action chip **Jul 29, 11:00: flag gauge** (11:00 ET = 15:00 UTC). Let the **Agent loop** panel light up step by step (event → re-read frames → site gauge → upstream gauge → flag gauge → person approves). Point at the striped red gauge column in the stage meter, the gauge tile **SUSPECT** badge, the triangle on the chart, the tool calls in the **Agent decision log**, then the flag in the **Human approval queue**. | In this replay we froze the gauge at 17.21 feet for 10 hours. Claude re-runs the OpenCV reader on earlier frames, checks the site gauge and an upstream gauge, and flags the gauge two hours after the fault starts. Fixed rules took four. The gauge is now marked suspect, and the flag waits for a person. | Frozen gauge: Claude 2 h, rules 4 h (3 of 3 runs) |
| 2:46–3:14 | Scenario menu → **Debris covers the plate (camera wrong)** (`s=occlusion`). Click the chip **Jul 29, 15:00: flag camera** (15:00 ET = 19:00 UTC). Point at the frame, then highlight the log line about marks falling from 11 to 7. | The camera can be wrong too. Here we put simulated debris over the plate. The rule baseline blamed the healthy gauge and published the bad camera stage. Claude saw the visible marks drop from 11 to 7 while both gauges were falling, and blamed the camera. It got this right in 3 of 3 runs, and for spray on the lens too. | Debris: rules blamed the gauge; Claude blamed the camera (3 of 3) |
| 3:14–3:32 | **Human approval queue**: click **Approve** in the replay (it shows "approved (replay, nothing sent)"). Cut to the mail inbox with the real SNS email from the live system (see checklist). | Nothing leaves the system without a person. In the public replay, Approve changes nothing. On the live system, an operator approves with a token, and Amazon SNS sends the email. | Human approval → Amazon SNS email |
| 3:32–3:54 | `docs/figures/architecture.png` full screen, then quick cuts of the AWS console: Lambda functions, EventBridge rule, Bedrock, SNS topic. | EventBridge starts the reader every 15 minutes. Both Lambda functions run on Graviton arm64 with OpenCV 5. Results go to a private S3 bucket, Bedrock is called only on events, and SNS sends approved items. A tick takes a median of 6.3 seconds. | Lambda (Graviton arm64) · EventBridge · S3 · Bedrock · SNS |
| 3:54–4:04 | Dashboard **Try the reader**: upload `tests/fixtures/hancock_flood_17.61ft.jpg`, show the stage and the raw JSON. | Anyone can test the reader. We upload a flood frame, and Lambda returns the stage it read. | POST /measure on AWS Lambda |
| 4:04–4:20 | Report Table II (`docs/StageSight_report.pdf`), or `results/cool/` summary. | To set up a new site, we re-read the camera's archive on EC2 Graviton4 with COOL. With 16 workers, a c8g.4xlarge reads all 1,869 frames in 4.1 seconds, for less than a tenth of a cent. | COOL on Graviton4: 1,869 frames in 4.1 s, about $0.0008 |
| 4:20–4:34 | Back to `s=flatline`, chip **Jul 29, 13:00: provisional stage, flood alert** (13:00 ET = 17:00 UTC); point at the safety-net flood alert in the queue. | Limits: one camera, one flood, simulated camera faults, daylight only. Claude also doubted the healthy camera at the second peak, so this flood alert came from our safety net. | Limits: one flood · synthetic camera faults · safety net drafted this alert |
| 4:34–4:40 | GitHub repo page `https://github.com/Waqar-msx9t/stagesight` with the URL large on screen. | Code, report and dashboard are on GitHub. Thank you. | github.com/Waqar-msx9t/stagesight |

## Recording checklist

**Setup**
- Browser window 1920x1080, page zoom 125% (raise to 150% for the log close-ups if text is small).
  Hide bookmarks bar and extensions; use a clean profile so no personal tabs show.
- Record with OBS Studio (or Windows Snipping Tool video), 1080p, 30 fps. Record each row as its own
  clip and join them in Clipchamp; retakes stay short.
- Webcam for 0:10–0:22 only (the rules require the team on screen).

**Preload these tabs, in order**
1. `DASH?mode=replay&s=flatline`, paused, scrubbed to about 13:00 ET (hook shot).
2. `DASH` (default view, for the dashboard tour).
3. `docs/figures/pipeline.png`, `docs/figures/flood.png`, `docs/figures/architecture.png` (open locally).
4. `DASH?mode=replay&s=flatline` paused at the start (agent segment).
5. `DASH?mode=replay&s=occlusion` (debris segment; you can also switch with the scenario menu).
6. `docs/StageSight_report.pdf` open at Table II (COOL).
7. AWS console, region **us-east-2 (Ohio)**, signed in before recording (see below).
8. Mail inbox with the SNS email open.
9. `https://github.com/Waqar-msx9t/stagesight`, scrolled to the README title.

**What to click**
- Agent segment: play the flatline replay from the start, then click the chip **Jul 29, 11:00: flag gauge** (gauge flag,
  2 h after the 09:00 ET fault start; the dashboard shows Eastern time on chips). Hold on the decision log long enough to read one tool call.
- Debris segment: chip **Jul 29, 15:00: flag camera**. The log line "Visible marks fell from 11 to 7"
  should be readable; zoom in during editing if not.
- Approval: in the replay, Approve only shows "approved (replay, nothing sent)". Say so, as scripted.
- Limits shot: flatline, chip **Jul 29, 13:00**: the flood alert there is marked as drafted by the safety net.
- Try the reader: upload `tests/fixtures/hancock_flood_17.61ft.jpg` (1080x1920, the only size
  `/measure` accepts). Show the stage and the raw JSON. Do not state an expected value on camera
  unless it appears in the response.

**AWS console proof (3:32–3:54, about 3 s each)**
- Lambda → Functions: `stagesight-scheduled` and `stagesight-api`, with Architecture **arm64** visible;
  open `stagesight-api` → Configuration → Function URL (matches `DASH`).
- EventBridge → Rules: `stagesight-every-15-min` (rate 15 minutes, target `stagesight-scheduled`).
- Bedrock → Model access (Anthropic Claude enabled), or CloudWatch logs of a tick showing a Bedrock call.
- SNS → Topics: `stagesight-alerts` with a **Confirmed** email subscription.
- S3: the private project bucket (Block public access: On). Optional if time is short.
- Blur or crop the account ID, the operator token (Lambda environment variables) and the email address.
  Never open the `stagesight-api` environment variables tab on camera unblurred.

**SNS email proof**
- Use a real email that SNS sent after a live approval. If none exists yet, approve a real pending item
  on the live dashboard (`?mode=live`) with the operator token, off camera, and record the inbox.
- If the live queue is empty before recording, show the SNS topic and subscription in the console and
  change the narration line to: "On the live system, an operator approves with a token, and Amazon SNS
  emails the result." Do not present a console "Publish message" test as an approved alert; if you use
  one, caption it "SNS test message".

**Close**
- GitHub repo page with the URL on screen; the report PDF is linked from the README
  (`docs/StageSight_report.pdf`). Make sure the repo is public before recording.

## Word count and pace check

| Section | Seconds | Words | Words/min |
|---|---|---|---|
| Hook 0:00–0:10 | 10 | 22 | 132 |
| Team 0:10–0:22 | 12 | 25 | 125 |
| Dashboard 0:22–0:45 | 23 | 53 | 138 |
| Pipeline 0:45–1:25 | 40 | 71 | 106 |
| Overlay 1:25–1:37 | 12 | 19 | 95 |
| Accuracy 1:37–2:02 | 25 | 58 | 139 |
| Events 2:02–2:16 | 14 | 28 | 120 |
| Frozen gauge 2:16–2:46 | 30 | 56 | 112 |
| Debris 2:46–3:14 | 28 | 62 | 133 |
| Approval + SNS 3:14–3:32 | 18 | 30 | 100 |
| Architecture 3:32–3:54 | 22 | 43 | 117 |
| Try the reader 3:54–4:04 | 10 | 17 | 102 |
| COOL 4:04–4:20 | 16 | 36 | 135 |
| Limits 4:20–4:34 | 14 | 29 | 124 |
| Close 4:34–4:40 | 6 | 9 | 90 |
| **Total** | **280 (4:40)** | **558** | **120** |

558 spoken words in 4:40 is about 120 words/min overall, and no row is above 140 words/min, so
there is room to pause while clicking. If a full read-through runs long, the video can stretch to
about 4:50 and still be 10 s under the 5:00 limit. Time one read-through aloud before recording.
