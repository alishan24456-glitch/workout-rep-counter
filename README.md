# Workout Rep Counter — AI Gym Trainer

Counts exercise repetitions from a pose skeleton and coaches form, live.
It tracks the body's joint angles (MediaPipe Pose) and counts a rep only
when the tracked angle completes a full down-and-up cycle, with a depth
grade for every rep.

> **Honest framing:** this is a geometry heuristic on pose landmarks
> (joint angles), not a trained action classifier and not a medical
> device. It counts a rep when the tracked angle completes a full
> down-up cycle; it cannot tell a good squat from a dangerous one, and
> the exact degree number at the bottom of a rep is not a reliable
> measurement on footage where the knee is occluded (see "Where real
> mode struggles"). Real-mode reliability depends on a clear, mostly
> full-body, side or three-quarter view of one person with the camera
> roughly level and fixed; a person half out of frame, or two people
> overlapping, will confuse it.

## What it counts

| Exercise | Tracked joint | DOWN | UP | GOOD DEPTH |
|---|---|---|---|---|
| Squat | knee (hip fallback) | < 130° | > 160° | min angle ≤ 100° |
| Push-up | elbow | < 110° | > 155° | min angle ≤ 90° |
| Bicep curl | elbow | < 75° | > 145° | min angle ≤ 70° |

- **Rep** = angle crosses DOWN, then returns past UP. Half-reps never
  count.
- **Depth verdict** (from the smallest angle reached during the rep):
  GOOD DEPTH / SHALLOW / TOO SHALLOW, per the table above.
- **Torso lean** (squats only): torso more than 35° from vertical at
  the bottom of a rep logs a "KEEP CHEST UP" tip.
- **Exercise auto-detect** (`--exercise auto`, the default) watches
  the first 2 s of movement: a near-horizontal torso means push-ups,
  a big knee-angle swing means squats, a big elbow swing with
  straight legs means bicep curls. Pass `--exercise squat`,
  `pushup` or `curl` to fix it.
- **Noise guards** (measured on real gym footage, not assumed): a rep
  that completes in under 0.6 s is landmark noise, not exercise, and
  is logged and ignored; and DOWN can only trigger after the angle
  has sat above the UP threshold for 10 straight frames (a confirmed
  standing / extended baseline), so the noisy first frames of a clip
  cannot open a phantom rep. Without these guards the demo clip
  counted 15 reps; with them it counts the true 8 (see below).
- **Live overlay**: skeleton, body box, big REP counter, live joint
  angle + UP/DOWN phase, a DEPTH gauge that fills as the joint
  closes, and good/shallow/too-shallow totals. The status panel sits
  in its own strip above the video (not overlaid on it) so it can
  never cover the person's head.

## Run it (Windows CMD)

```
pip install -r requirements.txt
python workout_rep_counter.py                  REM opens a video file chooser
python workout_rep_counter.py --source pick    REM same chooser, explicitly
python workout_rep_counter.py --source gym.mp4
python workout_rep_counter.py --source gym.mp4 --exercise squat --loop
python workout_rep_counter.py --source 0       REM webcam
python workout_rep_counter.py --simulate       REM rendered gym
```

Press **Q** to quit. The annotated video is written to `output.mp4`
(change with `--output`); the timestamped event log goes to
`events.log` (change with `--log`); the end-of-run summary goes to
`output_summary.txt` (change with `--summary`). Useful options:
`--exercise squat|pushup|curl|auto`, `--loop` (loop a short video
until `--max-seconds`), `--max-seconds 30`, `--no-display` for
headless runs.

MediaPipe Pose is needed **only for real mode**. The script supports
both the classic `mp.solutions` Pose and the current Tasks API
PoseLandmarker (newer mediapipe builds removed `mp.solutions`); with
the Tasks API the `pose_landmarker_lite.task` model auto-downloads on
first use (it is also included in this repo). On Linux the Tasks API
needs the system OpenGL libraries (`libegl1`, `libgles2`) installed;
without them real mode fails at startup with a `libEGL.so.1` error and
the script says so plainly. Simulation mode (`--simulate`) never needs
mediapipe at all.

## Modes

- **Real mode** — MediaPipe Pose tracks the body in a video file or
  webcam feed and supplies shoulder/hip/knee/ankle/elbow/wrist
  landmarks to the same rep counter. The side (left/right) with the
  better landmarks is used per body part, which is what makes
  side-view gym footage work.
- **Simulation mode** (`--simulate`) — a gym is rendered with OpenCV:
  a figure performs five squats. The scene draws small coloured joint
  markers (hip blue, knee green, ankle red, shoulder yellow, elbow
  magenta, wrist cyan) and the pixel estimator finds those markers in
  the rendered frame and computes the joint angles from them. The rep
  counter only ever sees those pixel-derived angles; it never peeks
  at the scene's scripted choreography.

## Verified demo runs

**Primary demo — REAL footage (30.0 s, his own code on natural
footage).** A man doing barbell squats in a gym, side view, fixed
camera (Pexels, free license: "Man Shoulder Squats with an Empty
Barbell" by Harvey Tan Villarino, video 6114481, 1920×1080, 25 fps,
7.6 s clip looped 4× to 30 s with `--loop`; the tracker resets at
each loop point, logged in the event log):

- **750 frames, pose on 728 (97.1%)**, 5 pose-lost spells.
- **8 REPS, all GOOD DEPTH**, mean rep duration 1.3 s — exactly 2
  squats per loop × 4 loops, matching a frame-by-frame count of the
  clip (squats bottom out at ~3.2–3.8 s and ~5.8–6.2 s in each loop).
  0 fast cycles rejected, 0 torso-lean warnings.
- **Auto-detect check** (separate 12 s run, `--exercise auto`):
  SQUAT detected at 2.0 s (torso 29°, knee range 148°, elbow range
  51°), 2 reps in 12 s — the auto path works end to end.
- H.264 LinkedIn copy exported and codec-verified h264; frames
  inspected at 3.5 s (deep squat, DOWN, depth gauge full) and 5 s
  (standing, UP, REPS: 1).

**Where real mode struggles (measured, same rebuild):** at the bottom
of each squat the squat-rack upright crosses the knee, and MediaPipe's
knee landmark jumps ~50 px off the true knee for ~0.4 s. The printed
minimum knee angle per rep (31–41°) is therefore understated — the
clean frames just before the occlusion read ~80–90°, which is the
true deep-squat angle. The depth verdict (GOOD DEPTH) is still
correct — the squats are visibly deep — but the exact degree number
at the bottom is not a reliable measurement on this clip, and the
skeleton visibly kinks at the bottom of the squat in the demo (see
the 3.5 s frame). The rep count itself is unaffected: the angle still
completes a full down-up cycle on every squat.

**Simulation — rep-counting logic verification (30 s, 900 frames):**
pose estimated on **900/900 frames (100%)**; **5 reps, all GOOD
DEPTH** (min angle 83–84° at the bottom of each squat), mean rep
duration 1.8 s, 0 pose-lost spells, 0 fast cycles rejected. This
rendered simulation is a logic test only — the primary demo above
is the real-footage one.

**Not verified in this build:** push-up and curl modes share the
exact same counter code (verified via squats in both modes), but no
push-up or curl footage was run — their thresholds come from standard
joint-angle ranges for those exercises, not from a measured run.

## Bugs found and fixed during verification

1. **Phantom reps on real footage (the big one).** The first real-mode
   run counted **15 reps** for 8 real squats: the tracker re-locking
   at the clip start and at each loop point produced 0.1–0.2 s
   down-up cycles. Fixed with the two noise guards above (0.6 s
   minimum rep duration + 10-frame arming above the UP threshold);
   the recount is exactly 8, with 0 rejected fast cycles — the
   arming guard stops the phantoms from even starting.
2. **Simulation pose estimator blind.** The wrist joint marker was
   never drawn (missing from the render loop), so the estimator
   returned no sample on any frame. Fixed by drawing it.
3. **Hip marker swallowed by the torso.** The hip marker shared its
   colour with the shirt, so the estimator found the whole torso
   instead of the hip disc. Fixed by giving the hip marker its own
   colour (pure blue).
4. **Knee marker hidden mid-squat.** The hanging wrist crossed the
   knee disc at the bottom of the squat and covered it (60 of 900
   simulation frames lost). Fixed by swinging the arms back as the
   figure descends; the simulation is now 900/900.
5. **Status panel covered the person's head.** The overlay bar sat on
   top of the video, and on this clip the head sits as high as y=23
   in a 720p frame — the first renders hid the head entirely. Fixed
   by compositing: the panel lives in its own strip above a scaled
   video area, so the full person is always visible.
6. **Simulation body box too small.** It came from a saturation mask
   that missed the low-saturation trousers and ended at the hips.
   Fixed by building the box from the joint markers.
7. **mediapipe 1.2.x API move.** `BaseOptions` moved from
   `tasks.python.vision` (1.1.x) to `tasks.python.core` (1.2.x), and
   Tasks API landmarks carry `presence`, not `visibility`. The
   provider now tries both locations / attributes. Also, on this
   Linux build machine the Tasks API needs the stock Ubuntu mesa
   libraries (`libegl1`, `libgles2`, `libegl-mesa0`, `libgbm1`,
   `libglapi-mesa`, `libglvnd0`) extracted to a throwaway folder and
   pointed at with `LD_LIBRARY_PATH` — nothing installed system-wide.
   On a normal Windows PC real mode starts as-is.

## Files

- `workout_rep_counter.py` — the whole system (single file)
- `pose_landmarker_lite.task` — MediaPipe Pose model (auto-downloaded
  if missing; only used by real mode)
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer
