#!/usr/bin/env python3
"""
Workout Rep Counter -- AI Gym Trainer
======================================

Counts exercise repetitions from a pose skeleton and coaches form, live:

* **Rep counting** -- a rep is one full down-and-up cycle of the tracked
  joint angle (knee for squats, elbow for push-ups and bicep curls).
  The counter only increments when the angle crosses the DOWN threshold
  and then returns past the UP threshold, so half-reps and jitter never
  count.
* **Form feedback** -- every rep is graded on depth (how far the joint
  angle closed at the bottom of the rep): GOOD DEPTH / SHALLOW / TOO
  SHALLOW, plus a torso-lean warning for squats.
* **Exercise auto-detect** -- ``--exercise auto`` (the default) watches
  the first seconds of movement: a near-horizontal torso means
  push-ups, a big knee-angle swing means squats, a big elbow swing
  with straight legs means bicep curls. Pass ``--exercise squat``,
  ``pushup`` or ``curl`` to fix it.

One shared rep-counting pipeline serves both run modes:

* **Real mode** (``--source``) -- MediaPipe Pose tracks the body in a
  video file or webcam feed and supplies shoulder/hip/knee/ankle/
  elbow/wrist landmarks. The side (left/right) with the better
  landmarks is used each frame, which is what makes side-view gym
  footage work.
* **Simulation mode** (``--simulate``) -- a gym is rendered with
  OpenCV: a figure performs five squats. The scene draws small
  coloured joint markers (hip blue, knee green, ankle red, shoulder
  yellow, elbow magenta) and the pixel estimator finds those markers
  in the rendered frame and computes the joint angles from them. The
  rep counter only ever sees those pixel-derived angles; it never
  peeks at the scene's scripted choreography (it does not know "a
  squat is scheduled at 4 s").

How a rep is judged -- and what it is NOT
------------------------------------------
This is a geometry heuristic on pose landmarks (joint angles), not a
trained action classifier and not a medical device. It counts a rep
when the tracked angle completes a full down-up cycle; it cannot tell
a good squat from a dangerous one, and it says so in the README.
Real-mode reliability depends on a clear, mostly full-body, side or
three-quarter view of one person with the camera roughly level and
fixed; a person half out of frame, or two people overlapping, will
confuse it. When the knee/ankle landmarks are missing (feet out of
frame), squats fall back to the hip angle (shoulder-hip-knee), which
also closes as the body drops.

* ``knee_angle`` = angle at the knee between hip-knee-ankle (squats).
* ``hip_angle`` = angle at the hip between shoulder-hip-knee.
* ``elbow_angle`` = angle at the elbow between shoulder-elbow-wrist
  (push-ups, curls).
* **DOWN** = angle below the exercise's down threshold; **UP** = angle
  above its up threshold. A rep completes on the return to UP after a
  DOWN. Angles are median-filtered over the last 5 frames.
* **Noise guards** -- two real failure modes measured on gym footage:
  (1) a rep that completes in under 0.6 s is landmark noise, not
  exercise (the tracker re-locking at a clip start or loop point, or
  a knee jumping behind a rack upright), and is logged and ignored;
  (2) DOWN can only trigger after the angle has sat above the UP
  threshold for 10 straight frames (a confirmed standing / extended
  baseline), so the noisy first frames of a clip cannot open a
  phantom rep.
* **Depth verdict** (from the smallest angle reached during the rep):
  squat knee < 100 deg or hip < 95 deg = GOOD DEPTH; 100-130 deg =
  SHALLOW; above that = TOO SHALLOW. Push-up elbow < 90 deg = GOOD
  DEPTH; 90-120 deg = SHALLOW; above = TOO SHALLOW. Curl elbow
  < 70 deg = FULL CURL; 70-100 deg = PARTIAL; above = TOO SHALLOW.
* **Torso lean** (squats only): torso more than 35 deg from vertical
  at the bottom of a rep logs a "KEEP CHEST UP" tip.

The rendered figure in the simulation is drawn from the same joint
geometry the pixel estimator measures, so what the viewer sees
matches what the counter measures.

Examples
--------
    python workout_rep_counter.py --simulate
    python workout_rep_counter.py --simulate --output demo.mp4 --no-display
    python workout_rep_counter.py --source gym.mp4
    python workout_rep_counter.py --source gym.mp4 --exercise squat --loop
    python workout_rep_counter.py --source 0
    python workout_rep_counter.py --source pick
    python workout_rep_counter.py            (opens the file picker)

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720   # simulation canvas
SIM_FPS = 30                        # simulation frame rate
FLOOR_Y = 620                       # floor line in the simulation gym

SMOOTH_WINDOW = 5        # median filter over recent joint angles
AUTO_OBSERVE_S = 2.0     # seconds of movement watched before auto-detect

# Per-exercise thresholds (degrees). "down" = angle at the bottom of a
# rep, "up" = angle at the top. depth_good/depth_ok grade the rep.
EXERCISES = {
    "squat": {
        "joint": "knee", "fallback_joint": "hip",
        "down": 130.0, "up": 160.0,
        "down_fallback": 130.0, "up_fallback": 160.0,
        "depth_good": 100.0, "depth_ok": 130.0,
        "label": "SQUAT", "unit": "reps",
    },
    "pushup": {
        "joint": "elbow", "fallback_joint": None,
        "down": 110.0, "up": 155.0,
        "down_fallback": None, "up_fallback": None,
        "depth_good": 90.0, "depth_ok": 120.0,
        "label": "PUSH-UP", "unit": "reps",
    },
    "curl": {
        "joint": "elbow", "fallback_joint": None,
        "down": 75.0, "up": 145.0,
        "down_fallback": None, "up_fallback": None,
        "depth_good": 70.0, "depth_ok": 100.0,
        "label": "BICEP CURL", "unit": "reps",
    },
}

# Simulation joint-marker colours (BGR). Each marker is a solid disc,
# so the pixel estimator can find it by colour alone.
MARKER_COLOURS = {
    "hip": (255, 0, 0),         # pure blue -- distinct from the shirt
    "knee": (0, 200, 60),        # green
    "ankle": (0, 60, 230),       # red
    "shoulder": (0, 220, 230),   # yellow
    "elbow": (230, 0, 200),      # magenta
    "wrist": (230, 200, 0),      # cyan
}
COLOUR_SKIN = (140, 180, 240)       # BGR peach
COLOUR_SHIRT = (230, 120, 30)       # BGR strong blue
COLOUR_TROUSERS = (70, 70, 110)     # BGR dark slate
COLOUR_SHOES = (40, 40, 40)

COLOUR_UP = (0, 190, 70)         # green (BGR)
COLOUR_DOWN = (0, 200, 230)      # amber (BGR)
COLOUR_GOOD = (0, 190, 70)
COLOUR_WARN = (0, 200, 230)
COLOUR_BAD = (0, 55, 230)
COLOUR_TEXT = (255, 255, 255)
COLOUR_SKELETON = (0, 255, 255)  # yellow skeleton
COLOUR_NOPOSE = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def joint_angle(a, b, c) -> float:
    """Angle (deg) at point b between segments b->a and b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def _smoothstep(a: float, b: float, t: float) -> float:
    """Smooth 0..1 interpolation between keyframe values a and b."""
    t = max(0.0, min(1.0, t))
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Joint sample -- the one interface both providers speak
# ---------------------------------------------------------------------------

@dataclass
class JointSample:
    """Everything the rep counter needs, in image coordinates.

    Angles are in degrees (180 = straight joint). ``knee_angle`` /
    ``hip_angle`` / ``elbow_angle`` are None when their landmarks are
    not measurable in this frame. ``torso_angle`` is the shoulder-hip
    line away from vertical. ``points`` are the joints drawn on the
    overlay; ``bbox`` is the body's bounding box.
    """

    knee_angle: float | None
    hip_angle: float | None
    elbow_angle: float | None
    torso_angle: float
    bbox: tuple[int, int, int, int]
    points: dict[str, tuple[int, int]]


class JointProvider:
    """Interface every joint source implements: frame in, sample out."""

    def sample(self, frame: np.ndarray) -> JointSample | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real mode: MediaPipe Pose (Tasks API, classic solutions as fallback)
# ---------------------------------------------------------------------------

class MediaPipePoseProvider(JointProvider):
    """Real-mode provider backed by MediaPipe Pose (lazily imported).

    Works with BOTH mediapipe APIs: the classic solutions Pose when
    present, and the current Tasks API PoseLandmarker on newer builds
    where ``mp.solutions`` no longer exists. Landmark indices are the
    standard Pose layout in both APIs: nose 0, shoulders 11/12, elbows
    13/14, wrists 15/16, hips 23/24, knees 25/26, ankles 27/28.
    """

    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28

    TASKS_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
    TASKS_MODEL_FILE = "pose_landmarker_lite.task"

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Real mode needs the 'mediapipe' package.\n"
                "Install it with:  pip install mediapipe\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc
        self._mp = mp
        self._pose = None        # legacy solutions Pose, when present
        self._landmarker = None  # Tasks API PoseLandmarker, the fallback
        self._frame_index = 0

        # Backend 1 (preferred): the classic solutions Pose, present in
        # mediapipe <= 0.10.14 and many later builds.
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "pose"):
            self._pose = solutions.pose.Pose(
                static_image_mode=False, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            return

        # Backend 2: the current MediaPipe Tasks API. Newer mediapipe
        # builds removed mp.solutions entirely and ship only this API.
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError(
                "This mediapipe build has neither the classic Pose "
                "(mp.solutions) nor the Tasks API.\n"
                "Fix:  pip install --upgrade mediapipe\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc
        self._mp_python = mp_python
        self._mp_vision = mp_vision
        model_path = self._ensure_tasks_model()
        # BaseOptions moved between mediapipe releases: 1.1.x keeps it
        # under vision, 1.2.x under tasks.python.core.
        base_options_cls = getattr(mp_vision, "BaseOptions", None)
        if base_options_cls is None:
            base_options_cls = mp_python.BaseOptions
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options_cls(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                "The MediaPipe Tasks PoseLandmarker failed to start "
                f"({exc}). Try 'pip install --upgrade mediapipe'; on "
                "Linux the system OpenGL libraries may also be missing.\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc

    def _ensure_tasks_model(self) -> str:
        """Download the PoseLandmarker model next to this script if absent."""
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, self.TASKS_MODEL_FILE)
        if not os.path.exists(path):
            print(f"Downloading pose model ({self.TASKS_MODEL_FILE}) ...")
            urllib.request.urlretrieve(self.TASKS_MODEL_URL, path)
            print("Pose model downloaded.")
        return path

    # -- landmark plumbing -------------------------------------------------
    def _landmarks(self, frame, t_ms):
        """Return a list of (x, y, visible) in pixels, or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if self._pose is not None:
            res = self._pose.process(rgb)
            if not res.pose_landmarks:
                return None
            return [(lm.x * w, lm.y * h, lm.visibility >= 0.5)
                    for lm in res.pose_landmarks.landmark]
        # Tasks API path.
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_index += 1
        result = self._landmarker.detect_for_video(mp_image, int(t_ms))
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        # Tasks API landmarks carry `presence` (1.2.x) or `visibility`
        # (older builds); use whichever the build provides.
        return [(p.x * w, p.y * h,
                 getattr(p, "presence",
                         getattr(p, "visibility", 1.0)) >= 0.5)
                for p in lm]

    def sample(self, frame: np.ndarray, t_ms: float = 0) -> JointSample | None:
        lm = self._landmarks(frame, t_ms)
        if lm is None:
            return None
        h, w = frame.shape[:2]

        def mid(a, b):
            return ((lm[a][0] + lm[b][0]) / 2, (lm[a][1] + lm[b][1]) / 2)

        # Pick the better side per body part: in side-view gym footage
        # one side faces the camera and the other is hidden behind the
        # body, so per-part visibility beats a fixed side.
        def best_side(l_idx, r_idx):
            l_vis = lm[l_idx][2]
            r_vis = lm[r_idx][2]
            if l_vis and not r_vis:
                return (lm[l_idx][0], lm[l_idx][1]), True
            if r_vis and not l_vis:
                return (lm[r_idx][0], lm[r_idx][1]), True
            # Both visible (or both hidden): use the one with the
            # higher presence -- here both flags are equal, so prefer
            # the side whose landmarks sit further left in frame is
            # meaningless; take the mean instead.
            return mid(l_idx, r_idx), (l_vis or r_vis)

        shoulder, sh_ok = best_side(self.L_SHOULDER, self.R_SHOULDER)
        hip, hip_ok = best_side(self.L_HIP, self.R_HIP)
        if not (sh_ok and hip_ok):
            return None
        knee, knee_ok = best_side(self.L_KNEE, self.R_KNEE)
        ankle, ankle_ok = best_side(self.L_ANKLE, self.R_ANKLE)
        elbow, elbow_ok = best_side(self.L_ELBOW, self.R_ELBOW)
        wrist, wrist_ok = best_side(self.L_WRIST, self.R_WRIST)

        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]
        torso_angle = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 90.0

        knee_angle = None
        if knee_ok and ankle_ok:
            knee_angle = joint_angle(hip, knee, ankle)
        hip_angle = None
        if knee_ok:
            hip_angle = joint_angle(shoulder, hip, knee)
        elbow_angle = None
        if elbow_ok and wrist_ok:
            elbow_angle = joint_angle(shoulder, elbow, wrist)

        xs = [p[0] for p in lm if p[2]]
        ys = [p[1] for p in lm if p[2]]
        if not xs:
            return None
        bbox = (int(max(0, min(xs) - 10)), int(max(0, min(ys) - 10)),
                int(min(w, max(xs) + 10)), int(min(h, max(ys) + 10)))
        points = {
            "head": (int(lm[self.NOSE][0]), int(lm[self.NOSE][1])),
            "shoulder": (int(shoulder[0]), int(shoulder[1])),
            "hip": (int(hip[0]), int(hip[1])),
            "knee": (int(knee[0]), int(knee[1])),
            "ankle": (int(ankle[0]), int(ankle[1])),
            "elbow": (int(elbow[0]), int(elbow[1])),
            "wrist": (int(wrist[0]), int(wrist[1])),
        }
        return JointSample(knee_angle=knee_angle, hip_angle=hip_angle,
                           elbow_angle=elbow_angle, torso_angle=torso_angle,
                           bbox=bbox, points=points)


# ---------------------------------------------------------------------------
# Simulation mode: rendered gym + pixel-only joint estimation
# ---------------------------------------------------------------------------

@dataclass
class FigurePose:
    """Joint geometry of the rendered figure (degrees / pixels).

    hip: (x, y) of the hip joint. torso: torso angle from vertical,
    positive = leaning toward the facing direction (left). thigh/shin:
    leg segment angles from straight down. arm/forearm: arm segment
    angles from hanging straight down.
    """

    hip: tuple[float, float]
    torso: float = 0.0
    thigh: float = 0.0
    shin: float = 0.0
    arm: float = 10.0
    forearm: float = 5.0


class SimScene:
    """Renders the gym and the figure. Knows the choreography; the
    counter never sees it -- it only ever receives rendered pixels."""

    TORSO_LEN, THIGH_LEN, SHIN_LEN = 150, 120, 120
    ARM_LEN, FOREARM_LEN = 100, 95
    HEAD_R = 26
    BAR_X = 640            # squat-rack uprights flank the figure
    STAND_HIP_Y = 380
    SQUAT_HIP_Y = 500      # hip drops ~120 px at the bottom of a squat

    # Five squats: bottom of each rep at these times (seconds).
    REP_BOTTOMS = [4.0, 9.5, 15.0, 20.5, 26.0]
    REP_HALF_WIDTH = 1.6   # seconds from standing to bottom

    def figure_pose(self, t: float) -> FigurePose:
        """Keyframe choreography for time t (seconds into the run)."""
        # Squat depth profile: 0 = standing, 1 = bottom of the squat.
        depth = 0.0
        for bottom in self.REP_BOTTOMS:
            if abs(t - bottom) < self.REP_HALF_WIDTH:
                k = 1.0 - abs(t - bottom) / self.REP_HALF_WIDTH
                depth = max(depth, k * k * (3 - 2 * k))  # smoothstep
        # Standing idle before the first rep / after the last one.
        hip_y = _smoothstep(self.STAND_HIP_Y, self.SQUAT_HIP_Y, depth)
        # At the bottom: thigh ~78 deg forward, shin ~-18 deg (knee
        # forward over the toes); torso leans ~18 deg.
        return FigurePose(
            hip=(640, hip_y),
            torso=_smoothstep(2.0, 18.0, depth),
            thigh=_smoothstep(0.0, 78.0, depth),
            shin=_smoothstep(0.0, -18.0, depth),
            # Arms swing back as he descends (hands stay clear of the
            # knee marker -- a hanging wrist crosses the knee disc at
            # the bottom of the squat and hides it from the estimator).
            arm=_smoothstep(10.0, -45.0, depth),
            forearm=_smoothstep(5.0, 30.0, depth))

    def _segment(self, origin, length, angle_deg, up=False):
        """End point of a segment starting at origin, angle from
        vertical (positive = toward the facing direction, left).
        ``up=True`` points the segment upward (torso/head); the
        default points it downward (legs, hanging arms)."""
        a = math.radians(angle_deg)
        dx = -math.sin(a)
        dy = -math.cos(a) if up else math.cos(a)
        return (origin[0] + length * dx, origin[1] + length * dy)

    def joints(self, t: float) -> dict[str, tuple[float, float]]:
        """Joint positions of the rendered figure at time t."""
        p = self.figure_pose(t)
        hx, hy = p.hip
        shoulder = self._segment((hx, hy), self.TORSO_LEN, p.torso, up=True)
        head = self._segment(shoulder, 32, p.torso, up=True)
        knee = self._segment((hx, hy), self.THIGH_LEN, p.thigh)
        ankle = self._segment(knee, self.SHIN_LEN, p.shin)
        elbow = self._segment(shoulder, self.ARM_LEN, p.arm)
        wrist = self._segment(elbow, self.FOREARM_LEN, p.forearm)
        return {"head": head, "shoulder": shoulder, "hip": (hx, hy),
                "knee": knee, "ankle": ankle, "elbow": elbow,
                "wrist": wrist}

    def render(self, t: float) -> np.ndarray:
        frame = np.full((SIM_HEIGHT, SIM_WIDTH, 3), (200, 196, 190), np.uint8)
        # Wall / floor / skirting.
        cv2.rectangle(frame, (0, FLOOR_Y), (SIM_WIDTH, SIM_HEIGHT),
                      (150, 140, 130), -1)
        cv2.line(frame, (0, FLOOR_Y), (SIM_WIDTH, FLOOR_Y), (90, 85, 80), 4)
        # Squat rack: two uprights + top bar (dark grey, desaturated).
        for ux in (self.BAR_X - 190, self.BAR_X + 190):
            cv2.rectangle(frame, (ux - 12, 130), (ux + 12, FLOOR_Y),
                          (70, 70, 70), -1)
        cv2.rectangle(frame, (self.BAR_X - 202, 130),
                      (self.BAR_X + 202, 148), (70, 70, 70), -1)
        # Barbell plates stacked against the wall.
        for px, py, r in ((150, 560, 46), (150, 470, 38), (1130, 560, 46),
                          (1130, 470, 38)):
            cv2.circle(frame, (px, py), r, (60, 60, 60), -1)
            cv2.circle(frame, (px, py), r - 14, (90, 90, 90), -1)
        # Mirror on the wall.
        cv2.rectangle(frame, (950, 110), (1220, 330), (175, 180, 185), -1)
        cv2.rectangle(frame, (950, 110), (1220, 330), (120, 120, 120), 3)
        # Bench.
        cv2.rectangle(frame, (90, 520), (330, 540), (90, 85, 80), -1)
        cv2.line(frame, (120, 540), (120, FLOOR_Y), (90, 85, 80), 8)
        cv2.line(frame, (300, 540), (300, FLOOR_Y), (90, 85, 80), 8)

        # -- the figure -------------------------------------------------
        j = self.joints(t)
        hip, shoulder = j["hip"], j["shoulder"]
        knee, ankle = j["knee"], j["ankle"]
        elbow, wrist = j["elbow"], j["wrist"]
        head = j["head"]

        def line(a, b, colour, thick):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     colour, thick, cv2.LINE_AA)

        line(hip, shoulder, COLOUR_SHIRT, 34)       # torso
        line(shoulder, elbow, COLOUR_SHIRT, 20)     # upper arm
        line(elbow, wrist, COLOUR_SKIN, 16)         # forearm + hand
        line(hip, knee, COLOUR_TROUSERS, 26)        # thigh
        line(knee, ankle, COLOUR_TROUSERS, 22)      # shin
        cv2.circle(frame, (int(ankle[0]), int(ankle[1])), 12,
                   COLOUR_SHOES, -1)
        cv2.circle(frame, (int(head[0]), int(head[1])), self.HEAD_R,
                   COLOUR_SKIN, -1)                 # head
        # Joint markers: solid coloured discs the pixel estimator
        # finds. They sit exactly on the joints, so the angles the
        # estimator measures are the angles the figure was drawn with.
        for name, pos in (("shoulder", shoulder), ("hip", hip),
                          ("knee", knee), ("ankle", ankle),
                          ("elbow", elbow), ("wrist", wrist)):
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 13,
                       MARKER_COLOURS[name], -1)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 13,
                       (255, 255, 255), 2)
        return frame


class PixelJointEstimator(JointProvider):
    """Simulation-mode provider: estimates joint angles from the
    rendered pixels only. Each joint carries a solid coloured marker
    disc; the estimator finds each disc's centroid by colour and
    computes the angles from those centroids. It knows nothing about
    the choreography -- only the marker positions in this frame."""

    def _marker_centre(self, frame, bgr):
        lo = np.array([max(0, c - 35) for c in bgr], dtype=np.uint8)
        hi = np.array([min(255, c + 35) for c in bgr], dtype=np.uint8)
        mask = cv2.inRange(frame, lo, hi)
        # The white ring around each marker must not join the mask.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 80:
            return None
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    def sample(self, frame: np.ndarray) -> JointSample | None:
        pts = {}
        for name, colour in MARKER_COLOURS.items():
            centre = self._marker_centre(frame, colour)
            if centre is None:
                return None
            pts[name] = centre
        # Head: largest skin-coloured blob.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, np.array([5, 80, 120]),
                           np.array([30, 255, 255]))
        contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            head_cnt = max(contours, key=cv2.contourArea)
            hm = cv2.moments(head_cnt)
            if hm["m00"] != 0:
                pts["head"] = (hm["m10"] / hm["m00"], hm["m01"] / hm["m00"])
        if "head" not in pts:
            pts["head"] = pts["shoulder"]

        # Body box from the joint markers themselves: the trousers
        # are low-saturation, so a saturation mask of the figure
        # misses the legs entirely and the box ends at the hips.
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        bbox = (int(max(0, min(xs) - 30)), int(max(0, min(ys) - 30)),
                int(min(frame.shape[1], max(xs) + 30)),
                int(min(frame.shape[0], max(ys) + 30)))

        knee_angle = joint_angle(pts["hip"], pts["knee"], pts["ankle"])
        hip_angle = joint_angle(pts["shoulder"], pts["hip"], pts["knee"])
        elbow_angle = joint_angle(pts["shoulder"], pts["elbow"],
                                  pts["wrist"])
        dx = pts["shoulder"][0] - pts["hip"][0]
        dy = pts["shoulder"][1] - pts["hip"][1]
        torso_angle = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 90.0
        points = {k: (int(v[0]), int(v[1])) for k, v in pts.items()}
        return JointSample(knee_angle=knee_angle, hip_angle=hip_angle,
                           elbow_angle=elbow_angle, torso_angle=torso_angle,
                           bbox=bbox, points=points)


# ---------------------------------------------------------------------------
# Rep counter -- shared by both modes
# ---------------------------------------------------------------------------

class RepCounter:
    """Counts reps from JointSamples and grades form.

    The counter only ever sees JointSamples -- it has no access to the
    simulation choreography or to the raw video. A rep is one full
    down-up cycle of the tracked joint angle: the angle must cross the
    DOWN threshold, then return past the UP threshold. The smallest
    angle reached during the rep grades its depth.
    """

    def __init__(self, exercise="auto", log_path="events.log",
                 summary_path="output_summary.txt",
                 min_rep_seconds=0.6):
        self.exercise = exercise
        self.log_path = log_path
        self.summary_path = summary_path
        # A real squat / push-up / curl takes well over half a second
        # down and up. Anything faster is landmark noise (the tracker
        # re-locking at the start of a clip or after a loop point, or
        # a knee jumping behind a rack upright) -- never a rep.
        self.min_rep_seconds = min_rep_seconds
        self.angles = deque(maxlen=SMOOTH_WINDOW)
        self.phase = "UP"           # UP (top) / DOWN (bottom)
        self.min_angle_in_rep = None
        self.min_torso_in_rep = None
        self.rep_start_t = None
        # Arming: DOWN can only trigger after the tracked angle has
        # sat above the UP threshold for 10 straight frames (a
        # confirmed standing / extended baseline). Stops the noisy
        # first frames of a clip from opening a phantom rep.
        self._up_streak = 0
        self._armed = False
        self.rep_durations = []
        self.counters = {
            "frames": 0, "pose_frames": 0, "reps": 0,
            "good_depth": 0, "shallow": 0, "too_shallow": 0,
            "lean_warnings": 0, "loops": 0, "no_pose_spells": 0,
            "rejected_fast": 0,
        }
        self.exercise_detected = None if exercise == "auto" else exercise
        self._observe = {"knee": [], "elbow": [], "torso": []} \
            if exercise == "auto" else None
        self._observe_until = None
        self._no_pose = False
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._summary_file = open(summary_path, "w", encoding="utf-8")
        self._log("Workout rep counter started "
                  f"(exercise={exercise}).")

    # -- logging -----------------------------------------------------------
    def _log(self, message: str, t: float | None = None):
        stamp = f"{t:6.1f}s" if t is not None else "      "
        self._log_file.write(f"[{stamp}] {message}\n")
        self._log_file.flush()

    # -- exercise selection --------------------------------------------------
    def _tracked_angle(self, sample: JointSample):
        """(angle, joint-name) the counter tracks for the chosen
        exercise, or (None, None) when its landmarks are missing."""
        cfg = EXERCISES[self.exercise_detected]
        joint = cfg["joint"]
        angle = getattr(sample, f"{joint}_angle")
        if angle is not None:
            return angle, joint
        fb = cfg.get("fallback_joint")
        if fb:
            angle = getattr(sample, f"{fb}_angle")
            if angle is not None:
                return angle, fb
        return None, None

    def _thresholds(self, joint: str):
        cfg = EXERCISES[self.exercise_detected]
        if joint == cfg["joint"]:
            return cfg["down"], cfg["up"]
        return cfg["down_fallback"], cfg["up_fallback"]

    def _grade(self, min_angle: float) -> str:
        cfg = EXERCISES[self.exercise_detected]
        if min_angle <= cfg["depth_good"]:
            return "GOOD DEPTH"
        if min_angle <= cfg["depth_ok"]:
            return "SHALLOW"
        return "TOO SHALLOW"

    def _auto_detect(self, sample: JointSample, t: float):
        """Pick the exercise from the first seconds of movement."""
        if self._observe_until is None:
            self._observe_until = t + AUTO_OBSERVE_S
        if sample.knee_angle is not None:
            self._observe["knee"].append(sample.knee_angle)
        if sample.elbow_angle is not None:
            self._observe["elbow"].append(sample.elbow_angle)
        self._observe["torso"].append(sample.torso_angle)
        if t < self._observe_until:
            return
        knee_range = (max(self._observe["knee"]) - min(self._observe["knee"])
                      if len(self._observe["knee"]) > 5 else 0.0)
        elbow_range = (max(self._observe["elbow"]) - min(self._observe["elbow"])
                       if len(self._observe["elbow"]) > 5 else 0.0)
        torso_med = sorted(self._observe["torso"])[len(self._observe["torso"]) // 2]
        if torso_med > 45.0:
            picked = "pushup"
        elif knee_range >= elbow_range:
            picked = "squat"
        else:
            picked = "curl"
        self.exercise_detected = picked
        self._observe = None
        self._log(f"Auto-detected exercise: {EXERCISES[picked]['label']} "
                  f"(torso {torso_med:.0f} deg, knee range {knee_range:.0f} "
                  f"deg, elbow range {elbow_range:.0f} deg)", t)

    # -- main update ---------------------------------------------------------
    def update(self, sample: JointSample | None, t: float):
        """Feed one frame's sample; returns (tracked_angle, phase)."""
        self.counters["frames"] += 1
        if sample is None:
            if not self._no_pose:
                self._no_pose = True
                self.counters["no_pose_spells"] += 1
            return None, "NO PERSON"
        self._no_pose = False
        self.counters["pose_frames"] += 1

        if self.exercise_detected is None:
            self._auto_detect(sample, t)
            if self.exercise_detected is None:
                return None, "DETECTING"
        angle, joint = self._tracked_angle(sample)
        if angle is None:
            return None, "NO JOINTS"
        self.angles.append(angle)
        smooth = sorted(self.angles)[len(self.angles) // 2]
        down_th, up_th = self._thresholds(joint)

        # Arming: count consecutive frames above the UP threshold.
        if smooth >= up_th:
            self._up_streak += 1
        else:
            self._up_streak = 0
        if self._up_streak >= 10:
            self._armed = True

        if self.phase == "UP":
            if smooth <= down_th and self._armed:
                self.phase = "DOWN"
                self.min_angle_in_rep = smooth
                self.min_torso_in_rep = sample.torso_angle
                self.rep_start_t = t
        else:  # DOWN
            if smooth < (self.min_angle_in_rep or smooth):
                self.min_angle_in_rep = smooth
            if sample.torso_angle < (self.min_torso_in_rep or 999):
                self.min_torso_in_rep = sample.torso_angle
            if smooth >= up_th:
                self.phase = "UP"
                self._up_streak = 0
                self._armed = False
                self._finish_rep(t)
        return smooth, self.phase

    def _finish_rep(self, t: float):
        cfg = EXERCISES[self.exercise_detected]
        duration = t - (self.rep_start_t or t)
        if duration < self.min_rep_seconds:
            # Too fast to be a real rep: landmark noise, not exercise.
            self.counters["rejected_fast"] += 1
            self._log(f"Ignored fast cycle ({duration:.1f} s, "
                      f"min angle {self.min_angle_in_rep:.0f} deg) -- "
                      f"landmark noise, not a rep", t)
            self.min_angle_in_rep = None
            self.min_torso_in_rep = None
            self.rep_start_t = None
            return
        self.counters["reps"] += 1
        verdict = self._grade(self.min_angle_in_rep)
        key = {"GOOD DEPTH": "good_depth", "SHALLOW": "shallow",
               "TOO SHALLOW": "too_shallow"}[verdict]
        self.counters[key] += 1
        self.rep_durations.append(duration)
        tip = ""
        if (self.exercise_detected == "squat"
                and self.min_torso_in_rep is not None
                and self.min_torso_in_rep > 35.0):
            self.counters["lean_warnings"] += 1
            tip = " -- KEEP CHEST UP (torso leaned " \
                  f"{self.min_torso_in_rep:.0f} deg)"
        self._log(f"REP {self.counters['reps']}: {cfg['label']} "
                  f"min angle {self.min_angle_in_rep:.0f} deg -- "
                  f"{verdict} ({duration:.1f} s){tip}", t)
        self.min_angle_in_rep = None
        self.min_torso_in_rep = None
        self.rep_start_t = None

    def on_loop(self, t: float):
        """A looped video restarts mid-rep: reset the phase so the cut
        can never complete a phantom rep."""
        self.counters["loops"] += 1
        if self.phase == "DOWN":
            self._log("Loop point -- in-flight rep discarded "
                      "(phase reset to UP)", t)
        self.phase = "UP"
        self.angles.clear()
        self._up_streak = 0
        self._armed = False
        self.min_angle_in_rep = None
        self.min_torso_in_rep = None
        self.rep_start_t = None

    def close(self, duration_s: float):
        self._log("Workout rep counter stopped.")
        self._log_file.close()
        c = self.counters
        avg_depth = None
        if self.rep_durations:
            avg_rep = sum(self.rep_durations) / len(self.rep_durations)
        else:
            avg_rep = None
        lines = [
            "Workout Rep Counter -- end-of-run summary",
            f"Exercise: {EXERCISES[self.exercise_detected]['label'] if self.exercise_detected else 'not detected'}",
            f"Duration: {duration_s:.1f} s",
            f"Frames processed: {c['frames']}",
            f"Frames with a pose: {c['pose_frames']} "
            f"({100.0 * c['pose_frames'] / max(1, c['frames']):.1f}%)",
            f"REPS: {c['reps']}",
            f"  Good depth: {c['good_depth']}",
            f"  Shallow: {c['shallow']}",
            f"  Too shallow: {c['too_shallow']}",
            f"  Torso-lean warnings: {c['lean_warnings']}",
            f"  Mean rep duration: "
            f"{avg_rep:.1f} s" if avg_rep is not None else "  Mean rep duration: n/a",
            f"Fast cycles rejected as noise: {c['rejected_fast']}",
            f"Video loops: {c['loops']}",
            f"Pose-lost spells: {c['no_pose_spells']}",
        ]
        self._summary_file.write("\n".join(lines) + "\n")
        self._summary_file.close()
        return lines


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame, sample, angle, phase, counter, t, fps):
    """Draws the skeleton, body box and depth gauge on the frame.

    The status panel is NOT drawn here: it lives in its own strip
    above the video (see draw_panel / the compositing in main), so
    it can never cover the person's head -- on the squat footage
    the head sits as high as y=23 in a 720p frame, i.e. inside any
    top overlay bar.
    """
    h, w = frame.shape[:2]
    exercise = counter.exercise_detected
    label = EXERCISES[exercise]["label"] if exercise else "DETECTING..."
    reps = counter.counters["reps"]
    phase_colour = COLOUR_UP if phase == "UP" else (
        COLOUR_DOWN if phase == "DOWN" else COLOUR_NOPOSE)
    if sample is not None:
        pts = sample.points
        for a, b in (("head", "shoulder"), ("shoulder", "hip"),
                     ("hip", "knee"), ("knee", "ankle"),
                     ("shoulder", "elbow"), ("elbow", "wrist")):
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], COLOUR_SKELETON, 3, cv2.LINE_AA)
        for name, pt in pts.items():
            cv2.circle(frame, pt, 6 if name == "head" else 5,
                       COLOUR_SKELETON, -1)
        x1, y1, x2, y2 = sample.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), phase_colour, 2)
    # Angle gauge (bottom): a bar that fills as the joint closes.
    if angle is not None and exercise:
        cfg = EXERCISES[exercise]
        down_th, up_th = counter._thresholds(
            cfg["joint"] if getattr(sample, f"{cfg['joint']}_angle", None) is not None
            else (cfg.get("fallback_joint") or cfg["joint"]))
        frac = max(0.0, min(1.0, (up_th - angle) / max(1.0, up_th - down_th)))
        gx, gy, gw, gh = 20, h - 44, 360, 22
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (60, 60, 60), -1)
        cv2.rectangle(frame, (gx, gy), (gx + int(gw * frac), gy + gh),
                      phase_colour, -1)
        cv2.putText(frame, "DEPTH", (gx + gw + 12, gy + 17),
                    FONT, 0.6, COLOUR_TEXT, 2, cv2.LINE_AA)
    return frame


PANEL_HEIGHT = 108   # status strip above the video (see draw_overlay)


def draw_panel(width, angle, phase, counter, t, fps):
    """The status strip: exercise, rep count, angle, phase, totals."""
    strip = np.zeros((PANEL_HEIGHT, width, 3), dtype=np.uint8)
    exercise = counter.exercise_detected
    label = EXERCISES[exercise]["label"] if exercise else "DETECTING..."
    reps = counter.counters["reps"]
    phase_colour = COLOUR_UP if phase == "UP" else (
        COLOUR_DOWN if phase == "DOWN" else COLOUR_NOPOSE)
    cv2.putText(strip, f"AI GYM TRAINER  --  {label}",
                (20, 34), FONT, 0.95, COLOUR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(strip, f"REPS: {reps}",
                (20, 76), FONT, 1.1, COLOUR_GOOD, 2, cv2.LINE_AA)
    angle_txt = f"{angle:.0f} deg" if angle is not None else "-- deg"
    cv2.putText(strip, f"angle {angle_txt}   phase {phase}",
                (260, 76), FONT, 0.7, phase_colour, 2, cv2.LINE_AA)
    cv2.putText(strip, f"t={t:5.1f}s   fps={fps:4.1f}   "
                       f"good {counter.counters['good_depth']}  "
                       f"shallow {counter.counters['shallow']}  "
                       f"too shallow {counter.counters['too_shallow']}",
                (20, 100), FONT, 0.55, COLOUR_TEXT, 2, cv2.LINE_AA)
    return strip


# ---------------------------------------------------------------------------
# Source selection (Windows CMD file chooser, per the standing rule)
# ---------------------------------------------------------------------------

def pick_video_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Workout rep counter: pose-based reps + form feedback.")
    parser.add_argument("--source", default=None,
                        help="video file, webcam index, or 'pick' "
                             "(default: opens the file chooser)")
    parser.add_argument("--simulate", action="store_true",
                        help="rendered gym simulation instead of a camera")
    parser.add_argument("--exercise", default="auto",
                        choices=["auto", "squat", "pushup", "curl"],
                        help="exercise to count (default: auto-detect)")
    parser.add_argument("--loop", action="store_true",
                        help="loop a short video until --max-seconds")
    parser.add_argument("--output", default="output.mp4",
                        help="annotated video written here")
    parser.add_argument("--log", default="events.log",
                        help="timestamped event log")
    parser.add_argument("--summary", default="output_summary.txt",
                        help="end-of-run summary text")
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="stop after this many seconds")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window")
    args = parser.parse_args(argv)

    if args.simulate:
        provider: JointProvider = PixelJointEstimator()
        scene = SimScene()
        fps = SIM_FPS
        total_frames = int(args.max_seconds * SIM_FPS)
        cap = None
        width, height = SIM_WIDTH, SIM_HEIGHT
    else:
        source = args.source
        if source is None or str(source).lower() == "pick":
            picked = pick_video_file()
            if not picked:
                print("No video file chosen. Exiting.")
                return 1
            source = picked
        if str(source).isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open video source: {source}")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        provider = MediaPipePoseProvider()
        scene = None
        total_frames = None

    counter = RepCounter(exercise=args.exercise, log_path=args.log,
                         summary_path=args.summary)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"Could not open video writer: {args.output}")
        return 1

    frame_index = 0
    t_start = time.time()
    try:
        while True:
            if scene is not None:
                if frame_index >= total_frames:
                    break
                t = frame_index / SIM_FPS
                frame = scene.render(t)
            else:
                ok, frame = cap.read()
                if not ok:
                    if args.loop and isinstance(source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            break
                        counter.on_loop(frame_index / fps)
                    else:
                        break
                t = frame_index / fps
                if args.max_seconds and t >= args.max_seconds:
                    break
            if isinstance(provider, MediaPipePoseProvider):
                sample = provider.sample(frame, t_ms=t * 1000.0)
            else:
                sample = provider.sample(frame)
            angle, phase = counter.update(sample, t)
            elapsed = time.time() - t_start
            run_fps = (frame_index + 1) / max(elapsed, 1e-6)
            frame = draw_overlay(frame, sample, angle, phase,
                                 counter, t, run_fps)
            # Composite: status strip on top, annotated video below
            # it (scaled to fit). The strip never covers the person.
            panel = draw_panel(width, angle, phase, counter, t, run_fps)
            video_area = cv2.resize(frame, (width, height - PANEL_HEIGHT))
            canvas = np.vstack([panel, video_area])
            writer.write(canvas)
            if not args.no_display:
                cv2.imshow("AI Gym Trainer -- Workout Rep Counter  (Q to quit)", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            frame_index += 1
    finally:
        writer.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    duration = counter.counters["frames"] / fps
    lines = counter.close(duration)
    print()
    print("--- " + lines[0] + " ---")
    for line in lines[1:]:
        print(line)
    print(f"Annotated video:       {args.output}")
    print(f"Event log:             {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
