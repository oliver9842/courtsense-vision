import os
import tempfile
import math
import traceback
import logging
import statistics

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Logging — compatible with both Flask dev server and gunicorn
# ---------------------------------------------------------------------------

gunicorn_logger = logging.getLogger("gunicorn.error")

app = Flask(__name__)

if gunicorn_logger.handlers:
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

mp_pose = mp.solutions.pose

MAX_UPLOAD_GB = 4
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_GB * 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Shot detection tuning
POSE_SAMPLE_INTERVAL = 3       # run pose every Nth frame (speed vs accuracy)
POST_RELEASE_COOLDOWN = 45     # frames to wait after a release before tracking next shot
MIN_SHOT_FRAMES = 8            # ignore shots shorter than this (noise filter)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Basketball (orange) detection
# ---------------------------------------------------------------------------

ORANGE_LOWER = np.array([5, 120, 120], dtype=np.uint8)
ORANGE_UPPER = np.array([25, 255, 255], dtype=np.uint8)


def detect_ball_center(frame: np.ndarray):
    """Return (cx, cy) of the largest orange blob, or None."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def estimate_rim_entry_angle(trajectory: list) -> float | None:
    if len(trajectory) < 4:
        return None
    pts = trajectory[-6:]
    if len(pts) < 2:
        return None
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return None
    return round(math.degrees(math.atan2(-dy, dx)), 1)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def get_y(landmarks, idx, h):
    return landmarks[idx].y * h


def classify_phase(landmarks, h, w) -> str:
    r_wrist   = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value,    h)
    r_elbow   = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value,    h)
    r_shoulder = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, h)
    r_hip     = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value,      h)

    torso = abs(r_shoulder - r_hip)
    if torso == 0:
        return "idle"

    if r_wrist > r_hip:
        return "dip"
    if abs(r_wrist - r_shoulder) < torso * 0.4 and r_elbow >= r_shoulder:
        return "load"
    if r_wrist < r_shoulder or r_elbow < r_shoulder:
        return "release"
    return "idle"


# ---------------------------------------------------------------------------
# Feedback helpers (shared by both endpoints)
# ---------------------------------------------------------------------------

def shot_feedback(release_speed_sec, rim_entry_angle_deg, phase_seq, ball_detected):
    tips = []

    if not ball_detected:
        tips.append(
            "No basketball was detected. Ensure the ball is clearly visible and well-lit."
        )

    if "dip" not in phase_seq:
        tips.append(
            "No pre-shot dip detected. A controlled dip loads energy into your legs "
            "and helps generate upward power."
        )
    elif "load" not in phase_seq:
        tips.append(
            "Dip detected but the load phase was missing. Focus on bringing the ball "
            "up to your shooting pocket before releasing."
        )

    if release_speed_sec is not None:
        if release_speed_sec < 0.3:
            tips.append(
                f"Very quick release ({release_speed_sec:.2f}s). Make sure you're not "
                "rushing and skipping the set position."
            )
        elif release_speed_sec > 1.5:
            tips.append(
                f"Shot motion took {release_speed_sec:.2f}s — work on a quicker release "
                "to make it harder to contest."
            )
        else:
            tips.append(
                f"Good release tempo at {release_speed_sec:.2f}s."
            )

    if rim_entry_angle_deg is not None:
        if rim_entry_angle_deg > 45:
            tips.append(
                f"Excellent arc at {rim_entry_angle_deg:.1f}° — high arc maximises the "
                "effective rim target size."
            )
        elif rim_entry_angle_deg > 30:
            tips.append(
                f"Decent arc at {rim_entry_angle_deg:.1f}°. Aim for 45°+ for a larger margin."
            )
        else:
            tips.append(
                f"Low arc at {rim_entry_angle_deg:.1f}°. Practice a higher follow-through."
            )

    if not tips:
        tips.append(
            "Shot mechanics look solid. Focus on consistency under game-speed pressure."
        )

    return tips


# ---------------------------------------------------------------------------
# Single-shot analysis (used by /analyze)
# ---------------------------------------------------------------------------

def analyze_single_shot(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps              = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_processed = 0
    phase_seq        = []
    last_phase       = "idle"
    dip_frame        = None
    release_frame    = None
    ball_trajectory  = []
    ball_count       = 0

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_processed += 1
            h, w = frame.shape[:2]

            bp = detect_ball_center(frame)
            if bp:
                ball_trajectory.append(bp)
                ball_count += 1

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                phase = classify_phase(results.pose_landmarks.landmark, h, w)
                if phase != last_phase:
                    phase_seq.append(phase)
                    last_phase = phase
                    if phase == "dip" and dip_frame is None:
                        dip_frame = frames_processed
                    if phase == "release" and dip_frame and not release_frame:
                        release_frame = frames_processed

    cap.release()
    app.logger.info("Single-shot: %d frames at %.1f fps", frames_processed, fps)

    release_speed_sec   = round((release_frame - dip_frame) / fps, 3) if dip_frame and release_frame else None
    rim_entry_angle_deg = estimate_rim_entry_angle(ball_trajectory)
    ball_detected       = ball_count > 0
    shot_detected       = any(p in phase_seq for p in ("dip", "load", "release"))
    feedback            = shot_feedback(release_speed_sec, rim_entry_angle_deg, phase_seq, ball_detected)

    return {
        "shot_detected":       shot_detected,
        "frames_processed":    frames_processed,
        "release_speed_sec":   release_speed_sec,
        "rim_entry_angle_deg": rim_entry_angle_deg,
        "feedback":            feedback,
    }


# ---------------------------------------------------------------------------
# Full-film analysis (used by /analyze-film)
# ---------------------------------------------------------------------------

# Shot state machine states
_IDLE     = "idle"
_DIP      = "dip"
_LOAD     = "load"
_RELEASE  = "release"


def _compute_consistency(values: list) -> float | None:
    """
    Return a 0-100 consistency score based on the coefficient of variation.
    100 = perfectly consistent, 0 = wildly inconsistent.
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    mean = statistics.mean(clean)
    if mean == 0:
        return 100.0
    stdev = statistics.stdev(clean)
    cv    = stdev / mean          # coefficient of variation
    score = max(0.0, 100.0 - cv * 100.0)
    return round(score, 1)


def analyze_full_film(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps              = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_processed = 0

    app.logger.info(
        "Film analysis: ~%d frames (%.1f min) at %.1f fps",
        total_frames, total_frames / fps / 60, fps,
    )

    # Per-frame tracking
    shot_state      = _IDLE
    cooldown        = 0

    # Current shot window
    dip_frame       = None
    release_frame   = None
    window_ball     = []      # ball positions within current shot window
    window_phases   = []

    completed_shots = []      # list of dicts, one per detected shot

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,           # fastest model for long videos
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_processed += 1

            if cooldown > 0:
                cooldown -= 1
                continue

            h, w = frame.shape[:2]

            # Ball tracking every frame (cheap)
            bp = detect_ball_center(frame)
            if bp and shot_state != _IDLE:
                window_ball.append(bp)

            # Pose only every Nth frame
            if frames_processed % POSE_SAMPLE_INTERVAL != 0:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_landmarks:
                continue

            phase = classify_phase(results.pose_landmarks.landmark, h, w)

            # ---- state machine ----
            if shot_state == _IDLE:
                if phase == "dip":
                    shot_state  = _DIP
                    dip_frame   = frames_processed
                    window_ball = []
                    window_phases = ["dip"]

            elif shot_state == _DIP:
                if phase == "load":
                    shot_state = _LOAD
                    window_phases.append("load")
                elif phase == "idle" and (frames_processed - dip_frame) > MIN_SHOT_FRAMES * 3:
                    # dip faded without load — abort
                    shot_state = _IDLE
                    dip_frame  = None

            elif shot_state == _LOAD:
                if phase == "release":
                    shot_state    = _RELEASE
                    release_frame = frames_processed
                    window_phases.append("release")

            elif shot_state == _RELEASE:
                # One extra pose sample past release to collect final ball trajectory
                if (frames_processed - release_frame) >= POSE_SAMPLE_INTERVAL * 2:
                    # --- record the shot ---
                    span = frames_processed - dip_frame
                    if span >= MIN_SHOT_FRAMES:
                        timestamp_sec       = round(dip_frame / fps, 2)
                        release_speed_sec   = round((release_frame - dip_frame) / fps, 3)
                        rim_entry_angle_deg = estimate_rim_entry_angle(window_ball)

                        completed_shots.append({
                            "shot_number":         len(completed_shots) + 1,
                            "timestamp_sec":       timestamp_sec,
                            "release_speed_sec":   release_speed_sec,
                            "rim_entry_angle_deg": rim_entry_angle_deg,
                            "ball_detected":       len(window_ball) > 0,
                            "phases_detected":     list(window_phases),
                            "feedback":            shot_feedback(
                                release_speed_sec, rim_entry_angle_deg,
                                window_phases, len(window_ball) > 0,
                            ),
                        })

                    shot_state    = _IDLE
                    dip_frame     = None
                    release_frame = None
                    window_ball   = []
                    window_phases = []
                    cooldown      = POST_RELEASE_COOLDOWN

    cap.release()
    app.logger.info(
        "Film analysis done: %d frames processed, %d shots found",
        frames_processed, len(completed_shots),
    )

    # ---- aggregate summary ----
    total_shots = len(completed_shots)

    speeds  = [s["release_speed_sec"]   for s in completed_shots if s["release_speed_sec"]   is not None]
    angles  = [s["rim_entry_angle_deg"] for s in completed_shots if s["rim_entry_angle_deg"] is not None]

    avg_release_speed_sec   = round(statistics.mean(speeds), 3) if speeds else None
    avg_rim_entry_angle_deg = round(statistics.mean(angles), 1) if angles else None

    # Best shot = highest rim entry angle (best arc); fall back to lowest release time
    best_shot = None
    if completed_shots:
        shots_with_angle = [s for s in completed_shots if s["rim_entry_angle_deg"] is not None]
        if shots_with_angle:
            best_shot = max(shots_with_angle, key=lambda s: s["rim_entry_angle_deg"])["shot_number"]
        else:
            shots_with_speed = [s for s in completed_shots if s["release_speed_sec"] is not None]
            if shots_with_speed:
                best_shot = min(shots_with_speed, key=lambda s: s["release_speed_sec"])["shot_number"]

    # Consistency: average of speed-consistency and angle-consistency
    speed_consistency = _compute_consistency(speeds)
    angle_consistency = _compute_consistency(angles)
    consistency_scores = [c for c in [speed_consistency, angle_consistency] if c is not None]
    consistency_score  = round(statistics.mean(consistency_scores), 1) if consistency_scores else None

    return {
        "frames_processed":         frames_processed,
        "total_shots_detected":     total_shots,
        "avg_release_speed_sec":    avg_release_speed_sec,
        "avg_rim_entry_angle_deg":  avg_rim_entry_angle_deg,
        "best_shot":                best_shot,
        "consistency_score":        consistency_score,
        "shots":                    completed_shots,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return "CourtSense Vision Engine is running"


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Send a multipart form with key 'video'."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        app.logger.info("Analyzing uploaded clip: %s", tmp.name)
        result = analyze_single_shot(tmp.name)
        return jsonify(result), 200
    except Exception as exc:
        app.logger.error("Analysis failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/analyze-film", methods=["POST"])
def analyze_film():
    """
    Full game-film analysis endpoint.

    Accepts large video uploads (up to 4 GB). Scans the entire video,
    auto-detects every shot attempt, and returns a session summary plus
    per-shot metrics with timestamps.

    Request: multipart/form-data with a 'video' field.

    Response JSON:
    {
        "frames_processed":        int,
        "total_shots_detected":    int,
        "avg_release_speed_sec":   float | null,
        "avg_rim_entry_angle_deg": float | null,
        "best_shot":               int | null,   // shot_number of best shot
        "consistency_score":       float | null, // 0-100
        "shots": [
            {
                "shot_number":         int,
                "timestamp_sec":       float,
                "release_speed_sec":   float | null,
                "rim_entry_angle_deg": float | null,
                "ball_detected":       bool,
                "phases_detected":     [str],
                "feedback":            [str]
            },
            ...
        ]
    }
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Send a multipart form with key 'video'."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        app.logger.info("Receiving full-film upload...")
        file.save(tmp.name)
        tmp.close()
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        app.logger.info("Film saved: %.1f MB — starting analysis", size_mb)
        result = analyze_full_film(tmp.name)
        return jsonify(result), 200
    except Exception as exc:
        app.logger.error("Film analysis failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point — dev only (gunicorn imports app directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting dev server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
