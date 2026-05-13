import os
import tempfile
import math
import logging
import statistics
import threading
import uuid
import time
import json

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

gunicorn_logger = logging.getLogger("gunicorn.error")

app = Flask(__name__)
CORS(app, origins="*")
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

# Per-endpoint upload size limits
MAX_SINGLE_SHOT_MB = 50          # /analyze  — short clip, must fit in memory
MAX_FILM_MB        = 4 * 1024    # /analyze-film — full game film, async processing

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

POSE_SAMPLE_INTERVAL  = 3    # run pose every Nth frame
POST_RELEASE_COOLDOWN = 45   # frames to skip after a release
MIN_SHOT_FRAMES       = 8    # ignore shots shorter than this

JOB_DIR = os.path.join(tempfile.gettempdir(), "courtsense_jobs")
JOB_TTL_SECONDS = 3600       # keep job files for 1 hour

os.makedirs(JOB_DIR, exist_ok=True)

_jobs_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Job store (file-based so it works across gunicorn workers)
# ---------------------------------------------------------------------------

def _job_path(job_id: str) -> str:
    return os.path.join(JOB_DIR, f"{job_id}.json")


def _write_job(job_id: str, data: dict) -> None:
    with _jobs_lock:
        with open(_job_path(job_id), "w") as f:
            json.dump(data, f)


def _read_job(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with _jobs_lock:
        with open(path) as f:
            return json.load(f)


def _cleanup_old_jobs() -> None:
    """Remove job files older than JOB_TTL_SECONDS (runs in background)."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for fname in os.listdir(JOB_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(JOB_DIR, fname)
        try:
            if os.path.getmtime(fpath) < cutoff:
                os.unlink(fpath)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Basketball detection
# ---------------------------------------------------------------------------

ORANGE_LOWER = np.array([5, 120, 120], dtype=np.uint8)
ORANGE_UPPER = np.array([25, 255, 255], dtype=np.uint8)


def detect_ball_center(frame: np.ndarray):
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
    r_wrist    = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value,    h)
    r_elbow    = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value,    h)
    r_shoulder = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, h)
    r_hip      = get_y(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value,      h)
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
# Feedback
# ---------------------------------------------------------------------------

def shot_feedback(release_speed_sec, rim_entry_angle_deg, phase_seq, ball_detected):
    tips = []
    if not ball_detected:
        tips.append("No basketball detected. Ensure the ball is clearly visible and well-lit.")
    if "dip" not in phase_seq:
        tips.append("No pre-shot dip detected. A controlled dip loads power into your legs.")
    elif "load" not in phase_seq:
        tips.append("Dip detected but load phase missing. Bring the ball to your shooting pocket first.")
    if release_speed_sec is not None:
        if release_speed_sec < 0.3:
            tips.append(f"Very quick release ({release_speed_sec:.2f}s). Don't skip the set position.")
        elif release_speed_sec > 1.5:
            tips.append(f"Slow release ({release_speed_sec:.2f}s). Work on a quicker motion.")
        else:
            tips.append(f"Good release tempo at {release_speed_sec:.2f}s.")
    if rim_entry_angle_deg is not None:
        if rim_entry_angle_deg > 45:
            tips.append(f"Excellent arc at {rim_entry_angle_deg:.1f}° — maximises effective rim size.")
        elif rim_entry_angle_deg > 30:
            tips.append(f"Decent arc at {rim_entry_angle_deg:.1f}°. Aim for 45°+ for more margin.")
        else:
            tips.append(f"Low arc at {rim_entry_angle_deg:.1f}°. Practice a higher follow-through.")
    if not tips:
        tips.append("Shot mechanics look solid. Focus on consistency under pressure.")
    return tips


# ---------------------------------------------------------------------------
# Single-shot analysis (/analyze)
# ---------------------------------------------------------------------------

def analyze_single_shot(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    frames_processed = 0
    phase_seq = []
    last_phase = "idle"
    dip_frame = release_frame = None
    ball_trajectory = []
    ball_count = 0

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=0, enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frames_processed += 1
            if frame_idx % 3 != 0:
                continue
            h, w = frame.shape[:2]
            bp = detect_ball_center(frame)
            if bp:
                ball_trajectory.append(bp)
                ball_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    return {
        "shot_detected":       shot_detected,
        "frames_processed":    frames_processed,
        "release_speed_sec":   release_speed_sec,
        "rim_entry_angle_deg": rim_entry_angle_deg,
        "feedback":            shot_feedback(release_speed_sec, rim_entry_angle_deg, phase_seq, ball_detected),
    }


# ---------------------------------------------------------------------------
# Full-film analysis core (runs in background thread)
# ---------------------------------------------------------------------------

def _compute_consistency(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    mean = statistics.mean(clean)
    if mean == 0:
        return 100.0
    cv = statistics.stdev(clean) / mean
    return round(max(0.0, 100.0 - cv * 100.0), 1)


def _analyze_film_worker(job_id: str, video_path: str) -> None:
    """Runs in a background thread. Writes progress + final result to job file."""
    try:
        _write_job(job_id, {
            "status": "processing",
            "progress": {"frames_processed": 0, "shots_found": 0},
            "created_at": time.time(),
        })

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed = 0

        app.logger.info(
            "[job:%s] Film: ~%d frames (%.1f min) at %.1f fps",
            job_id, total_frames, total_frames / fps / 60, fps,
        )

        shot_state    = "idle"
        cooldown      = 0
        dip_frame     = release_frame = None
        window_ball   = []
        window_phases = []
        completed_shots = []

        PROGRESS_INTERVAL = 500   # update job file every N frames

        with mp_pose.Pose(
            static_image_mode=False, model_complexity=0, enable_segmentation=False,
            min_detection_confidence=0.45, min_tracking_confidence=0.45,
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

                bp = detect_ball_center(frame)
                if bp and shot_state != "idle":
                    window_ball.append(bp)

                if frames_processed % POSE_SAMPLE_INTERVAL != 0:
                    if frames_processed % PROGRESS_INTERVAL == 0:
                        _write_job(job_id, {
                            "status": "processing",
                            "progress": {
                                "frames_processed": frames_processed,
                                "total_frames": total_frames,
                                "shots_found": len(completed_shots),
                                "percent_complete": round(frames_processed / max(total_frames, 1) * 100, 1),
                            },
                            "created_at": time.time(),
                        })
                    continue

                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if not results.pose_landmarks:
                    continue

                phase = classify_phase(results.pose_landmarks.landmark, h, w)

                if shot_state == "idle":
                    if phase == "dip":
                        shot_state    = "dip"
                        dip_frame     = frames_processed
                        window_ball   = []
                        window_phases = ["dip"]

                elif shot_state == "dip":
                    if phase == "load":
                        shot_state = "load"
                        window_phases.append("load")
                    elif phase == "idle" and (frames_processed - dip_frame) > MIN_SHOT_FRAMES * 3:
                        shot_state = "idle"
                        dip_frame  = None

                elif shot_state == "load":
                    if phase == "release":
                        shot_state    = "release"
                        release_frame = frames_processed
                        window_phases.append("release")

                elif shot_state == "release":
                    if (frames_processed - release_frame) >= POSE_SAMPLE_INTERVAL * 2:
                        span = frames_processed - dip_frame
                        if span >= MIN_SHOT_FRAMES:
                            ts  = round(dip_frame / fps, 2)
                            spd = round((release_frame - dip_frame) / fps, 3)
                            ang = estimate_rim_entry_angle(window_ball)
                            completed_shots.append({
                                "shot_number":         len(completed_shots) + 1,
                                "timestamp_sec":       ts,
                                "release_speed_sec":   spd,
                                "rim_entry_angle_deg": ang,
                                "ball_detected":       len(window_ball) > 0,
                                "phases_detected":     list(window_phases),
                                "feedback":            shot_feedback(spd, ang, window_phases, len(window_ball) > 0),
                            })
                        shot_state    = "idle"
                        dip_frame     = release_frame = None
                        window_ball   = []
                        window_phases = []
                        cooldown      = POST_RELEASE_COOLDOWN

        cap.release()

        # ---- aggregate ----
        speeds = [s["release_speed_sec"]   for s in completed_shots if s["release_speed_sec"]   is not None]
        angles = [s["rim_entry_angle_deg"] for s in completed_shots if s["rim_entry_angle_deg"] is not None]

        avg_speed = round(statistics.mean(speeds), 3) if speeds else None
        avg_angle = round(statistics.mean(angles), 1) if angles else None

        best_shot = None
        if completed_shots:
            with_angle = [s for s in completed_shots if s["rim_entry_angle_deg"] is not None]
            if with_angle:
                best_shot = max(with_angle, key=lambda s: s["rim_entry_angle_deg"])["shot_number"]
            else:
                with_speed = [s for s in completed_shots if s["release_speed_sec"] is not None]
                if with_speed:
                    best_shot = min(with_speed, key=lambda s: s["release_speed_sec"])["shot_number"]

        c_scores = [c for c in [_compute_consistency(speeds), _compute_consistency(angles)] if c is not None]
        consistency = round(statistics.mean(c_scores), 1) if c_scores else None

        result = {
            "frames_processed":         frames_processed,
            "total_shots_detected":     len(completed_shots),
            "avg_release_speed_sec":    avg_speed,
            "avg_rim_entry_angle_deg":  avg_angle,
            "best_shot":                best_shot,
            "consistency_score":        consistency,
            "shots":                    completed_shots,
        }

        _write_job(job_id, {
            "status":     "done",
            "result":     result,
            "created_at": time.time(),
        })

        app.logger.info("[job:%s] Done — %d shots in %d frames", job_id, len(completed_shots), frames_processed)

    except Exception as exc:
        app.logger.error("[job:%s] Failed: %s", job_id, exc, exc_info=True)
        _write_job(job_id, {
            "status":     "failed",
            "error":      str(exc),
            "created_at": time.time(),
        })
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass
        threading.Thread(target=_cleanup_old_jobs, daemon=True).start()


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

    # Early rejection via Content-Length header (avoids writing to disk)
    content_length = request.content_length
    limit_bytes = MAX_SINGLE_SHOT_MB * 1024 * 1024
    if content_length and content_length > limit_bytes:
        return jsonify({
            "error": f"File too large for /analyze. Max {MAX_SINGLE_SHOT_MB} MB. "
                     f"For full game film (up to {MAX_FILM_MB // 1024} GB) use /analyze-film."
        }), 413

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        actual_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        if actual_mb > MAX_SINGLE_SHOT_MB:
            return jsonify({
                "error": f"File too large for /analyze ({actual_mb:.1f} MB; max {MAX_SINGLE_SHOT_MB} MB). "
                         f"For full game film (up to {MAX_FILM_MB // 1024} GB) use /analyze-film."
            }), 413
        app.logger.info("Analyzing uploaded clip: %.1f MB", actual_mb)
        return jsonify(analyze_single_shot(tmp.name)), 200
    except Exception as exc:
        app.logger.error("Analysis failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/analyze-film", methods=["POST"])
def analyze_film_submit():
    """
    Submit a full game film for async analysis.

    Returns a job ID immediately. Poll GET /analyze-film/<job_id> for status.

    Request: multipart/form-data with a 'video' field (up to 4 GB).

    Response (202):
    {
        "job_id": "uuid",
        "status": "queued",
        "poll_url": "/analyze-film/<job_id>"
    }
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Send a multipart form with key 'video'."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Early rejection via Content-Length header (avoids writing to disk)
    content_length = request.content_length
    film_limit_bytes = MAX_FILM_MB * 1024 * 1024
    if content_length and content_length > film_limit_bytes:
        return jsonify({
            "error": f"File too large. Max {MAX_FILM_MB // 1024} GB for /analyze-film."
        }), 413

    suffix  = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp     = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    job_id  = str(uuid.uuid4())

    try:
        app.logger.info("[job:%s] Receiving upload...", job_id)
        file.save(tmp.name)
        tmp.close()
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        if size_mb > MAX_FILM_MB:
            return jsonify({
                "error": f"File too large ({size_mb:.1f} MB). Max {MAX_FILM_MB // 1024} GB for /analyze-film."
            }), 413
        app.logger.info("[job:%s] Saved %.1f MB — queuing analysis", job_id, size_mb)

        # Process synchronously and return results directly
        _analyze_film_worker(job_id, tmp.name)
        job_data = _read_job(job_id)
        if job_data and job_data.get("status") == "done":
            result = job_data.get("result", {})
            return jsonify({"status": "done", **result}), 200
        else:
            return jsonify({"error": "Analysis failed"}), 500

    except Exception as exc:
        app.logger.error("[job:%s] Submit failed: %s", job_id, exc, exc_info=True)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return jsonify({"error": str(exc)}), 500


@app.route("/analyze-film/<job_id>", methods=["GET"])
def analyze_film_status(job_id: str):
    """
    Poll the status of a film analysis job.

    Possible responses:

    Queued (200):
        {"job_id": "...", "status": "queued"}

    Processing (200):
        {
            "job_id": "...",
            "status": "processing",
            "progress": {
                "frames_processed": 12000,
                "total_frames": 54000,
                "shots_found": 5,
                "percent_complete": 22.2
            }
        }

    Done (200):
        {
            "job_id": "...",
            "status": "done",
            "result": {
                "frames_processed": 54000,
                "total_shots_detected": 23,
                "avg_release_speed_sec": 0.81,
                "avg_rim_entry_angle_deg": 41.3,
                "best_shot": 14,
                "consistency_score": 78.4,
                "shots": [ ... ]
            }
        }

    Failed (200):
        {"job_id": "...", "status": "failed", "error": "..."}

    Not found (404):
        {"error": "Job not found. It may have expired (TTL: 1 hour)."}
    """
    job = _read_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found. It may have expired (TTL: 1 hour)."}), 404

    response = {"job_id": job_id, "status": job["status"]}

    if job["status"] == "processing" and "progress" in job:
        response["progress"] = job["progress"]
    elif job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return jsonify(response), 200


@app.route("/generate-email", methods=["POST"])
def generate_email():
    data = request.get_json()
    
    player_name = data.get("player_name", "")
    grad_year = data.get("grad_year", "")
    position = data.get("position", "")
    gpa = data.get("gpa", "")
    height = data.get("height", "")
    release_speed = data.get("release_speed", "")
    rim_angle = data.get("rim_entry_angle", "")
    college_name = data.get("college_name", "")
    coach_name = data.get("coach_name", "")
    player_story = data.get("player_story", "")

    prompt = f"""Write a professional basketball recruitment email from a player to a college coach.

Player: {player_name}, {position}, Class of {grad_year}, {height}, GPA {gpa}
Verified AI Stats: Release speed {release_speed}s, Rim entry angle {rim_angle} degrees
Target: Coach {coach_name} at {college_name}
Player story: {player_story}

Write a personalized 250-word email highlighting system fit, verified stats, and the player's story. Be specific about why this player fits this program. Sign off with the player's name."""

    import urllib.request
    import json as json_lib
    import urllib.request
    import json as json_lib

    payload = json_lib.dumps({
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json_lib.loads(response.read())
            email_text = result["choices"][0]["message"]["content"]
            return jsonify({"email_text": email_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ---------------------------------------------------------------------------
# Entry point — dev only (gunicorn imports app directly)
# ---------------------------------------------------------------------------
@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    data = request.get_json()
    video_url = data.get("video_url", "")
    
    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run([
            "yt-dlp",
            "-f", "worst[ext=mp4]/worst",
            "--max-filesize", "500m",
            "-o", tmp_path,
            video_url
        ], timeout=120, check=True)

        result = analyze_video(tmp_path)
        return jsonify(result)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Download timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            @app.route("/analyze-tendency", methods=["POST"])
def analyze_tendency():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files["video"]
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        file.save(tmp.name)
        tmp.close()
        
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        drives_left = 0
        drives_right = 0
        shots_off_dribble = 0
        shots_catch_and_shoot = 0
        left_side_court = 0
        right_side_court = 0
        total_possessions = 0
        quick_releases = 0
        frame_idx = 0
        prev_hip_x = None
        possession_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 != 0:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_pose.process(rgb)
            
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                left_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                right_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                left_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                
                hip_x = (left_hip.x + right_hip.x) / 2
                shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
                
                # Court position
                if hip_x < 0.5:
                    left_side_court += 1
                else:
                    right_side_court += 1
                
                # Drive direction based on body rotation
                if prev_hip_x is not None:
                    movement = hip_x - prev_hip_x
                    if abs(movement) > 0.01:
                        total_possessions += 1
                        if movement < 0:
                            drives_left += 1
                        else:
                            drives_right += 1
                
                # Quick release detection
                possession_frames += 1
                if possession_frames < 10:
                    quick_releases += 1
                
                prev_hip_x = hip_x
        
        cap.release()
        
        total_drives = drives_left + drives_right
        total_court = left_side_court + right_side_court
        
        tendencies = {
            "drives_left_pct": round((drives_left / total_drives * 100) if total_drives > 0 else 0, 1),
            "drives_right_pct": round((drives_right / total_drives * 100) if total_drives > 0 else 0, 1),
            "left_side_court_pct": round((left_side_court / total_court * 100) if total_court > 0 else 0, 1),
            "right_side_court_pct": round((right_side_court / total_court * 100) if total_court > 0 else 0, 1),
            "frames_analyzed": frame_idx,
            "total_movements": total_drives,
            "summary": []
        }
        
        # Generate tendency summary
        if tendencies["drives_left_pct"] > 60:
            tendencies["summary"].append(f"Drives LEFT {tendencies['drives_left_pct']}% of the time — defenders should overplay left")
        elif tendencies["drives_right_pct"] > 60:
            tendencies["summary"].append(f"Drives RIGHT {tendencies['drives_right_pct']}% of the time — defenders should overplay right")
        else:
            tendencies["summary"].append("Balanced driver — goes both directions equally")
            
        if tendencies["left_side_court_pct"] > 65:
            tendencies["summary"].append(f"Operates on LEFT side of court {tendencies['left_side_court_pct']}% of the time")
        elif tendencies["right_side_court_pct"] > 65:
            tendencies["summary"].append(f"Operates on RIGHT side of court {tendencies['right_side_court_pct']}% of the time")
        
        return jsonify({"status": "done", "tendencies": tendencies}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting dev server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
