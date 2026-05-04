import os
import time
import tempfile
import math
import json
import traceback

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify

app = Flask(__name__)

mp_pose = mp.solutions.pose

MAX_UPLOAD_MB = 200
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Basketball (orange) detection helpers
# ---------------------------------------------------------------------------

ORANGE_LOWER = np.array([5, 120, 120], dtype=np.uint8)
ORANGE_UPPER = np.array([25, 255, 255], dtype=np.uint8)


def detect_ball_center(frame: np.ndarray):
    """Return (cx, cy) of the largest orange blob, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 100:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def estimate_rim_entry_angle(ball_trajectory: list) -> float | None:
    """
    Estimate the angle (degrees from horizontal) at which the ball enters the rim
    based on the final segment of the trajectory.
    Negative = descending (good for a shot).
    """
    if len(ball_trajectory) < 4:
        return None

    pts = ball_trajectory[-6:]
    if len(pts) < 2:
        return None

    x1, y1 = pts[0]
    x2, y2 = pts[-1]

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return None

    angle_rad = math.atan2(-dy, dx)
    angle_deg = math.degrees(angle_rad)
    return round(angle_deg, 1)


# ---------------------------------------------------------------------------
# Pose / shot-phase helpers
# ---------------------------------------------------------------------------

def get_landmark_y(landmarks, idx: int, frame_h: int) -> float:
    return landmarks[idx].y * frame_h


def get_landmark_x(landmarks, idx: int, frame_w: int) -> float:
    return landmarks[idx].x * frame_w


def classify_shot_phase(landmarks, frame_h: int, frame_w: int) -> str:
    """
    Classify the current pose into a shot phase using wrist/shoulder/hip positions.
    Phases: idle, dip, load, release
    """
    r_wrist_y = get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, frame_h)
    r_elbow_y = get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, frame_h)
    r_shoulder_y = get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, frame_h)
    r_hip_y = get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, frame_h)

    l_wrist_y = get_landmark_y(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, frame_h)
    l_shoulder_y = get_landmark_y(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, frame_h)

    avg_wrist_y = (r_wrist_y + l_wrist_y) / 2
    avg_shoulder_y = (r_shoulder_y + l_shoulder_y) / 2

    torso_height = abs(avg_shoulder_y - r_hip_y)
    if torso_height == 0:
        return "idle"

    wrist_below_hip = r_wrist_y > r_hip_y
    wrist_near_shoulder = abs(r_wrist_y - r_shoulder_y) < torso_height * 0.4
    wrist_above_shoulder = r_wrist_y < r_shoulder_y
    elbow_above_shoulder = r_elbow_y < r_shoulder_y

    if wrist_below_hip:
        return "dip"
    elif wrist_near_shoulder and not elbow_above_shoulder:
        return "load"
    elif wrist_above_shoulder or elbow_above_shoulder:
        return "release"
    else:
        return "idle"


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------

def generate_feedback(
    release_speed_sec: float | None,
    rim_entry_angle_deg: float | None,
    phase_sequence: list,
    ball_detected: bool,
) -> list:
    tips = []

    if not ball_detected:
        tips.append(
            "No basketball was detected in the video. Make sure the ball is clearly visible "
            "and well-lit for accurate rim entry angle analysis."
        )

    if "dip" not in phase_sequence:
        tips.append(
            "No pre-shot dip was detected. A controlled dip before your shot loads energy "
            "into your legs and helps generate upward power."
        )
    elif "load" not in phase_sequence:
        tips.append(
            "Dip was detected but the load phase was missing. Focus on bringing the ball up "
            "to your shooting pocket before releasing."
        )

    if release_speed_sec is not None:
        if release_speed_sec < 0.3:
            tips.append(
                f"Your release was very quick ({release_speed_sec:.2f}s). While speed can be "
                "good, make sure you are not rushing and skipping the set position."
            )
        elif release_speed_sec > 1.5:
            tips.append(
                f"Your shot motion took {release_speed_sec:.2f}s from dip to release, which is "
                "slow. Work on quickening your release to make it harder for defenders to contest."
            )
        else:
            tips.append(
                f"Good release tempo at {release_speed_sec:.2f}s. Keep that rhythm consistent "
                "across different shot types and distances."
            )

    if rim_entry_angle_deg is not None:
        if rim_entry_angle_deg > 45:
            tips.append(
                f"Excellent arc! The ball entered the rim at {rim_entry_angle_deg:.1f}° — "
                "a high arc increases the effective target size of the rim."
            )
        elif rim_entry_angle_deg > 30:
            tips.append(
                f"Decent arc at {rim_entry_angle_deg:.1f}°. Try to aim for 45°+ for a larger "
                "margin of error at the rim."
            )
        else:
            tips.append(
                f"Low arc detected at {rim_entry_angle_deg:.1f}°. A flat shot decreases your "
                "chances of the ball going in. Practice shooting upward with a high follow-through."
            )

    if not tips:
        tips.append(
            "Shot mechanics look solid overall. Keep refining consistency and work on "
            "replicating this form under game speed and pressure."
        )

    return tips


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_processed = 0

    phase_sequence = []
    last_phase = "idle"

    dip_frame = None
    release_frame = None

    ball_trajectory = []
    ball_detected_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1

            h, w = frame.shape[:2]

            ball_pos = detect_ball_center(frame)
            if ball_pos:
                ball_trajectory.append(ball_pos)
                ball_detected_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                phase = classify_shot_phase(lm, h, w)

                if phase != last_phase:
                    phase_sequence.append(phase)
                    last_phase = phase

                    if phase == "dip" and dip_frame is None:
                        dip_frame = frames_processed

                    if phase == "release" and dip_frame is not None and release_frame is None:
                        release_frame = frames_processed

    cap.release()

    release_speed_sec = None
    if dip_frame is not None and release_frame is not None and fps > 0:
        release_speed_sec = round((release_frame - dip_frame) / fps, 3)

    rim_entry_angle_deg = estimate_rim_entry_angle(ball_trajectory)

    ball_detected = ball_detected_count > 0
    shot_detected = ("dip" in phase_sequence or "load" in phase_sequence or "release" in phase_sequence)

    feedback = generate_feedback(release_speed_sec, rim_entry_angle_deg, phase_sequence, ball_detected)

    return {
        "shot_detected": shot_detected,
        "frames_processed": frames_processed,
        "release_speed_sec": release_speed_sec,
        "rim_entry_angle_deg": rim_entry_angle_deg,
        "feedback": feedback,
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

        result = analyze_video(tmp.name)
        return jsonify(result), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
