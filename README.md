# CourtSense Vision Engine

A Python Flask API that analyzes basketball shot videos using computer vision. Upload a video clip and get back instant coaching feedback — shot phase detection, release speed, and rim entry angle — all from a single HTTP request.

## What It Does

| Output | Description |
|---|---|
| `shot_detected` | Whether a recognizable shooting motion was found |
| `frames_processed` | Total frames analyzed |
| `release_speed_sec` | Time in seconds from dip to release |
| `rim_entry_angle_deg` | Estimated ball angle at rim entry (higher = better arc) |
| `feedback` | Array of plain-English coaching tips |

**How it works:**

1. **Pose estimation** — MediaPipe Pose tracks 33 body landmarks per frame and classifies each frame as `dip`, `load`, or `release` based on wrist/elbow/shoulder/hip positions
2. **Release speed** — measured as `(release_frame − dip_frame) / fps`
3. **Ball detection** — HSV color masking in the orange range finds the basketball each frame; the final trajectory segment gives the rim entry angle estimate

## Requirements

- Python 3.11
- See [`artifacts/vision-engine/requirements.txt`](artifacts/vision-engine/requirements.txt)

## Installation

```bash
git clone https://github.com/oliver9842/courtsense-vision.git
cd courtsense-vision/artifacts/vision-engine

pip install -r requirements.txt
```

## Running Locally

```bash
cd artifacts/vision-engine
python app.py
```

The server starts on port `5000` by default. Set the `PORT` environment variable to override.

## Running with Gunicorn (Production)

```bash
cd artifacts/vision-engine
gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:8000
```

Or use the included `Procfile` with a platform like Railway or Render — just point the root directory to `artifacts/vision-engine/`.

## API Reference

### `GET /`

Health check.

**Response:**
```
CourtSense Vision Engine is running
```

---

### `POST /analyze`

Analyze a basketball shot video.

**Request:** multipart/form-data with a `video` field containing the video file.

**Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` (max 200 MB)

**Response:**
```json
{
  "shot_detected": true,
  "frames_processed": 312,
  "release_speed_sec": 0.733,
  "rim_entry_angle_deg": 47.2,
  "feedback": [
    "Good release tempo at 0.73s. Keep that rhythm consistent across different shot types and distances.",
    "Excellent arc! The ball entered the rim at 47.2° — a high arc increases the effective target size of the rim."
  ]
}
```

**Example with curl:**

```bash
curl -X POST http://localhost:5000/analyze \
  -F "video=@my_shot.mp4"
```

**Error response:**
```json
{
  "error": "No video file provided. Send a multipart form with key 'video'."
}
```

## Deploying to Railway

1. Fork or push this repo to your GitHub account
2. Create a new Railway project → **Deploy from GitHub repo**
3. Set the **Root Directory** to `artifacts/vision-engine`
4. Railway detects the `Procfile` automatically and deploys with gunicorn

## Project Structure

```
artifacts/vision-engine/
├── app.py              # Flask application — all endpoints and analysis logic
├── requirements.txt    # Python dependencies
├── Procfile            # gunicorn production command
└── start.sh            # Local dev helper script

.python-version         # Python 3.11.0
```

## Tech Stack

- [Flask](https://flask.palletsprojects.com/) — web framework
- [MediaPipe](https://developers.google.com/mediapipe) — pose landmark detection
- [OpenCV](https://opencv.org/) — video frame processing and color detection
- [NumPy](https://numpy.org/) — array operations
- [gunicorn](https://gunicorn.org/) — production WSGI server
