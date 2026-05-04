import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "artifacts", "vision-engine"))

from app import app  # noqa: F401 — re-exported for gunicorn (main:app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
