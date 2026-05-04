# Workspace

## Overview

pnpm workspace monorepo using TypeScript, plus a Python Flask computer vision API.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## CourtSense Vision Engine (Python Flask)

Located at `artifacts/vision-engine/app.py`. Runs on port 5000.

### Endpoints

- `GET /` — Returns "CourtSense Vision Engine is running"
- `POST /analyze` — Accepts multipart video upload (field: `video`), returns JSON:
  ```json
  {
    "shot_detected": true,
    "frames_processed": 300,
    "release_speed_sec": 0.733,
    "rim_entry_angle_deg": 42.5,
    "feedback": ["...coaching tips..."]
  }
  ```

### Dependencies

- `flask` — web framework
- `mediapipe==0.10.9` — pose estimation (solutions API)
- `opencv-python-headless` — video frame processing
- `numpy` — array math

### Workflow

- Name: **CourtSense Vision Engine**
- Command: `cd artifacts/vision-engine && python app.py`

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## GitHub Integration Note

The Replit GitHub OAuth connector was dismissed. To push to GitHub in a future session, either:
1. Re-authorize via the Replit GitHub integration (connector:ccfg_github_01K4B9XD3VRVD2F99YM91YTCAF), or
2. Provide a GitHub Personal Access Token (PAT with `repo` scope) and store it as the secret `GITHUB_TOKEN`, then provide the desired repo name and visibility.
