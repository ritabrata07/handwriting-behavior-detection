# Emotion detection from handwriting

Interactive final-year AI/ML project website using a Flask API + CNN inference pipeline.

## Features
- Upload handwriting images with drag/drop UI.
- Predict emotion classes from trained model outputs.
- Confidence bars for Calm, Stressed, Angry, and Focused.
- Happy class displayed as `Pending` (not trained in current model artifacts).
- Health check and robust API error responses.

## Current Class Mapping
- `CLASS_0 -> Calm`
- `CLASS_1 -> Stressed`
- `CLASS_2 -> Angry`
- `CLASS_3 -> Focused`

## API Contract
### `GET /api/health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /api/predict`
- Content-Type: `multipart/form-data`
- Field name: `file`

Success shape:
```json
{
  "predicted_emotion": "Calm",
  "predicted_class": "CLASS_0",
  "confidence": 0.93,
  "probabilities": {
    "Calm": 0.93,
    "Stressed": 0.03,
    "Angry": 0.02,
    "Focused": 0.02
  },
  "happy_status": "pending_not_trained"
}
```

Error shape:
```json
{ "error_code": "MODEL_UNAVAILABLE", "message": "..." }
```

## Environment Variables
- `MODEL_ASSET_DIR` (default: `/Users/ritabratamaity/Handwriting Project/Project/model_files`)
- `FLASK_ENV` (default: `production`)
- `PORT` (default: `5000`)
- `MAX_UPLOAD_MB` (default: `8`)

## Docker-First Run (Primary)
1. Install Docker Desktop.
2. Optional: copy `.env.example` to `.env` and adjust `HOST_MODEL_ASSET_DIR`.
3. Start:
   ```bash
   docker compose up --build
   ```
4. Open: `http://localhost:5000`

## Local Fallback Run (No Docker)
Use Python 3.11 for TensorFlow compatibility.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MODEL_ASSET_DIR="/Users/ritabratamaity/Handwriting Project/Project/model_files"
python -m backend
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Project Structure
- `backend/` Flask app, config, inference pipeline
- `templates/` main page HTML
- `static/` CSS + JavaScript for interactive UI
- `tests/` unit and API tests
