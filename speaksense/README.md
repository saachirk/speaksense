# SpeakSense

## Automated Speech Diagnostic Engine

SpeakSense is an AI-assisted speech analysis platform built for the AblePro Solutions x BIG Foundation hackathon. It turns an audio sample into a clear diagnostic-style summary with speaker analysis, Kannada transcription support, acoustic biomarkers, classification, and a downloadable report.

## What It Does

- Identifies speakers and timestamps speech segments with pyannote diarization.
- Transcribes Kannada speech with faster-whisper.
- Extracts a catalogue of acoustic features including MFCCs, pitch, jitter, shimmer, HNR, formants, spectral, energy, and temporal measures.
- Classifies speaker profile as adult/child, male/female/unknown, and typical/atypical speech.
- Shows a React dashboard with clinical-style indicators and visualizations.
- Generates a PDF report for judges, clinicians, or review panels.

## Project Structure

```text
speaksense/
  src/
    api.py                 FastAPI backend
    pipeline.py            End-to-end command line runner
    preprocess.py          Audio normalization
    diarization.py         Speaker diarization
    transcription.py       Kannada ASR
    feature_extraction.py  Acoustic feature extraction
    classification.py      Ensemble models plus heuristic fallback
    explainability.py      Clinical explanations and SHAP support
    report.py              PDF report generation
    augment.py             Audio augmentation utilities
  frontend/
    src/App.jsx            React dashboard
  docs/
    acoustic_parameters.md Feature catalogue
  models/                  Trained model files
  outputs/                 Generated transcripts, reports, and JSON
```

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API

- `GET /health` - backend health check
- `GET /features/catalogue` - acoustic feature catalogue
- `POST /analyze` - upload audio and return classifications plus feature summaries
- `POST /transcribe` - upload audio and return Kannada transcript segments
- `POST /report` - generate a downloadable PDF report

## Model Note

SpeakSense uses trained ensemble models when files exist in `models/`. If the models are not trained yet, the app falls back to transparent clinical heuristics based on pitch, formants, jitter, shimmer, HNR, voicing, and pauses. This keeps the demo usable while making the source of the prediction explicit.

## Clinical Disclaimer

SpeakSense is an assistive screening and demonstration tool. It is not a medical device and does not replace evaluation by a qualified speech-language pathologist.
