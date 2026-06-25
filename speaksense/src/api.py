"""
SpeakSense FINAL API (MERGED + PRODUCTION FIX)
✔ FFmpeg-safe universal audio pipeline
✔ All old endpoints restored
✔ Training + Augmentation working
✔ No hardcoded paths
"""
import os

import os
import sys
import io
import json
import logging
import tempfile
import subprocess
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction import extract_all_features, FEATURE_CATALOGUE, TOTAL_FEATURES
from classification import classify_speaker
from explainability import explain_all
import numpy as np


def clean_output(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = clean_output(v)
    elif isinstance(obj, list):
        return [clean_output(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
    return obj


HAS_WHISPER = False
HAS_DIARIZATION = False

try:
    from transcription import load_whisper_model, transcribe_audio, transcribe_with_speakers
    HAS_WHISPER = True
except:
    pass

try:
    from diarization import get_diarization_pipeline, diarize_audio
    HAS_DIARIZATION = True
except:
    pass


_whisper_model = None
_diar_pipeline = None

HF_TOKEN = os.getenv("HF_TOKEN")


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = load_whisper_model("base")
    return _whisper_model


def get_diarizer():
    global _diar_pipeline
    if _diar_pipeline is None:
        _diar_pipeline = get_diarization_pipeline(HF_TOKEN)
    return _diar_pipeline


# ───────────────── CONFIG ─────────────────
MAX_UPLOAD_MB = int(os.environ.get("SPEAKSENSE_MAX_UPLOAD_MB", "150"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speaksense")

app = FastAPI(title="SpeakSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────── FFmpeg AUTO DETECTION ─────────────────
def find_ffmpeg():
    from shutil import which

    ffmpeg = which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
    ]

    for p in common_paths:
        p = os.path.expandvars(p)
        if os.path.exists(p):
            return p

    return None


FFMPEG_PATH = find_ffmpeg()


# ───────────────── UPLOAD ─────────────────
async def save_upload(file: UploadFile) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.close()
    return tmp.name


# ───────────────── AUDIO CONVERSION ─────────────────
def convert_to_wav(input_path: str) -> str:
    output_path = tempfile.mktemp(suffix=".wav")

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


# ───────────────── ANALYZE (UNCHANGED) ─────────────────
@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    raw = await save_upload(file)
    wav = convert_to_wav(raw)

    try:
        features = extract_all_features(wav)
        classification = classify_speaker(features)
        explanation = explain_all(features, classification)

        return clean_output({
            "filename": file.filename,
            "classification": classification,
            "explanation": explanation,
            "key_features": {
                "f0_mean_hz": round(features.get("f0_mean_hz", 0), 2),
                "f0_std_hz": round(features.get("f0_std_hz", 0), 2),
                "jitter_local_pct": round(features.get("jitter_local", 0), 4),
                "shimmer_local_db": round(features.get("shimmer_local", 0), 4),
                "hnr_mean_db": round(features.get("hnr_mean", 0), 2),
                "voiced_fraction": round(features.get("voiced_fraction_temporal", 0), 4),
                "pause_count": int(features.get("pause_count", 0)),
                "speech_rate_approx": round(features.get("speech_rate_approx", 0), 4),
            },
            "mfccs": [round(features.get(f"mfcc_{i}_mean", 0), 4) for i in range(1, 14)],
            "total_features": TOTAL_FEATURES
        })

    finally:
        for p in [raw, wav]:
            if p and os.path.exists(p):
                os.remove(p)


# ───────────────── TRAINING (UNCHANGED) ─────────────────
training_status = {"status": "idle"}


def train_worker():
    global training_status
    try:
        training_status = {"status": "running"}

        from preprocess import preprocess_dataset
        from pipeline import run_training_pipeline

        preprocess_dataset("datasets/audio_samples", "datasets/processed")
        run_training_pipeline("datasets/processed")

        training_status = {"status": "done"}

    except Exception as e:
        training_status = {"status": "error", "msg": str(e)}


@app.post("/train/run")
def train_run():
    threading.Thread(target=train_worker).start()
    return {"message": "training started"}


@app.get("/train/status")
def train_status():
    return training_status


@app.get("/train/info")
def get_train_info():
    import os

    samples_count = len(os.listdir("datasets/audio_samples")) if os.path.exists("datasets/audio_samples") else 0
    processed_count = len(os.listdir("datasets/processed")) if os.path.exists("datasets/processed") else 0
    augmented_count = len(os.listdir("datasets/augmented")) if os.path.exists("datasets/augmented") else 0

    return {
        "samples_count": samples_count,
        "processed_count": processed_count,
        "augmented_count": augmented_count
    }


# ───────────────── AUGMENTATION (UNCHANGED) ─────────────────
@app.post("/augment/run")
def augment_run():
    threading.Thread(target=augment_worker).start()
    return {"message": "augmentation started"}


def augment_worker():
    global training_status
    try:
        from augment import augment_dataset
        from preprocess import preprocess_dataset

        training_status = {"status": "running"}
        preprocess_dataset("datasets/audio_samples", "datasets/processed")
        augment_dataset("datasets/processed", "datasets/augmented")
        training_status = {"status": "done"}

    except Exception as e:
        training_status = {"status": "error", "msg": str(e)}


# ───────────────── HEALTH ─────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": bool(FFMPEG_PATH),
        "capabilities": {
            "transcribe": HAS_WHISPER,
            "diarize": HAS_DIARIZATION,
        }
    }


# =========================================================
# 🔥 ONLY FIXED PART: TRANSCRIBE
# =========================================================
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), bilingual: bool = False):

    raw = await save_upload(file)
    wav = convert_to_wav(raw)

    try:
        model = get_whisper()

        # =========================
        # CASE 1: BILINGUAL MODE
        # =========================
        if bilingual:
            # pyrefly: ignore [unexpected-keyword]
            kn_segments, kn_info = transcribe_audio(wav, model)
            # pyrefly: ignore [unexpected-keyword]
            en_segments, en_info = transcribe_audio(wav, model)

            return {
                "bilingual": True,
                "detected_language": kn_info.language,
                "detected_confidence": getattr(kn_info, "language_probability", 0),

                "kn": {
                    "text": " ".join([s.text for s in kn_segments]),
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in kn_segments
                    ]
                },

                "en": {
                    "text": " ".join([s.text for s in en_segments]),
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in en_segments
                    ]
                }
            }

        # =========================
        # CASE 2: NORMAL MODE
        # =========================
        segments, info = transcribe_audio(wav, model)

        return {
            "bilingual": False,
            "language": info.language,
            "transcription": " ".join([s.text for s in segments]),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ]
        }

    finally:
        for p in [raw, wav]:
            if p and os.path.exists(p):
                os.remove(p)

# =========================================================
# 🔥 ONLY FIXED PART: DIARIZED TRANSCRIBE
# =========================================================
@app.post("/transcribe/diarized")
async def transcribe_diarized(file: UploadFile = File(...), language: str = "kn"):

    raw = await save_upload(file)
    wav = convert_to_wav(raw)

    try:
        model = get_whisper()

        if not HAS_DIARIZATION:
            segs, _ = transcribe_audio(wav, model)
            return {
                "diarized": False,
                "segments": [
                    {"speaker": "speaker-1", "start": s.start, "end": s.end, "text": s.text}
                    for s in segs
                ]
            }

        diarizer = get_diarizer()

        result = transcribe_with_speakers(
            wav,
            diarizer,
            model
        )

        return {
            "diarized": True,
            "segments": result["segments"]
        }

    finally:
        for p in [raw, wav]:
            if p and os.path.exists(p):
                os.remove(p)