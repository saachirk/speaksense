"""
diarization.py
──────────────
Speaker diarization module (API-ready)
"""

import os
import torch
from pyannote.audio import Pipeline


# -----------------------------
# LOAD DIARIZATION PIPELINE
# -----------------------------
def get_diarization_pipeline(hf_token: str) -> Pipeline:
    model_id = "pyannote/speaker-diarization-3.1"

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    except TypeError:
        # pyrefly: ignore [unexpected-keyword]
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    return pipeline


# -----------------------------
# RUN DIARIZATION
# -----------------------------
def diarize_audio(audio_path: str, pipeline: Pipeline):
    """
    Returns cleaned diarization segments
    """

    diarization = pipeline(audio_path)

    segments = []
    speaker_map = {}
    counter = 1

    for turn, _, speaker in diarization.itertracks(yield_label=True):

        if speaker not in speaker_map:
            speaker_map[speaker] = f"speaker-{counter}"
            counter += 1

        segments.append({
            "speaker": speaker_map[speaker],
            "start": round(turn.start, 3),
            "end": round(turn.end, 3)
        })

    return segments