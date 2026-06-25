import os
import json
import torch
from faster_whisper import WhisperModel


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_whisper_model(model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[Whisper] Loading model={model_size} device={device}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    return model


# -----------------------------
# TRANSCRIBE AUDIO (FIXED API FUNCTION)
# -----------------------------
def transcribe_audio(audio_path, model):
    segments, info = model.transcribe(
        audio_path,
        language=None,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False
    )

    return list(segments), info


# -----------------------------
# SIMPLE SEGMENT FORMATTER (IMPORTANT FOR API)
# -----------------------------
def transcribe_segment(audio_path, model, language="kn", beam_size=1):
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True
    )

    return [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip()
        }
        for seg in segments
    ]


# -----------------------------
# SPEAKER MERGE (OPTIONAL USE)
# -----------------------------
def merge_speakers(diarization_segments, whisper_segments):
    enriched = []

    for spk in diarization_segments:
        spk_start = spk["start"]
        spk_end = spk["end"]

        texts = []

        for seg in whisper_segments:
            if seg["start"] <= spk_end and seg["end"] >= spk_start:
                texts.append(seg["text"])

        enriched.append({
            "speaker": spk["speaker"],
            "start": spk_start,
            "end": spk_end,
            "transcription": " ".join(texts)
        })

    return enriched


# -----------------------------
# DIARIZED PIPELINE WRAPPER
# -----------------------------
def transcribe_with_speakers(audio_path, diarization_pipeline, model):
    diarization = diarization_pipeline(audio_path)

    diarization_segments = [
        {
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    whisper_segments, info = transcribe_audio(audio_path, model)

    enriched = merge_speakers(diarization_segments, whisper_segments)

    return {
        "language_detected": info.language,
        "language_probability": float(info.language_probability),
        "segments": enriched
    }


# -----------------------------
# FORMAT OUTPUT
# -----------------------------
def format_output(result):
    lines = []

    for seg in result["segments"]:
        start = seg["start"]
        end = seg["end"]

        m1, s1 = int(start // 60), start % 60
        m2, s2 = int(end // 60), end % 60

        lines.append(f"[{m1:02d}:{s1:06.3f} → {m2:02d}:{s2:06.3f}] {seg['speaker']}")
        lines.append(seg["transcription"])
        lines.append("")

    return "\n".join(lines)