"""
feature_extraction.py (FINAL + TRANSCRIPTION SUPPORT)

✔ Stable feature extraction
✔ Compatible with FastAPI
✔ Adds optional Whisper transcription helper
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import warnings
import soundfile as sf

warnings.filterwarnings("ignore")


# ─────────────────────────────
# SAFE FLOAT HANDLER (FIX)
# ─────────────────────────────
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except:
        return default


def clean_dict(d):
    for k, v in d.items():
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                d[k] = 0.0
    return d


# ─────────────────────────────
# OPTIONAL TRANSCRIPTION SUPPORT
# ─────────────────────────────
def transcribe_audio(audio_path: str, model=None, language="kn"):
    try:
        if model is None:
            return ""

        from transcription import transcribe_segment

        chunks = transcribe_segment(audio_path, model, language=language)
        return " ".join([c["text"] for c in chunks])

    except Exception:
        return ""


# ─────────────────────────────
# FEATURE CATALOGUE
# ─────────────────────────────
FEATURE_CATALOGUE = {
    "MFCCs": 39,
    "Pitch (F0)": 6,
    "Jitter": 1,
    "Shimmer": 1,
    "HNR": 1,
    "Spectral Features": 4,
    "Formants": 3,
    "Energy / RMS": 3,
    "Temporal Features": 3
}

TOTAL_FEATURES = sum(FEATURE_CATALOGUE.values())


# ─────────────────────────────
# AUDIO LOADER
# ─────────────────────────────
def _load_audio(audio_path: str, target_sr: int = 16000):
    try:
        y, sr = sf.read(audio_path)

        if isinstance(y, np.ndarray) and len(y.shape) > 1:
            y = np.mean(y, axis=1)

        y = y.astype(np.float32)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return y, sr

    except Exception:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return y, sr


# ─────────────────────────────
# MFCC
# ─────────────────────────────
def extract_mfcc_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)

    out = {}
    for i in range(13):
        out[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        out[f"mfcc_delta_{i+1}_mean"] = float(np.mean(delta[i]))
    return out


# ─────────────────────────────
# PITCH
# ─────────────────────────────
def extract_pitch_features(y, sr):
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr,
        frame_length=2048, hop_length=512
    )

    f0 = np.array(f0)
    voiced_flag = np.array(voiced_flag)

    valid = f0[voiced_flag & ~np.isnan(f0)]

    if len(valid) == 0:
        return {
            "f0_mean_hz": 0.0,
            "f0_std_hz": 0.0,
            "f0_min_hz": 0.0,
            "f0_max_hz": 0.0,
            "f0_range_hz": 0.0,
            "voiced_fraction_pitch": 0.0
        }

    return {
        "f0_mean_hz": float(np.mean(valid)),
        "f0_std_hz": float(np.std(valid)),
        "f0_min_hz": float(np.min(valid)),
        "f0_max_hz": float(np.max(valid)),
        "f0_range_hz": float(np.max(valid) - np.min(valid)),
        "voiced_fraction_pitch": float(np.mean(voiced_flag))
    }


# ─────────────────────────────
# VOICE QUALITY
# ─────────────────────────────
def extract_voice_quality_features(audio_path):
    try:
        sound = parselmouth.Sound(audio_path)
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)

        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        hnr_obj = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(hnr_obj, "Get mean", 0, 0)

        return {
            "jitter_local": safe_float(jitter),
            "shimmer_local": safe_float(shimmer),
            "hnr_mean": safe_float(hnr),
        }

    except Exception:
        return {
            "jitter_local": 0.0,
            "shimmer_local": 0.0,
            "hnr_mean": 0.0,
        }


# ─────────────────────────────
# SPECTRAL
# ─────────────────────────────
def extract_spectral_features(y, sr):
    return {
        "spectral_centroid_mean": safe_float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_bandwidth_mean": safe_float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "spectral_rolloff_mean": safe_float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zcr_mean": safe_float(np.mean(librosa.feature.zero_crossing_rate(y))),
    }


# ─────────────────────────────
# FORMANTS
# ─────────────────────────────
def extract_formant_features(audio_path):
    try:
        sound = parselmouth.Sound(audio_path)
        formants = call(sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)

        duration = sound.get_total_duration()
        times = np.arange(0, duration, 0.01)

        f1, f2, f3 = [], [], []

        for t in times:
            try:
                f1.append(call(formants, "Get value at time", 1, t, "Hertz", "Linear"))
                f2.append(call(formants, "Get value at time", 2, t, "Hertz", "Linear"))
                f3.append(call(formants, "Get value at time", 3, t, "Hertz", "Linear"))
            except:
                pass

        return {
            "f1_mean_hz": safe_float(np.mean(f1)) if f1 else 0.0,
            "f2_mean_hz": safe_float(np.mean(f2)) if f2 else 0.0,
            "f3_mean_hz": safe_float(np.mean(f3)) if f3 else 0.0,
        }

    except Exception:
        return {"f1_mean_hz": 0.0, "f2_mean_hz": 0.0, "f3_mean_hz": 0.0}


# ─────────────────────────────
# ENERGY
# ─────────────────────────────
def extract_energy_features(y):
    rms = librosa.feature.rms(y=y)[0]
    return {
        "rms_mean": safe_float(np.mean(rms)),
        "rms_std": safe_float(np.std(rms)),
        "rms_max": safe_float(np.max(rms)),
    }


# ─────────────────────────────
# TEMPORAL
# ─────────────────────────────
def extract_temporal_features(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    threshold = np.mean(rms) * 0.3

    voiced = rms > threshold
    pauses = np.sum(np.diff(voiced.astype(int)) == -1)

    duration = len(y) / sr

    return {
        "pause_count": int(pauses),
        "speech_rate_approx": safe_float(pauses / duration if duration > 0 else 0.0),
        "voiced_fraction_temporal": safe_float(np.mean(voiced)),
    }


# ─────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────
def extract_all_features(audio_path: str, y=None, sr=16000):
    if y is None:
        y, sr = _load_audio(audio_path, sr)

    features = {"filename": audio_path}

    features.update(extract_mfcc_features(y, sr))
    features.update(extract_pitch_features(y, sr))
    features.update(extract_voice_quality_features(audio_path))
    features.update(extract_spectral_features(y, sr))
    features.update(extract_formant_features(audio_path))
    features.update(extract_energy_features(y))
    features.update(extract_temporal_features(y, sr))

    return clean_dict(features)