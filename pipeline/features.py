"""Acoustic feature extraction - speech treated as a time-series signal.

Frame features (every 20 ms): F0 pitch (pYIN), RMS energy, spectral centroid,
spectral flatness and 13 MFCCs.
Word features (one row per aligned word): duration, pause before the word,
local speaking rate, pitch level and range, loudness and clarity.

Pitch and loudness are SPEAKER-NORMALISED so different voices and microphones can be compared:
- pitch is in semitones relative to the speaker's own median pitch
- loudness is in dB relative to the speaker's own median loudness while speaking
"""
import numpy as np
import librosa

SR = 16000
HOP = 320            # 20 ms
N_FFT = 1024
HOP_S = HOP / SR


def load_audio(audio_path):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    return y


def frame_features(audio_path):
    y = load_audio(audio_path)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)

    n = min(len(f0), len(rms), len(centroid), len(flatness), mfcc.shape[1])
    f0 = f0[:n]
    rms_db = librosa.amplitude_to_db(rms[:n] + 1e-10, ref=1.0)

    # speaker normalisation
    voiced_f0 = f0[~np.isnan(f0)]
    f0_median = float(np.median(voiced_f0)) if len(voiced_f0) else 150.0
    f0_st = 12 * np.log2(f0 / f0_median)                 # NaN where unvoiced
    speech = rms_db > (np.max(rms_db) - 40)              # frames that contain speech
    level = float(np.median(rms_db[speech])) if speech.any() else float(np.median(rms_db))
    energy_rel = rms_db - level

    return {
        "t": np.arange(n) * HOP_S,
        "f0_hz": f0,
        "f0_st": f0_st,
        "f0_median_hz": f0_median,
        "rms_db": rms_db,
        "energy_rel": energy_rel,
        "centroid": centroid[:n],
        "flatness": flatness[:n],
        "mfcc": mfcc[:, :n],
        "duration": len(y) / SR,
    }


def _frames_between(frames, start, end):
    t = frames["t"]
    idx = np.where((t >= start) & (t < end))[0]
    if len(idx) == 0:  # very short word: take the nearest frame
        idx = np.array([min(int(round(start / HOP_S)), len(t) - 1)])
    return idx


def _nanmean(x, default=np.nan):
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else default


SILENCE_DB = -25.0   # frames this far below the speaker's median loudness count as silence


def _silence_before(frames, words, i):
    """Longest silent stretch between the middle of word i-1 and the middle of word i.
    Whisper often stretches word boundaries into silences, so we measure the pause from the
    audio energy itself. Returns (length_s, start_s, end_s)."""
    if i == 0:
        return 0.0, words[0]["start"], words[0]["start"]
    a = (words[i - 1]["start"] + words[i - 1]["end"]) / 2
    b = (words[i]["start"] + words[i]["end"]) / 2
    t = frames["t"]
    idx = np.where((t >= a) & (t < b))[0]
    gap = max(0.0, words[i]["start"] - words[i - 1]["end"])
    best = (gap, words[i - 1]["end"], words[i]["start"])
    if len(idx) == 0:
        return best
    silent = frames["energy_rel"][idx] < SILENCE_DB
    run_start = None
    for k, s in enumerate(list(silent) + [False]):
        if s and run_start is None:
            run_start = k
        elif not s and run_start is not None:
            length = (k - run_start) * HOP_S
            if length > best[0]:
                best = (length, float(t[idx[run_start]]), float(t[idx[run_start]]) + length)
            run_start = None
    return best


def word_features(frames, words):
    """One dict per word with the features compared against the baseline."""
    out = []
    n = len(words)
    for i, w in enumerate(words):
        lo, hi = max(0, i - 2), min(n - 1, i + 2)      # 5-word window
        span = words[hi]["end"] - words[lo]["start"]
        local_rate = (hi - lo + 1) / max(span, 0.2)

        idx = _frames_between(frames, w["start"], w["end"])
        win = _frames_between(frames, words[lo]["start"], words[hi]["end"])
        win_f0 = frames["f0_st"][win]
        win_f0 = win_f0[~np.isnan(win_f0)]
        f0_range = float(np.percentile(win_f0, 90) - np.percentile(win_f0, 10)) if len(win_f0) >= 3 else 0.0

        pause, pause_start, pause_end = _silence_before(frames, words, i)
        out.append({
            "duration": w["end"] - w["start"],
            "pause_before": pause,
            "pause_start": pause_start,
            "pause_end": pause_end,
            "local_rate": local_rate,
            "f0_mean": _nanmean(frames["f0_st"][idx], 0.0),
            "f0_range": f0_range,
            "energy_db": float(np.mean(frames["energy_rel"][idx])),
            "clarity": float(w.get("conf", 0.0)),
        })
    return out


def fft_spectrum(audio_path, start, end, max_hz=4000, points=200):
    """FFT magnitude (dB) of a time region, binned to `points` values up to max_hz."""
    y = load_audio(audio_path)
    seg = y[int(start * SR):int(end * SR)]
    if len(seg) < N_FFT:
        seg = np.pad(seg, (0, N_FFT - len(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    keep = freqs <= max_hz
    spec, freqs = spec[keep], freqs[keep]
    edges = np.linspace(0, max_hz, points + 1)
    out_f, out_db = [], []
    for k in range(points):
        m = (freqs >= edges[k]) & (freqs < edges[k + 1])
        if m.any():
            out_f.append(float((edges[k] + edges[k + 1]) / 2))
            out_db.append(float(20 * np.log10(np.mean(spec[m]) + 1e-9)))
    return {"freq": out_f, "db": out_db}


def mfcc_region(frames, start, end):
    idx = _frames_between(frames, start, end)
    return frames["mfcc"][:, idx].round(2).tolist()
