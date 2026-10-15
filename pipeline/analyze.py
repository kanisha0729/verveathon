"""One call that runs the whole pipeline and returns the result dict the dashboard renders.

    from pipeline.analyze import analyze
    result = analyze("data/flawed/human/flaw_L3_moderate.wav")

Results are cached in results/<file name>.json, the baseline's alignment and features in
results/_baseline_<name>.pkl, so re-running is instant.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.align import align, norm                                      # noqa: E402
from pipeline.features import frame_features, word_features, fft_spectrum, mfcc_region  # noqa: E402
from pipeline.compare import zscores, find_regions, rubric_scores, overlay, delivery_profile  # noqa: E402

BASELINES = {"jfk": {"audio": ROOT / "data/baseline/jfk_baseline.wav",
                     "transcript": ROOT / "data/transcripts/jfk.txt"}}
RESULTS = ROOT / "results"
PIPELINE_VERSION = 3   # bump when features/compare change, so old cached results are ignored


MIN_SPEECH_S = 2.0      # less audible speech than this -> the microphone captured (almost) nothing
MIN_COVERAGE = 0.25     # fewer script words recognised than this -> wrong text or unusable recording
SPEECH_DBFS = -50.0     # 20 ms frames louder than this (absolute dBFS) count as audible sound


class NoSpeechError(ValueError):
    """Raised (in strict mode) when a recording can't be scored fairly."""


def audio_check(audio):
    """Cheap sanity check of a recording BEFORE scoring. `audio` = path or raw bytes.
    Returns peak level, seconds of audible sound and a verdict. Pure digital silence (all zeros)
    is what Windows / the browser deliver when the microphone is blocked or muted."""
    import io
    import numpy as np
    import librosa
    src = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else str(audio)
    y, sr = librosa.load(src, sr=16000, mono=True)
    dur = len(y) / 16000
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=320)[0] if len(y) else np.zeros(1)
    loud_s = float(np.sum(20 * np.log10(rms + 1e-10) > SPEECH_DBFS) * 0.02)
    peak_db = 20 * np.log10(peak + 1e-10)
    if peak < 1e-4:
        verdict, msg = "silent", ("The recording is completely silent: the microphone sent no sound. "
                                  "Check the browser's microphone permission and the selected input device.")
    elif loud_s < MIN_SPEECH_S:
        verdict, msg = "too_quiet", (f"Only {loud_s:.1f} s of audible sound (peak {peak_db:.0f} dBFS). "
                                     "Speak closer to the microphone or raise the input volume.")
    else:
        verdict, msg = "ok", f"{loud_s:.1f} s of audible sound detected (peak {peak_db:.0f} dBFS)."
    return {"verdict": verdict, "message": msg, "peak_db": round(float(peak_db), 1),
            "speech_s": round(loud_s, 1), "duration": round(dur, 1)}


def _keep_real_words(words):
    return [w for w in words if norm(w["word"])]


def _process(audio_path, transcript_text):
    words = _keep_real_words(align(audio_path, transcript_text))
    frames = frame_features(audio_path)
    feats = word_features(frames, words)
    return words, frames, feats


def baseline_config(baseline="jfk"):
    """A baseline is either a built-in name ("jfk") or a dict made by custom_baseline()."""
    if isinstance(baseline, dict):
        return baseline
    cfg = BASELINES[baseline]
    return {"name": baseline, "label": "JFK, 1961 inaugural address", "audio": str(cfg["audio"]),
            "transcript": cfg["transcript"].read_text(encoding="utf-8")}


def custom_baseline(audio_path, transcript_text, label="Custom baseline"):
    """Use ANY ideal recording + its transcript as the baseline (e.g. a coach reading a new script)."""
    h = hashlib.md5(Path(audio_path).read_bytes() + transcript_text.encode("utf-8")).hexdigest()[:10]
    return {"name": f"custom_{h}", "label": label, "audio": str(audio_path), "transcript": transcript_text}


def load_baseline(baseline="jfk", progress=None):
    cfg = baseline_config(baseline)
    RESULTS.mkdir(exist_ok=True)
    cache = RESULTS / f"_baseline_{cfg['name']}_v{PIPELINE_VERSION}.pkl"
    audio = Path(cfg["audio"])
    if cache.exists() and cache.stat().st_mtime > audio.stat().st_mtime:
        with open(cache, "rb") as f:
            return pickle.load(f)
    if progress:
        progress("Analysing the baseline (first run only)", 0.05)
    data = _process(audio, cfg["transcript"])
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def _check_coverage(result, strict):
    if strict and result.get("coverage", 1.0) < MIN_COVERAGE:
        raise NoSpeechError(
            f"Only {result.get('matched_words', 0)} of {len(result.get('words', []))} script words were recognised "
            f"({result.get('coverage', 0):.0%}). Read the exact text shown in the transcript box, clearly and "
            f"from the beginning, then try again.")


def analyze(audio_path, transcript_text=None, baseline="jfk", progress=None, use_cache=True, strict=False):
    """progress(step_name, fraction) is called after each stage if given (for the UI).
    strict=True (used for live participant attempts) refuses to score silent / unrecognisable recordings
    instead of returning a meaningless 0, by raising NoSpeechError with a human-readable reason."""
    audio_path = Path(audio_path)
    if strict:
        chk = audio_check(audio_path)
        if chk["verdict"] != "ok":
            raise NoSpeechError(chk["message"])
    cfg = baseline_config(baseline)
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if cfg["name"] == "jfk" else f"__{cfg['name']}"
    digest = hashlib.md5(audio_path.read_bytes()).hexdigest()[:8]   # different audio -> different cache entry
    out_file = RESULTS / f"{audio_path.stem}_{digest}{suffix}.json"
    if use_cache and out_file.exists() and out_file.stat().st_mtime > audio_path.stat().st_mtime:
        cached = json.loads(out_file.read_text(encoding="utf-8"))
        if cached.get("pipeline_version") == PIPELINE_VERSION:
            _check_coverage(cached, strict)
            return cached

    if not transcript_text or not transcript_text.strip():
        transcript_text = cfg["transcript"]

    def step(name, frac):
        if progress:
            progress(name, frac)

    b_words, b_frames, b_feats = load_baseline(baseline, progress)
    step("Aligning words to the transcript", 0.2)
    p_words = _keep_real_words(align(audio_path, transcript_text))
    step("Extracting acoustic features", 0.45)
    p_frames = frame_features(audio_path)
    p_feats = word_features(p_frames, p_words)
    step("Comparing with the baseline", 0.75)
    z = zscores(p_feats, b_feats)
    regions = find_regions(p_words, b_words, p_feats, b_feats, z)
    step("Scoring", 0.9)
    matched = sum(1 for w in p_words if w["conf"] > 0)
    coverage = matched / max(len(p_words), 1)
    rubric, overall = rubric_scores(z, regions, coverage)

    result = {
        "file": audio_path.name,
        "baseline": cfg["name"],
        "baseline_label": cfg.get("label", cfg["name"]),
        "baseline_audio": cfg["audio"],
        "duration": round(float(p_frames["duration"]), 2),
        "words": p_words,
        "overlay": overlay(p_frames, b_frames, p_words, b_words, p_feats, b_feats),
        "regions": regions,
        "rubric": rubric,
        "overall": overall,
        "matched_words": int(matched),
        "coverage": round(float(coverage), 3),
        "profile": {"participant": delivery_profile(p_words, p_frames, p_feats),
                    "baseline": delivery_profile(b_words, b_frames, b_feats)},
        "pipeline_version": PIPELINE_VERSION,
    }
    out_file.write_text(json.dumps(result, indent=1), encoding="utf-8")
    _check_coverage(result, strict)
    step("Done", 1.0)
    return result


def region_spectra(audio_path, region, baseline="jfk"):
    """FFT + MFCC for one flaw region, participant vs baseline (same words). For the Spectrum tab."""
    b_audio = baseline_config(baseline)["audio"]
    _, b_frames, _ = load_baseline(baseline)
    p_frames = frame_features(audio_path)
    return {
        "fft_participant": fft_spectrum(audio_path, region["start"], region["end"]),
        "fft_baseline": fft_spectrum(b_audio, region["baseline_start"], region["baseline_end"]),
        "mfcc_participant": mfcc_region(p_frames, region["start"], region["end"]),
        "mfcc_baseline": mfcc_region(b_frames, region["baseline_start"], region["baseline_end"]),
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/flawed/human/flaw_L3_moderate.wav")
    r = analyze(target, use_cache=False)
    print(f"\n{r['file']}: overall {r['overall']}/100, matched words {r['matched_words']}/{len(r['words'])}")
    print("Rubric:", r["rubric"])
    for reg in r["regions"]:
        print(f" [{reg['flaw_type']:<10}] sev {reg['severity']:.1f}  {reg['explanation']}")
