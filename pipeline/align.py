"""Forced alignment: give every word of a KNOWN transcript a start/end time in the audio.

How it works
1. faster-whisper transcribes the audio with word-level timestamps.
2. difflib lines up the recognised words with the transcript words (same text, so most match).
3. Transcript words that Whisper missed get times interpolated from their neighbours (conf = 0).

Because every recording is aligned to the SAME transcript, word i in a participant's
recording can be compared directly with word i in the baseline recording.
"""
import re
import difflib
from pathlib import Path

import numpy as np

WHISPER_SIZE = "base"     # change to "tiny" if the laptop is slow
_model = None             # loaded once, reused for every file


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _model


def tokenize(text):
    """Split a transcript into words, dropping punctuation such as the em dash."""
    return re.findall(r"[A-Za-z0-9']+", text)


def norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _audio_duration(audio_path):
    import soundfile as sf
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


def align(audio_path, transcript_text):
    """Return [{word, start, end, conf}] - one entry per transcript word, in order."""
    tokens = tokenize(transcript_text)
    model = _get_model()
    segments, _ = model.transcribe(str(audio_path), language="en", word_timestamps=True, beam_size=5)

    recognised = []  # (normalised word, start, end, probability)
    for seg in segments:
        for w in (seg.words or []):
            n = norm(w.word)
            if n:
                recognised.append((n, float(w.start), float(w.end), float(w.probability)))

    times = [None] * len(tokens)
    a = [norm(t) for t in tokens]
    b = [r[0] for r in recognised]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                _, s, e, p = recognised[j1 + k]
                times[i1 + k] = (s, e, p)
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            # same number of words but recognised differently (e.g. mispronounced): keep the
            # timing, but halve the confidence because the word was not heard as written
            for k in range(i2 - i1):
                _, s, e, p = recognised[j1 + k]
                times[i1 + k] = (s, e, p * 0.5)

    # Fill words Whisper missed by spreading them evenly between known neighbours
    duration = _audio_duration(audio_path)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:  # nothing recognised at all: spread the words over the whole file
        edges = np.linspace(0, duration, len(tokens) + 1)
        return [{"word": tok, "start": round(float(edges[i]), 3), "end": round(float(edges[i + 1]), 3), "conf": 0.0}
                for i, tok in enumerate(tokens)]

    i = 0
    while i < len(tokens):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(tokens) and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else 0.0
        right = times[j][0] if j < len(tokens) else duration
        right = max(right, left + 0.05 * (j - i))
        edges = np.linspace(left, right, (j - i) + 1)
        for k in range(i, j):
            times[k] = (float(edges[k - i]), float(edges[k - i + 1]), 0.0)
        i = j

    out = []
    prev_end = 0.0
    for tok, (s, e, p) in zip(tokens, times):
        s = max(s, prev_end)          # keep times increasing
        e = max(e, s + 0.02)
        out.append({"word": tok, "start": round(s, 3), "end": round(e, 3), "conf": round(float(p), 3)})
        prev_end = e
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    text = (root / "data/transcripts/jfk.txt").read_text(encoding="utf-8")
    for f in ["data/baseline/jfk_baseline.wav", "data/flawed/human/flaw_L3_moderate.wav"]:
        words = align(root / f, text)
        matched = sum(1 for w in words if w["conf"] > 0)
        print(f"\n{f}: {len(words)} words, {matched} matched")
        for w in words[:12]:
            print(f"  {w['word']:<14} {w['start']:6.2f} - {w['end']:6.2f}  conf {w['conf']:.2f}")
