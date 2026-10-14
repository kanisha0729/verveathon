"""Step 1: word-level alignment of a recording to its transcript."""
import difflib
import re
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

_MODEL = None  # Whisper model, loaded once and reused


def _get_model():
    """Load faster-whisper "base" on CPU (int8) the first time it is needed."""
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _MODEL


def _norm(word):
    """Lowercase and strip punctuation so 'Freedom,' matches 'freedom'."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def align(audio_path, transcript_text):
    """Return [{word, start, end, conf}], one entry per transcript word, in order."""
    # Transcript words (drop tokens with no letters/digits, e.g. a lone dash)
    words = [w for w in transcript_text.split() if _norm(w)]

    # 1. Recognise the audio with word timestamps; the transcript is the prompt.
    #    Audio is loaded with librosa (16 kHz mono) and passed as an array,
    #    because faster-whisper's own decoder breaks with the installed PyAV.
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    segments, info = _get_model().transcribe(
        audio, language="en", word_timestamps=True,
        initial_prompt=transcript_text,
    )
    heard = [w for seg in segments for w in seg.words if _norm(w.word)]

    # 2. Map recognised words onto transcript words
    out = [{"word": w, "start": None, "end": None, "conf": 0.0} for w in words]
    matcher = difflib.SequenceMatcher(
        None, [_norm(w) for w in words], [_norm(h.word) for h in heard], autojunk=False
    )
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            h = heard[block.b + k]
            out[block.a + k].update(start=h.start, end=h.end, conf=h.probability)

    # 3. Unmatched words: spread them evenly between the neighbouring matched words
    i = 0
    while i < len(out):
        if out[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j]["start"] is None:
            j += 1  # out[i:j] is a run of unmatched words
        t0 = out[i - 1]["end"] if i > 0 else 0.0
        t1 = out[j]["start"] if j < len(out) else info.duration
        step = max(t1 - t0, 0.0) / (j - i)
        for k in range(i, j):
            out[k]["start"] = t0 + (k - i) * step
            out[k]["end"] = t0 + (k - i + 1) * step
        i = j

    # Round for readable output
    for w in out:
        w["start"], w["end"], w["conf"] = round(w["start"], 3), round(w["end"], 3), round(w["conf"], 3)
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    text = (root / "data/transcripts/jfk.txt").read_text(encoding="utf-8")
    for rel in ["data/baseline/jfk_baseline.wav", "data/flawed/human/flaw_L3_moderate.wav"]:
        result = align(root / rel, text)
        matched = sum(1 for w in result if w["conf"] > 0)
        print(f"\n{rel}: {len(result)} words, {matched} matched")
        for w in result[:12]:
            print(f"  {w['word']:<14} {w['start']:7.2f} - {w['end']:7.2f}  conf {w['conf']:.2f}")
