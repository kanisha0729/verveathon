"""Builds the synthetic half of the contrastive dataset and writes data/labels.csv.

From the baseline recording we inject delivery flaws into chosen sentences at 4 severity levels
(1 = near perfect ... 4 = egregious). Because the code injects the flaw, its exact start/end time
in the output file is known - these are the ground-truth labels used by pipeline/evaluate.py.

Run:  python data/make_synthetic.py
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.align import align, norm  # noqa: E402

SR = 16000
BASELINE = ROOT / "data/baseline/jfk_baseline.wav"
TRANSCRIPT = ROOT / "data/transcripts/jfk.txt"
HUMAN_DIR = ROOT / "data/flawed/human"
OUT_DIR = ROOT / "data/flawed/synthetic"
LABELS = ROOT / "data/labels.csv"

# flaw type -> parameter for severity 1..4
PARAMS = {
    "rushed":     [1.15, 1.35, 1.6, 2.0],     # time-stretch rate (higher = faster)
    "long_pause": [0.8, 1.5, 2.5, 4.0],       # inserted silence in seconds
    "low_volume": [4, 8, 14, 20],             # dB reduction
    "unclear":    [3000, 2000, 1200, 800],    # low-pass cut-off in Hz (+ noise)
}
# which sentence (0-based) each single-flaw file damages
SENTENCE_FOR = {"rushed": 3, "long_pause": 2, "low_volume": 1, "unclear": 4}
# severity-4 files that combine several flaws: list of (flaw type, sentence)
MIXED = {
    "syn_mixed_a_s4": [("low_volume", 0), ("long_pause", 2), ("rushed", 3)],
    "syn_mixed_b_s4": [("rushed", 1), ("unclear", 3), ("long_pause", 4)],
}
# Nive's human recordings: file -> (severity, [(flaw type, sentence list or "all")])
# Edit these to match what Nive actually did in each recording.
HUMAN_LABELS = {
    "flaw_L1_near_perfect.wav": (1, []),
    "flaw_L2_mild.wav": (2, [("rushed", [3])]),
    "flaw_L3_moderate.wav": (3, [("monotone", "all"), ("long_pause", [2])]),
    "flaw_L4_egregious.wav": (4, [("rushed", "all"), ("low_volume", "all"), ("unclear", "all")]),
}


def sentence_spans(words, transcript):
    """Map aligned words to sentences; return [(start, end, first_word_idx, last_word_idx)]."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", transcript.strip()) if s]
    sent_of_token = []
    for si, s in enumerate(sentences):
        sent_of_token += [si] * len([t for t in re.findall(r"[A-Za-z0-9']+", s) if norm(t)])
    words = [w for w in words if norm(w["word"])]
    if len(sent_of_token) != len(words):   # safety: proportional mapping
        sent_of_token = [min(len(sentences) - 1, int(i * len(sentences) / len(words))) for i in range(len(words))]
    spans = []
    for si in range(len(sentences)):
        idx = [i for i, s in enumerate(sent_of_token) if s == si]
        spans.append((words[idx[0]]["start"], words[idx[-1]]["end"], idx[0], idx[-1]))
    return spans, words


def apply_flaw(y, ftype, sev, span, words):
    """Return (new audio, (label_start, label_end), (position, length_change))."""
    s, e = int(span[0] * SR), int(span[1] * SR)
    p = PARAMS[ftype][sev - 1]
    if ftype == "rushed":
        fast = librosa.effects.time_stretch(y[s:e], rate=p)
        return np.concatenate([y[:s], fast, y[e:]]), (s / SR, (s + len(fast)) / SR), (e, len(fast) - (e - s))
    if ftype == "long_pause":
        mid = (span[2] + span[3]) // 2                    # pause after the middle word of the sentence
        pos = int(words[mid]["end"] * SR)
        sil = np.zeros(int(p * SR), dtype=y.dtype)
        return np.concatenate([y[:pos], sil, y[pos:]]), (pos / SR, (pos + len(sil)) / SR), (pos, len(sil))
    if ftype == "low_volume":
        y = y.copy()
        y[s:e] *= 10 ** (-p / 20)
        return y, (s / SR, e / SR), (e, 0)
    if ftype == "unclear":
        y = y.copy()
        sos = butter(6, p, btype="low", fs=SR, output="sos")
        seg = sosfiltfilt(sos, y[s:e])
        rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
        rng = np.random.default_rng(42)
        seg = seg + rng.normal(0, rms * 0.15 * sev, len(seg))
        y[s:e] = seg
        return y, (s / SR, e / SR), (e, 0)
    raise ValueError(ftype)


def build(y, plan, spans, words):
    """Apply several flaws, keeping earlier labels correct when later edits shift time."""
    labels = []
    for ftype, si, sev in sorted(plan, key=lambda x: -spans[x[1]][0]):   # latest sentence first
        y, (ls, le), (pos, delta) = apply_flaw(y, ftype, sev, spans[si], words)
        # an edit that changes the length shifts every label recorded after it
        shift = delta / SR
        for lab in labels:
            if lab[1] >= pos / SR - 1e-6:
                lab[1] = round(lab[1] + shift, 3)
                lab[2] = round(lab[2] + shift, 3)
        labels.append([ftype, round(ls, 3), round(le, 3)])
    return y, sorted(labels, key=lambda x: x[1])


def main():
    transcript = TRANSCRIPT.read_text(encoding="utf-8")
    y, _ = librosa.load(str(BASELINE), sr=SR, mono=True)
    print("Aligning baseline...")
    spans, words = sentence_spans(align(BASELINE, transcript), transcript)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [["data/baseline/jfk_baseline.wav", "baseline", 0, "none", "", ""]]
    for ftype, sev_params in PARAMS.items():
        for sev in range(1, 5):
            out, labels = build(y, [(ftype, SENTENCE_FOR[ftype], sev)], spans, words)
            name = f"syn_{ftype}_s{sev}.wav"
            sf.write(OUT_DIR / name, out, SR)
            for lt, ls, le in labels:
                rows.append([f"data/flawed/synthetic/{name}", "synthetic", sev, lt, ls, le])
    for name, plan in MIXED.items():
        out, labels = build(y, [(f, s, 4) for f, s in plan], spans, words)
        sf.write(OUT_DIR / f"{name}.wav", out, SR)
        for lt, ls, le in labels:
            rows.append([f"data/flawed/synthetic/{name}.wav", "synthetic", 4, lt, ls, le])

    for fname, (sev, flaws) in HUMAN_LABELS.items():
        path = HUMAN_DIR / fname
        if not path.exists():
            print("  (missing, skipped)", fname)
            continue
        print("Aligning", fname)
        h_spans, _ = sentence_spans(align(path, transcript), transcript)
        if not flaws:
            rows.append([f"data/flawed/human/{fname}", "human", sev, "none", "", ""])
        for ftype, which in flaws:
            chosen = range(len(h_spans)) if which == "all" else which
            for si in chosen:
                rows.append([f"data/flawed/human/{fname}", "human", sev, ftype,
                             round(h_spans[si][0], 3), round(h_spans[si][1], 3)])

    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "severity", "flaw_type", "start", "end"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} label rows to {LABELS}")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
