"""Compare a participant's delivery with the baseline delivery of the SAME transcript.

Word i of the participant <-> word i of the baseline.
For every word feature:   z = (participant - baseline) / spread of that feature across the baseline
A word is flagged when |z| > Z_THRESHOLD in the direction that means a flaw.
Neighbouring flagged words of the same flaw type are merged into a time region (temporal grounding),
and every region gets a human-readable explanation built from its numbers (causal explainability).
Rubric scores are fixed formulas on the z-scores, so results are fully reproducible (no LLM, no randomness).
"""
import numpy as np

Z_THRESHOLD = 1.8
Z_CLIP = 5.0          # cap extreme z so one outlier can't wreck a score
SMOOTH_WORDS = 3      # z is averaged over 3 neighbouring words before flagging (reduces one-word noise)
MERGE_GAP = 3         # flagged words at most this many words apart join into one region
MIN_WORDS = 2         # regions shorter than this are ignored (except pauses, which are single events)

# feature -> list of (flaw type, direction). direction +1 = too high is a flaw, -1 = too low is a flaw
FLAW_RULES = {
    "local_rate":   [("rushed", +1), ("dragging", -1)],
    "pause_before": [("long_pause", +1)],
    "energy_db":    [("low_volume", -1)],
    "f0_range":     [("monotone", -1)],
    "clarity":      [("unclear", -1)],
}
# minimum spread per feature, so a very steady baseline doesn't make tiny differences look huge
MIN_SPREAD = {"local_rate": 0.4, "pause_before": 0.15, "energy_db": 2.0, "f0_range": 1.0, "clarity": 0.08}

RUBRIC = {   # rubric criterion -> feature
    "pacing": "local_rate",
    "pausing": "pause_before",
    "expressiveness": "f0_range",
    "volume": "energy_db",
    "clarity": "clarity",
}


def mmss(t):
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def zscores(p_feats, b_feats):
    """Return {feature: array of z per word}."""
    n = min(len(p_feats), len(b_feats))
    z = {}
    for feat in FLAW_RULES:
        b = np.array([f[feat] for f in b_feats[:n]], dtype=float)
        p = np.array([f[feat] for f in p_feats[:n]], dtype=float)
        spread = max(float(np.std(b)), MIN_SPREAD[feat])
        z[feat] = np.clip((p - b) / spread, -Z_CLIP, Z_CLIP)
    return z


def _explain(ftype, start, end, words_text, first_word, p, b, z):
    span = f"{mmss(start)}-{mmss(end)} ('{words_text}')"
    if ftype == "rushed":
        return (f"{span}: speaking rate {p:.1f} words/s vs baseline {b:.1f} "
                f"({p / max(b, 0.1):.1f}x faster, z = {z:+.1f}). Rushing removes emphasis from the key phrase.")
    if ftype == "dragging":
        return (f"{span}: speaking rate {p:.1f} words/s vs baseline {b:.1f} "
                f"({b / max(p, 0.1):.1f}x slower, z = {z:+.1f}). Dragging loses the audience's attention.")
    if ftype == "long_pause":
        return (f"{span}: pause of {p:.2f} s before '{first_word}' vs baseline {b:.2f} s (z = {z:+.1f}). "
                f"An unplanned pause mid-phrase breaks the flow.")
    if ftype == "low_volume":
        return (f"{span}: loudness {p:+.1f} dB vs baseline {b:+.1f} dB, each relative to the speaker's own average "
                f"(z = {z:+.1f}). The phrase becomes hard to hear.")
    if ftype == "monotone":
        return (f"{span}: pitch range {p:.1f} semitones vs baseline {b:.1f} (z = {z:+.1f}). "
                f"A flat pitch makes the delivery sound monotonous.")
    if ftype == "unclear":
        return (f"{span}: recognition confidence {p:.2f} vs baseline {b:.2f} (z = {z:+.1f}). "
                f"Words are not articulated clearly enough to be recognised.")
    return span


def _smooth(x, k=SMOOTH_WORDS):
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def find_regions(p_words, b_words, p_feats, b_feats, z):
    n = min(len(p_words), len(b_words))
    regions = []
    for feat, rules in FLAW_RULES.items():
        zf = z[feat][:n] if feat == "pause_before" else _smooth(z[feat][:n])
        for ftype, direction in rules:
            flagged = [i for i in range(n) if direction * zf[i] > Z_THRESHOLD]
            if feat == "pause_before":
                flagged = [i for i in flagged if i > 0]
            groups = []
            gap = 1 if ftype == "long_pause" else MERGE_GAP
            for i in flagged:
                if groups and i - groups[-1][-1] <= gap:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            for g in groups:
                i0, i1 = g[0], g[-1]
                if ftype != "long_pause" and (i1 - i0 + 1) < MIN_WORDS:
                    continue
                g = list(range(i0, i1 + 1)) if ftype != "long_pause" else g
                if ftype == "long_pause":
                    # the flaw IS the silence itself (measured from the audio energy)
                    start, end = p_feats[i0]["pause_start"], p_feats[i1]["pause_end"]
                    b_start, b_end = b_words[i0 - 1]["end"], max(b_words[i1]["start"], b_words[i0 - 1]["end"] + 0.1)
                else:
                    start, end = p_words[i0]["start"], p_words[i1]["end"]
                    b_start, b_end = b_words[i0]["start"], b_words[i1]["end"]
                agg = np.max if ftype == "long_pause" else np.mean   # a pause can be split across 2 words
                pv = float(agg([p_feats[i][feat] for i in g]))
                bv = float(np.mean([b_feats[i][feat] for i in g]))
                zv = float(agg([z[feat][i] for i in g]))
                if ftype == "long_pause":
                    pv = max(pv, float(end - start))     # total silence of the merged region
                ws = [p_words[i]["word"] for i in range(i0, i1 + 1)]
                text = " ".join(ws[:5]) + ("..." if len(ws) > 5 else "")
                regions.append({
                    "start": round(start, 2), "end": round(max(end, start + 0.05), 2),
                    "baseline_start": round(b_start, 2), "baseline_end": round(max(b_end, b_start + 0.05), 2),
                    "i0": i0, "i1": i1,
                    "words": " ".join(ws),
                    "flaw_type": ftype,
                    "severity": round(float(np.mean([abs(z[feat][i]) for i in g])), 2),
                    "metrics": {"feature": feat, "participant": round(pv, 3), "baseline": round(bv, 3), "z": round(zv, 2)},
                    "explanation": _explain(ftype, start, end, text, ws[-1] if ftype == "long_pause" else ws[0], pv, bv, zv),
                })
    regions.sort(key=lambda r: r["start"])
    return regions


def rubric_scores(z):
    scores = {}
    for crit, feat in RUBRIC.items():
        mean_abs = float(np.mean(np.abs(z[feat]))) if len(z[feat]) else 0.0
        scores[crit] = round(max(0.0, 10.0 - 2.0 * mean_abs), 1)
    overall = int(round(sum(scores.values()) * 2))
    return scores, overall


def _knots(words):
    k = []
    for w in words:
        k.extend([w["start"], w["end"]])
    return np.maximum.accumulate(np.array(k, dtype=float))


def overlay(p_frames, b_frames, p_words, b_words, p_feats, b_feats, step=5):
    """Warp baseline features onto the participant timeline (linear inside each word / gap)."""
    n = min(len(p_words), len(b_words))
    kp, kb = _knots(p_words[:n]), _knots(b_words[:n])
    t = p_frames["t"][::step]
    tb = np.interp(t, kp, kb)
    hop = b_frames["t"][1] - b_frames["t"][0]
    bidx = np.clip(np.round(tb / hop).astype(int), 0, len(b_frames["t"]) - 1)
    pidx = np.arange(0, len(p_frames["t"]), step)[:len(t)]

    starts = np.array([w["start"] for w in p_words[:n]])
    widx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, n - 1)

    def smooth(x, k=3):
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k) / k, mode="same")

    def clean(arr, nd=2):
        return [None if (v is None or np.isnan(v)) else round(float(v), nd) for v in arr]

    return {
        "t": clean(t),
        "f0_unit": "semitones relative to each speaker's median pitch",
        "f0_participant": clean(p_frames["f0_st"][pidx]),
        "f0_baseline": clean(b_frames["f0_st"][bidx]),
        "energy_participant": clean(smooth(p_frames["energy_rel"][pidx])),
        "energy_baseline": clean(smooth(b_frames["energy_rel"][bidx])),
        "rate_participant": clean([p_feats[i]["local_rate"] for i in widx]),
        "rate_baseline": clean([b_feats[i]["local_rate"] for i in widx]),
    }
