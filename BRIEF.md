# BRIEF.md: Verveathon 2K26, PS1 (re-read before every task)

## Product
OrateLens: an objective, reproducible speech delivery evaluator. It compares a participant's delivery of a text with an ideal delivery of the SAME text, finds the exact time regions where the delivery fails, explains each flaw with numbers, and gives rubric scores in an interactive dashboard.

## Team and rules
- Kanisha owns data/ and pipeline/. Nive owns app.py, dummy_result.json and .streamlit/.
- Windows, CPU only. Simple, readable Python with short comments, because we must explain every part to judges.
- One task at a time. Only create or edit the files the task names. No extra features.
- No LLM. All scoring is deterministic and reproducible.
- After each task, explain the code in short bullet points.

## Dataset (data/)
- data/transcripts/jfk.txt: the JFK inaugural address excerpt
- data/baseline/jfk_baseline.wav: JFK's real delivery (16 kHz mono)
- data/flawed/human/: Nive's recordings flaw_L1_near_perfect, flaw_L2_mild, flaw_L3_moderate, flaw_L4_egregious (16 kHz mono wav)
- data/flawed/synthetic/: generated from the baseline with flaws injected at known times, severity 1 (near perfect) to 4 (egregious)
- data/labels.csv columns: file, source (baseline/human/synthetic), severity (0-4, baseline = 0), flaw_type, start, end (seconds in that file)
- Flaw types: rushed, dragging, long_pause, low_volume, monotone, unclear

## Pipeline (pipeline/)
1. align.py: align(audio_path, transcript_text) -> [{word, start, end, conf}], one entry per transcript word in order. Method: faster-whisper "base" (cpu, int8, word_timestamps=True, transcript passed as initial_prompt), then map the recognised words onto the transcript words with difflib.SequenceMatcher. Unmatched words get interpolated times and conf 0.
2. features.py:
   - frame_features(audio_path): every 20 ms: f0 (librosa.pyin, 65-400 Hz), rms_db, spectral_centroid, spectral_flatness, 13 MFCCs, plus "t"
   - word_features(frames, words): per word: duration, pause_before, local_rate (words per second over a 5-word window), f0_mean, f0_range (max-min of voiced f0 over a 5-word window), energy_db (mean rms_db), clarity (= conf)
   - fft_spectrum(audio_path, start, end): FFT magnitude in dB vs frequency for a region
3. compare.py: word i of the participant <-> word i of the baseline (same transcript).
   - For each word feature: z = (participant - baseline) / (std of that feature across all baseline words + small epsilon)
   - Flag rules: local_rate z > 2 -> rushed; local_rate z < -2 -> dragging; pause_before z > 2 -> long_pause; energy_db z < -2 -> low_volume; f0_range z < -2 -> monotone; clarity z < -2 -> unclear
   - Merge neighbouring flagged words of the same type (gap of 1 word or less) into regions: {start, end, words, flaw_type, severity (mean |z|), metrics {participant, baseline, z}, explanation}
   - Explanation template: "0:12-0:18 ('The energy, the faith...'): speaking rate 4.2 words/s vs baseline 2.1 (2.0x faster, z = 3.4). Rushing removes emphasis from the key phrase."
   - Rubric (each 0-10): pacing (local_rate), pausing (pause_before), expressiveness (f0_range), volume (energy_db), clarity (clarity). Score = max(0, 10 - 2 x mean |z| over all words). Overall = sum of the 5 scores x 2 (0-100).
   - Overlay: warp baseline frame features onto the participant timeline using the word alignment (linear inside each word) so both can be plotted on one time axis.
4. analyze.py: analyze(audio_path, transcript_text, baseline="jfk") -> result dict below. Save to results/[file name].json and reuse it if it exists. Cache the baseline's features.
5. evaluate.py: run analyze() on every labeled file; report the overall score per severity, and temporal IoU, recall and precision per flaw type.

## Result format
{
  "duration": float,
  "words": [{"word", "start", "end", "conf"}],
  "overlay": {"t": [...], "f0_participant": [...], "f0_baseline": [...], "energy_participant": [...], "energy_baseline": [...], "rate_participant": [...], "rate_baseline": [...]},
  "regions": [{"start", "end", "words", "flaw_type", "severity", "metrics": {"participant", "baseline", "z"}, "explanation"}],
  "rubric": {"pacing", "pausing", "expressiveness", "volume", "clarity"},
  "overall": int
}

## UI
Dark theme. Participant line teal #2BB3A8, baseline line dashed grey #8B8D98.
Flaw colours: rushed/dragging amber #F5A524, long_pause purple #8E6CEF, low_volume blue #3E8EF7, monotone red #E5484D, unclear grey #A0A4AE.
