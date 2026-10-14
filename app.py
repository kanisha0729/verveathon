"""OrateLens - objective, explainable feedback on how a speech is delivered.

Run:  streamlit run app.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pipeline.analyze import analyze, region_spectra, BASELINES

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data/labels.csv"
EVAL = ROOT / "data/eval.json"
UPLOADS = ROOT / "results/uploads"
BASELINE_AUDIO = BASELINES["jfk"]["audio"]
DEFAULT_TRANSCRIPT = BASELINES["jfk"]["transcript"].read_text(encoding="utf-8")
DEMO_FILE = "data/flawed/human/flaw_L3_moderate.wav"

FLAW_COLORS = {
    "rushed": "#F5A524", "dragging": "#F5A524", "long_pause": "#8E6CEF",
    "low_volume": "#3E8EF7", "monotone": "#E5484D", "unclear": "#A0A4AE",
}
FLAW_NAMES = {
    "rushed": "Rushed", "dragging": "Dragging", "long_pause": "Long pause",
    "low_volume": "Low volume", "monotone": "Monotone", "unclear": "Unclear",
}
TEAL, GREY = "#2BB3A8", "#8B8D98"
RUBRIC_NAMES = {"pacing": "Pacing", "pausing": "Pausing", "expressiveness": "Expressiveness",
                "volume": "Volume", "clarity": "Clarity"}

st.set_page_config(page_title="OrateLens", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.card {border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:14px 16px; background:rgba(255,255,255,0.03);}
.card .label {font-size:0.8rem; letter-spacing:0.06em; text-transform:uppercase; opacity:0.7;}
.card .value {font-size:1.9rem; font-weight:700; line-height:1.2;}
.big .value {font-size:2.8rem;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:700; color:#111;}
.transcript {line-height:2.1; font-size:1.02rem;}
.transcript span.w {padding:2px 3px; border-radius:4px;}
.pause {display:inline-block; padding:0 6px; margin:0 2px; border-radius:4px; font-size:0.75rem; color:#111;}
</style>
""", unsafe_allow_html=True)


def mmss(t):
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def score_color(v, top):
    frac = v / top
    return "#3FB950" if frac >= 0.7 else ("#F5A524" if frac >= 0.4 else "#E5484D")


@st.cache_data(show_spinner=False)
def load_labels():
    if not LABELS.exists():
        return pd.DataFrame(columns=["file", "source", "severity", "flaw_type", "start", "end"])
    return pd.read_csv(LABELS)


def to_wav(uploaded):
    """Save an uploaded/recorded file as 16 kHz mono wav (works for mp3/m4a too)."""
    import librosa
    import soundfile as sf
    UPLOADS.mkdir(parents=True, exist_ok=True)
    raw = UPLOADS / uploaded.name
    raw.write_bytes(uploaded.getvalue())
    y, _ = librosa.load(str(raw), sr=16000, mono=True)
    out = UPLOADS / (Path(uploaded.name).stem + ".wav")
    sf.write(out, y, 16000)
    return out


def run_analysis(path, transcript):
    with st.status("Analysing the speech...", expanded=True) as status:
        def progress(name, frac):
            status.write(f"{name}...")
        result = analyze(path, transcript, progress=progress)
        status.update(label="Analysis complete", state="complete", expanded=False)
    st.session_state.result = result
    st.session_state.audio_path = str(path)
    st.session_state.play = None


# ---------------------------------------------------------------- header + inputs
st.title("OrateLens")
st.caption("Objective, explainable feedback on how a speech is delivered, compared word by word "
           "with an ideal delivery of the same text.")

labels = load_labels()
c1, c2, c3 = st.columns([1.2, 1.4, 0.8], gap="large")
with c1:
    st.subheader("1. Speech audio")
    mode = st.radio("Source", ["Pick from dataset", "Upload a file", "Record now"], horizontal=True,
                    label_visibility="collapsed")
    chosen_path = None
    if mode == "Pick from dataset":
        files = list(dict.fromkeys(labels["file"].tolist())) if len(labels) else [DEMO_FILE]
        meta = {r.file: f"{r.source} · severity {r.severity}" for r in labels.itertuples()} if len(labels) else {}
        f = st.selectbox("Dataset recording", files,
                         index=files.index(DEMO_FILE) if DEMO_FILE in files else 0,
                         format_func=lambda x: f"{Path(x).stem}  ({meta.get(x, '')})")
        chosen_path = ROOT / f
    elif mode == "Upload a file":
        up = st.file_uploader("Upload participant audio", type=["wav", "mp3", "m4a"])
        if up is not None:
            chosen_path = up
    else:
        rec = st.audio_input("Record the speech")
        if rec is not None:
            rec.name = "recording.wav"
            chosen_path = rec
with c2:
    st.subheader("2. Transcript")
    transcript = st.text_area("Text the speaker reads (must be the same text as the baseline)",
                              DEFAULT_TRANSCRIPT, height=150)
with c3:
    st.subheader("3. Analyse")
    if st.button("Analyse speech", type="primary", width="stretch", disabled=chosen_path is None):
        path = chosen_path if isinstance(chosen_path, Path) else to_wav(chosen_path)
        run_analysis(path, transcript)
    if st.button("Load demo result", width="stretch"):
        run_analysis(ROOT / DEMO_FILE, DEFAULT_TRANSCRIPT)
    st.caption("Baseline: JFK, 1961 inaugural address excerpt.")

st.divider()
result = st.session_state.get("result")
if not result:
    st.info("Pick a recording (or upload / record one) and click **Analyse speech**, "
            "or click **Load demo result** to see where a delivery breaks down.")

# ---------------------------------------------------------------- scores
if result:
    audio_path = st.session_state.audio_path
    regions = result["regions"]
    cols = st.columns([1.3, 1, 1, 1, 1, 1])
    with cols[0]:
        col = score_color(result["overall"], 100)
        st.markdown(f"<div class='card big'><div class='label'>Overall score</div>"
                    f"<div class='value' style='color:{col}'>{result['overall']}<span style='font-size:1rem;opacity:.6'>/100</span></div>"
                    f"<div style='opacity:.7;font-size:.85rem'>{result['file']} · {len(regions)} flaw regions</div></div>",
                    unsafe_allow_html=True)
    for i, (k, name) in enumerate(RUBRIC_NAMES.items()):
        v = result["rubric"][k]
        with cols[i + 1]:
            st.markdown(f"<div class='card'><div class='label'>{name}</div>"
                        f"<div class='value' style='color:{score_color(v, 10)}'>{v}<span style='font-size:.9rem;opacity:.6'>/10</span></div></div>",
                        unsafe_allow_html=True)
    st.write("")

tabs = st.tabs(["📈 Overlay", "🧭 Flaw regions", "🔬 Spectrum", "🗂️ Dataset & evaluation", "⚙️ How it works"])

# ---------------------------------------------------------------- overlay
with tabs[0]:
    if result:
        ov = result["overlay"]
        t = np.array(ov["t"], dtype=float)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Pitch (semitones vs. speaker's median)",
                                            "Loudness (dB vs. speaker's median)", "Speech rate (words/s)"))
        series = [("f0", 1), ("energy", 2), ("rate", 3)]
        for key, row in series:
            p = [np.nan if v is None else v for v in ov[f"{key}_participant"]]
            b = [np.nan if v is None else v for v in ov[f"{key}_baseline"]]
            fig.add_trace(go.Scatter(x=t, y=b, name="Baseline (JFK)", line=dict(color=GREY, dash="dash", width=1.6),
                                     showlegend=(row == 1), legendgroup="b", connectgaps=False), row=row, col=1)
            fig.add_trace(go.Scatter(x=t, y=p, name="Participant", line=dict(color=TEAL, width=2),
                                     showlegend=(row == 1), legendgroup="p", connectgaps=False), row=row, col=1)
        shown = set()
        for r in regions:
            color = FLAW_COLORS.get(r["flaw_type"], "#888")
            fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=color, opacity=0.18, line_width=0)
            fig.add_trace(go.Scatter(
                x=[(r["start"] + r["end"]) / 2], y=[None], mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                name=FLAW_NAMES.get(r["flaw_type"], r["flaw_type"]),
                showlegend=r["flaw_type"] not in shown, legendgroup=r["flaw_type"]), row=1, col=1)
            shown.add(r["flaw_type"])
        # hover strip with explanations along the top chart
        if regions:
            fig.add_trace(go.Scatter(
                x=[(r["start"] + r["end"]) / 2 for r in regions],
                y=[np.nanmax([v for v in ov["f0_participant"] if v is not None] or [0]) + 1.5] * len(regions),
                mode="markers", marker=dict(size=11, color=[FLAW_COLORS.get(r["flaw_type"]) for r in regions],
                                            symbol="triangle-down"),
                hovertext=[f"<b>{FLAW_NAMES.get(r['flaw_type'])}</b><br>" + r["explanation"].replace(". ", ".<br>")
                           for r in regions],
                hoverinfo="text", showlegend=False), row=1, col=1)
        ticks = np.arange(0, (t.max() if len(t) else 1) + 5, 5)
        fig.update_xaxes(tickvals=ticks, ticktext=[mmss(x) for x in ticks], row=3, col=1, title="Time")
        fig.update_layout(height=720, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=60, b=10),
                          hovermode="closest")
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Shaded bands are the detected flaw regions. Hover the triangles for the explanation. "
                   "The baseline is time-warped onto the participant's timeline using the word alignment, "
                   "so both lines show the same words at the same x position.")

# ---------------------------------------------------------------- flaw regions
with tabs[1]:
    if result:
        play = st.session_state.get("play")
        if play:
            st.markdown(f"**Now playing:** {play['label']}")
            st.audio(play["path"], start_time=int(play["start"]))
        order = st.radio("Sort by", ["Time", "Severity"], horizontal=True)
        regs = sorted(regions, key=lambda r: r["start"] if order == "Time" else -r["severity"])
        if not regs:
            st.success("No significant deviations from the baseline were found.")
        for k, r in enumerate(regs):
            color = FLAW_COLORS.get(r["flaw_type"], "#888")
            with st.container(border=True):
                a, b = st.columns([3, 1])
                with a:
                    st.markdown(f"<span class='badge' style='background:{color}'>{FLAW_NAMES.get(r['flaw_type'])}</span> "
                                f"&nbsp; **{mmss(r['start'])} – {mmss(r['end'])}** &nbsp; · &nbsp; "
                                f"<i>“{r['words'][:120]}”</i>", unsafe_allow_html=True)
                    st.write(r["explanation"])
                    m = r["metrics"]
                    st.caption(f"Feature: {m['feature']} · participant {m['participant']} · baseline {m['baseline']} · z = {m['z']}")
                    st.progress(min(1.0, r["severity"] / 5), text=f"Severity {r['severity']:.1f} (mean |z|)")
                with b:
                    if st.button("▶ Play yours", key=f"py{k}", width="stretch"):
                        st.session_state.play = {"path": audio_path, "start": max(0, r["start"] - 0.5),
                                                 "label": f"participant from {mmss(r['start'])}"}
                        st.rerun()
                    if st.button("▶ Play baseline", key=f"pb{k}", width="stretch"):
                        st.session_state.play = {"path": str(BASELINE_AUDIO), "start": max(0, r["baseline_start"] - 0.5),
                                                 "label": f"baseline (same words) from {mmss(r['baseline_start'])}"}
                        st.rerun()

        # highlighted transcript
        st.markdown("#### Transcript with flaws highlighted")
        word_color, pause_at = {}, {}
        for r in regions:
            if r["flaw_type"] == "long_pause":
                pause_at[r["i1"]] = r
            else:
                for i in range(r["i0"], r["i1"] + 1):
                    word_color.setdefault(i, FLAW_COLORS.get(r["flaw_type"]))
        parts = []
        for i, w in enumerate(result["words"]):
            if i in pause_at:
                parts.append(f"<span class='pause' style='background:{FLAW_COLORS['long_pause']}'>⏸ "
                             f"{pause_at[i]['end'] - pause_at[i]['start']:.1f}s</span>")
            c = word_color.get(i)
            style = f"background:{c}40;border-bottom:2px solid {c}" if c else ""
            parts.append(f"<span class='w' style='{style}' title='{mmss(w['start'])}'>{w['word']}</span>")
        st.markdown(f"<div class='transcript'>{' '.join(parts)}</div>", unsafe_allow_html=True)
        st.caption(" · ".join(f"<span style='color:{c}'>■</span> {FLAW_NAMES[k]}" for k, c in FLAW_COLORS.items()
                              if k != "dragging"), unsafe_allow_html=True)

# ---------------------------------------------------------------- spectrum
with tabs[2]:
    if result and regions:
        opts = list(range(len(regions)))
        idx = st.selectbox("Flaw region", opts, format_func=lambda i: f"{mmss(regions[i]['start'])}–{mmss(regions[i]['end'])}  "
                                                                    f"{FLAW_NAMES.get(regions[i]['flaw_type'])}: {regions[i]['words'][:60]}")
        r = regions[idx]

        @st.cache_data(show_spinner="Computing spectra...")
        def spectra(path, start, end, bstart, bend):
            return region_spectra(path, {"start": start, "end": end, "baseline_start": bstart, "baseline_end": bend})

        sp = spectra(audio_path, r["start"], r["end"], r["baseline_start"], r["baseline_end"])
        f = go.Figure()
        f.add_trace(go.Scatter(x=sp["fft_baseline"]["freq"], y=sp["fft_baseline"]["db"], name="Baseline (same words)",
                               line=dict(color=GREY, dash="dash")))
        f.add_trace(go.Scatter(x=sp["fft_participant"]["freq"], y=sp["fft_participant"]["db"], name="Participant",
                               line=dict(color=TEAL)))
        f.update_layout(title="FFT magnitude spectrum of the region", xaxis_title="Frequency (Hz)",
                        yaxis_title="Magnitude (dB)", height=380, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f, width="stretch")
        h1, h2 = st.columns(2)
        for col, key, title in [(h1, "mfcc_participant", "MFCCs: participant"), (h2, "mfcc_baseline", "MFCCs: baseline")]:
            with col:
                hm = go.Figure(go.Heatmap(z=sp[key], colorscale="Tealgrn", showscale=False))
                hm.update_layout(title=title, height=300, xaxis_title="Frame (20 ms)", yaxis_title="MFCC #",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40))
                st.plotly_chart(hm, width="stretch")
        st.caption("Low-pass / muffled speech loses energy above ~2 kHz; quiet speech shifts the whole spectrum down. "
                   "MFCCs summarise the spectral envelope (timbre) frame by frame.")
    elif result:
        st.info("No flaw regions to inspect.")

# ---------------------------------------------------------------- dataset & evaluation
with tabs[3]:
    st.markdown("#### Contrastive dataset")
    st.write("The same JFK text delivered ideally (baseline) and with increasing flaws: four human recordings "
             "and synthetic versions with flaws injected at exactly known times (ground truth).")
    if len(labels):
        f1, f2 = st.columns(2)
        src = f1.multiselect("Source", sorted(labels["source"].unique()), default=sorted(labels["source"].unique()))
        sev = f2.multiselect("Severity", sorted(labels["severity"].unique()), default=sorted(labels["severity"].unique()))
        view = labels[labels["source"].isin(src) & labels["severity"].isin(sev)]
        st.dataframe(view, width="stretch", hide_index=True, height=280)
        pick = st.selectbox("Listen to a dataset file", list(dict.fromkeys(view["file"].tolist())) or ["-"])
        if pick != "-" and (ROOT / pick).exists():
            st.audio(str(ROOT / pick))
    if EVAL.exists():
        ev = json.loads(EVAL.read_text(encoding="utf-8"))
        st.markdown("#### Evaluation against the labels")
        e1, e2 = st.columns([1, 1.2])
        with e1:
            sv = ev["score_by_severity"]
            bar = go.Figure(go.Bar(x=[f"Severity {k}" for k in sv], y=list(sv.values()), marker_color=TEAL,
                                   text=list(sv.values()), textposition="outside"))
            bar.update_layout(title="Mean overall score by severity", yaxis_range=[0, 105], height=340,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, width="stretch")
        with e2:
            rows = [{"Flaw type": FLAW_NAMES.get(k, k), "Labels": v["labels"], "Recall": v["recall"],
                     "Precision": v["precision"], "Mean IoU": v["mean_iou"]} for k, v in ev["per_flaw_type"].items()]
            st.markdown(f"**Temporal grounding** (a hit = IoU ≥ {ev.get('iou_threshold', 0.3)})")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("IoU = overlap between the detected flaw region and the labeled region, divided by their union.")
    else:
        st.info("Run `python pipeline/evaluate.py` to generate data/eval.json.")

# ---------------------------------------------------------------- how it works
with tabs[4]:
    steps = [
        ("1. Inputs", "Participant audio + the transcript of the text they read."),
        ("2. Forced alignment", "faster-whisper word timestamps, matched to the transcript with difflib, so every transcript word gets a start/end time."),
        ("3. Acoustic features", "Every 20 ms: F0 pitch (pYIN), RMS energy, spectral centroid/flatness, 13 MFCCs. Per word: speech rate, pause before (measured from silence), pitch range, loudness, clarity."),
        ("4. Word-by-word comparison", "Word i of the participant vs word i of the baseline. z = (participant − baseline) / spread of the baseline."),
        ("5. Temporal grounding", "Words with |z| > 1.8 (smoothed over 3 words) are merged into flaw regions with exact start/end times."),
        ("6. Explanations + rubric", "Each region gets a numeric, human-readable cause. Rubric per criterion = max(0, 10 − 2·mean|z|); overall = sum × 2."),
    ]
    cols = st.columns(3)
    for i, (title, body) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"<div class='card' style='min-height:150px'><div style='font-weight:700;color:{TEAL}'>{title}</div>"
                        f"<div style='opacity:.85;margin-top:6px'>{body}</div></div>", unsafe_allow_html=True)
            st.write("")
    st.markdown("""
**Why compare against a baseline of the same text?** Good delivery depends on the words: a dramatic pause before
*"ask not"* is a strength, but the same pause in the middle of a phrase is a flaw. Comparing word by word with an
ideal delivery of the same text captures that.

**Why reproducible?** No LLM and no randomness: every score is a fixed formula on z-scores, so the same audio always
gets the same score.

**Fair across voices:** pitch is measured in semitones relative to each speaker's own median, and loudness in dB
relative to each speaker's own level, so a female participant can be compared with a male baseline, and a phone
recording with a studio one.
""")
