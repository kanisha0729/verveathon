"""OrateLens - objective, explainable feedback on how a speech is delivered.

Run:  streamlit run app.py
Demo logins:  Participant -> any name        Judge -> access code VERVE2026
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pipeline.analyze import analyze, region_spectra, BASELINES
try:   # newer pipeline/analyze.py supports custom baselines; older versions still run
    from pipeline.analyze import custom_baseline
except ImportError:
    custom_baseline = None

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data/labels.csv"
EVAL = ROOT / "data/eval.json"
UPLOADS = ROOT / "results/uploads"
ATTEMPTS = ROOT / "results/attempts.json"
BASELINE_AUDIO = BASELINES["jfk"]["audio"]
DEFAULT_TRANSCRIPT = BASELINES["jfk"]["transcript"].read_text(encoding="utf-8")
DEMO_FILE = "data/flawed/human/flaw_L3_moderate.wav"
JUDGE_CODE = "VERVE2026"

FLAW_COLORS = {"rushed": "#F5A524", "dragging": "#F59E0B", "long_pause": "#8E6CEF",
               "low_volume": "#3E8EF7", "monotone": "#F0506E", "unclear": "#A0A4AE"}
FLAW_NAMES = {"rushed": "Rushed", "dragging": "Dragging", "long_pause": "Long pause",
              "low_volume": "Low volume", "monotone": "Monotone", "unclear": "Unclear"}
TEAL, VIOLET, GREY = "#2DD4BF", "#8B5CF6", "#8B8FA3"
RUBRIC = {"pacing": "Pacing", "pausing": "Pausing", "expressiveness": "Expressiveness",
          "volume": "Volume", "clarity": "Clarity"}
TIPS = {
    "pacing": "Slow down on the key phrases. Aim to stay within ±20% of the baseline speaking rate.",
    "pausing": "Pause at punctuation and before important lines, never in the middle of a phrase.",
    "expressiveness": "Vary your pitch: lift on the words that matter and let sentences fall at the end.",
    "volume": "Project steadily. Don't let your voice drop at the start or end of sentences.",
    "clarity": "Open your mouth and finish your consonants, especially on long words.",
}
DEMO_SPEAKERS = {  # dataset recordings shown as sample participants on the judge board
    "data/flawed/human/flaw_L1_near_perfect.wav": "Speaker A",
    "data/flawed/human/flaw_L2_mild.wav": "Speaker B",
    "data/flawed/human/flaw_L3_moderate.wav": "Speaker C",
    "data/flawed/human/flaw_L4_egregious.wav": "Speaker D",
}

st.set_page_config(page_title="OrateLens", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', 'Plus Jakarta Sans', sans-serif !important; letter-spacing: -0.01em; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; height: 2.5rem; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(139,92,246,0.18), transparent 60%),
                      radial-gradient(900px 500px at 100% 0%, rgba(45,212,191,0.14), transparent 60%), #0B1020; }
.block-container { padding-top: 3.6rem !important; max-width: 1350px; }
.brand { line-height: 1.6; padding-top: 4px; }
section[data-testid="stSidebar"] { background: rgba(15,20,38,0.92); border-right: 1px solid rgba(255,255,255,0.06); }
.glass { border: 1px solid rgba(255,255,255,0.09); border-radius: 18px; padding: 18px 20px;
         background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); }
.kpi .label { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #9AA3B8; font-weight: 700; }
.kpi .value { font-family: 'Sora', sans-serif; font-size: 2rem; font-weight: 700; line-height: 1.15; }
.kpi.hero .value { font-size: 3.2rem; }
.kpi .sub { color: #9AA3B8; font-size: .85rem; }
.brand { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.5rem;
         background: linear-gradient(90deg, #2DD4BF, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-title { font-family: 'Sora', sans-serif; font-weight: 800; font-size: clamp(2.2rem, 4.2vw, 3.6rem); line-height: 1.05; margin: .3rem 0 .8rem; }
.hero-title span { background: linear-gradient(90deg, #2DD4BF, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { color: #B6BED0; font-size: 1.08rem; max-width: 34rem; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700;
        border: 1px solid rgba(255,255,255,0.12); color: #CFD6E6; margin: 0 6px 6px 0; background: rgba(255,255,255,0.04); }
.badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 800; color: #0B1020; }
.feature { display: flex; gap: 12px; align-items: flex-start; margin: 14px 0; }
.feature .ico { font-size: 1.3rem; width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
                background: rgba(45,212,191,0.12); flex: none; }
.feature b { display: block; } .feature span { color: #9AA3B8; font-size: .92rem; }
.transcript { line-height: 2.2; font-size: 1.02rem; }
.transcript span.w { padding: 2px 4px; border-radius: 5px; }
.pause { display: inline-block; padding: 0 7px; margin: 0 2px; border-radius: 5px; font-size: .75rem; color: #0B1020; font-weight: 700; }
.section-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.35rem; margin: .2rem 0 .6rem; }
.muted { color: #9AA3B8; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; padding: 10px 16px; }
div.stButton > button { border-radius: 12px; font-weight: 700; }
.rank { font-family: 'Sora', sans-serif; font-size: 1.4rem; font-weight: 800; color: #8B5CF6; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- helpers
def mmss(t):
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def score_color(v, top):
    f = v / top
    return "#34D399" if f >= 0.7 else ("#F5A524" if f >= 0.4 else "#F0506E")


def kpi(label, value, top, sub="", hero=False):
    c = score_color(value, top)
    return (f"<div class='glass kpi {'hero' if hero else ''}'><div class='label'>{label}</div>"
            f"<div class='value' style='color:{c}'>{value}<span style='font-size:.9rem;color:#9AA3B8'>/{top}</span></div>"
            f"<div class='sub'>{sub}</div></div>")


def plotly_dark(fig, height):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans, sans-serif", color="#D7DCE8"),
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_labels():
    if not LABELS.exists():
        return pd.DataFrame(columns=["file", "source", "severity", "flaw_type", "start", "end"])
    return pd.read_csv(LABELS)


def load_attempts():
    try:
        return json.loads(ATTEMPTS.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_attempt(name, result, audio_path, baseline="jfk"):
    attempts = load_attempts()
    attempts.append({"participant": name, "file": result["file"], "audio_path": str(audio_path),
                     "time": datetime.now().strftime("%d %b %H:%M"), "ts": time.time(),
                     "overall": result["overall"], "rubric": result["rubric"],
                     "regions": len(result["regions"]), "baseline": baseline,
                     "baseline_label": result.get("baseline_label", "JFK, 1961 inaugural address"),
                     "flaw_counts": {k: sum(1 for r in result["regions"] if r["flaw_type"] == k) for k in FLAW_NAMES}})
    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.write_text(json.dumps(attempts, indent=1), encoding="utf-8")


def to_wav(uploaded, prefix=""):
    import librosa
    import soundfile as sf
    UPLOADS.mkdir(parents=True, exist_ok=True)
    raw = UPLOADS / uploaded.name
    raw.write_bytes(uploaded.getvalue())
    y, _ = librosa.load(str(raw), sr=16000, mono=True)
    out = UPLOADS / f"{prefix}{Path(uploaded.name).stem}_{int(time.time())}.wav"
    sf.write(out, y, 16000)
    return out


def run_analysis(path, transcript, baseline="jfk"):
    with st.status("Analysing the speech...", expanded=True) as status:
        result = analyze(path, transcript, baseline=baseline, progress=lambda name, frac: status.write(f"✓ {name}"))
        status.update(label="Analysis complete", state="complete", expanded=False)
    return result


def radar(series, height=380):
    cats = list(RUBRIC.values())
    fig = go.Figure()
    palette = [TEAL, VIOLET, "#F5A524", "#F0506E", "#3E8EF7", "#34D399"]
    for i, (name, rub) in enumerate(series):
        vals = [rub[k] for k in RUBRIC] + [rub[list(RUBRIC)[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], name=name, fill="toself",
                                      line=dict(color=palette[i % len(palette)], width=2), opacity=0.75))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 10], gridcolor="rgba(255,255,255,0.1)"),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
                      legend=dict(orientation="h", y=-0.1))
    return plotly_dark(fig, height)


# ------------------------------------------------------------------------------------------- analysis views
def render_scores(result):
    cols = st.columns([1.4, 1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(kpi("Overall score", result["overall"], 100,
                        f"{len(result['regions'])} flaw regions found", hero=True), unsafe_allow_html=True)
    for i, (k, name) in enumerate(RUBRIC.items()):
        with cols[i + 1]:
            st.markdown(kpi(name, result["rubric"][k], 10), unsafe_allow_html=True)


def render_overlay(result):
    ov, regions = result["overlay"], result["regions"]
    blabel = "Baseline (" + result.get("baseline_label", "JFK") .split(",")[0] + ")"
    t = np.array(ov["t"], dtype=float)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Pitch (semitones vs. speaker's median)", "Loudness (dB vs. speaker's median)",
                                        "Speech rate (words / s)"))
    for key, row in [("f0", 1), ("energy", 2), ("rate", 3)]:
        p = [np.nan if v is None else v for v in ov[f"{key}_participant"]]
        b = [np.nan if v is None else v for v in ov[f"{key}_baseline"]]
        fig.add_trace(go.Scatter(x=t, y=b, name=blabel, line=dict(color=GREY, dash="dash", width=1.5),
                                 showlegend=row == 1, legendgroup="b"), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=p, name="Participant", line=dict(color=TEAL, width=2.2),
                                 showlegend=row == 1, legendgroup="p"), row=row, col=1)
    shown = set()
    for r in regions:
        c = FLAW_COLORS.get(r["flaw_type"], "#888")
        fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=c, opacity=0.2, line_width=0)
        if r["flaw_type"] not in shown:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=c, size=11, symbol="square"),
                                     name=FLAW_NAMES[r["flaw_type"]]), row=1, col=1)
            shown.add(r["flaw_type"])
    if regions:
        top = max([v for v in ov["f0_participant"] if v is not None] or [0]) + 1.5
        fig.add_trace(go.Scatter(x=[(r["start"] + r["end"]) / 2 for r in regions], y=[top] * len(regions), mode="markers",
                                 marker=dict(size=12, symbol="triangle-down",
                                             color=[FLAW_COLORS.get(r["flaw_type"]) for r in regions]),
                                 hovertext=[f"<b>{FLAW_NAMES[r['flaw_type']]}</b><br>" + r["explanation"].replace(". ", ".<br>")
                                            for r in regions], hoverinfo="text", showlegend=False), row=1, col=1)
    ticks = np.arange(0, (t.max() if len(t) else 1) + 5, 5)
    fig.update_xaxes(tickvals=ticks, ticktext=[mmss(x) for x in ticks], row=3, col=1)
    plotly_dark(fig, 720).update_layout(legend=dict(orientation="h", y=1.09), hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded bands = detected flaw regions · hover the ▼ markers for the explanation · the baseline is "
               "time-warped onto your timeline using the word alignment, so both lines show the same words at the same x.")


def render_regions(result, audio_path, key):
    regions = result["regions"]
    play = st.session_state.get(f"play_{key}")
    if play:
        st.markdown(f"**▶ Now playing:** {play['label']}")
        st.audio(play["path"], start_time=int(play["start"]))
    order = st.radio("Sort by", ["Time", "Severity"], horizontal=True, key=f"sort_{key}")
    regs = sorted(regions, key=lambda r: r["start"] if order == "Time" else -r["severity"])
    if not regs:
        st.success("No significant deviations from the baseline. Excellent delivery!")
    for k, r in enumerate(regs):
        c = FLAW_COLORS.get(r["flaw_type"], "#888")
        with st.container(border=True):
            a, b = st.columns([4, 1])
            with a:
                st.markdown(f"<span class='badge' style='background:{c}'>{FLAW_NAMES[r['flaw_type']]}</span>&nbsp;&nbsp;"
                            f"<b>{mmss(r['start'])} – {mmss(r['end'])}</b>&nbsp;&nbsp;<span class='muted'>“{r['words'][:110]}”</span>",
                            unsafe_allow_html=True)
                st.markdown(r["explanation"])
                m = r["metrics"]
                st.caption(f"{m['feature']} · you {m['participant']} · baseline {m['baseline']} · z = {m['z']}")
                st.progress(min(1.0, r["severity"] / 5), text=f"Severity {r['severity']:.1f}")
            with b:
                if st.button("▶ Play yours", key=f"py_{key}_{k}", width="stretch"):
                    st.session_state[f"play_{key}"] = {"path": audio_path, "start": max(0, r["start"] - 0.5),
                                                       "label": f"you, from {mmss(r['start'])}"}
                    st.rerun()
                if st.button("▶ Play baseline", key=f"pb_{key}_{k}", width="stretch"):
                    st.session_state[f"play_{key}"] = {"path": result.get("baseline_audio", str(BASELINE_AUDIO)), "start": max(0, r["baseline_start"] - 0.5),
                                                       "label": f"baseline (same words), from {mmss(r['baseline_start'])}"}
                    st.rerun()
    st.markdown("<div class='section-title'>Transcript with flaws highlighted</div>", unsafe_allow_html=True)
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
        style = f"background:{c}33;border-bottom:2px solid {c}" if c else ""
        parts.append(f"<span class='w' style='{style}' title='{mmss(w['start'])}'>{w['word']}</span>")
    st.markdown(f"<div class='glass transcript'>{' '.join(parts)}</div>", unsafe_allow_html=True)


def render_spectrum(result, audio_path, key, baseline="jfk"):
    regions = result["regions"]
    if not regions:
        st.info("No flaw regions to inspect.")
        return
    idx = st.selectbox("Flaw region", list(range(len(regions))), key=f"spec_{key}",
                       format_func=lambda i: f"{mmss(regions[i]['start'])}–{mmss(regions[i]['end'])} · "
                                             f"{FLAW_NAMES[regions[i]['flaw_type']]} · {regions[i]['words'][:50]}")
    r = regions[idx]

    @st.cache_data(show_spinner="Computing spectra...")
    def spectra(path, s, e, bs, be, bname):
        return region_spectra(path, {"start": s, "end": e, "baseline_start": bs, "baseline_end": be}, baseline)

    sp = spectra(audio_path, r["start"], r["end"], r["baseline_start"], r["baseline_end"],
                 baseline if isinstance(baseline, str) else baseline["name"])
    f = go.Figure()
    f.add_trace(go.Scatter(x=sp["fft_baseline"]["freq"], y=sp["fft_baseline"]["db"], name="Baseline (same words)",
                           line=dict(color=GREY, dash="dash")))
    f.add_trace(go.Scatter(x=sp["fft_participant"]["freq"], y=sp["fft_participant"]["db"], name="You",
                           line=dict(color=TEAL, width=2.2), fill="tonexty", fillcolor="rgba(45,212,191,0.08)"))
    plotly_dark(f, 380).update_layout(title="FFT magnitude spectrum of the region", xaxis_title="Frequency (Hz)",
                                      yaxis_title="Magnitude (dB)", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f, width="stretch")
    h1, h2 = st.columns(2)
    for col, k, title in [(h1, "mfcc_participant", "MFCCs: you"), (h2, "mfcc_baseline", "MFCCs: baseline")]:
        with col:
            hm = go.Figure(go.Heatmap(z=sp[k], colorscale="Viridis", showscale=False))
            plotly_dark(hm, 290).update_layout(title=title, xaxis_title="Frame (20 ms)", yaxis_title="MFCC #")
            st.plotly_chart(hm, width="stretch")


def render_analysis(result, audio_path, key, baseline="jfk"):
    render_scores(result)
    st.write("")
    t1, t2, t3 = st.tabs(["📈  Overlay vs baseline", "🧭  Flaw regions", "🔬  Spectrum"])
    with t1:
        render_overlay(result)
    with t2:
        render_regions(result, audio_path, key)
    with t3:
        render_spectrum(result, audio_path, key, baseline)


def render_dataset_eval():
    labels = load_labels()
    st.markdown("<div class='section-title'>Contrastive dataset</div>", unsafe_allow_html=True)
    st.markdown("<span class='muted'>The same JFK text delivered ideally (baseline) and with increasing flaws: four human "
                "recordings and synthetic versions with flaws injected at exactly known times (ground truth).</span>",
                unsafe_allow_html=True)
    if len(labels):
        c = st.columns(4)
        c[0].markdown(kpi("Recordings", labels["file"].nunique(), labels["file"].nunique()), unsafe_allow_html=True)
        c[1].markdown(kpi("Labeled flaws", int((labels["flaw_type"] != "none").sum()), int((labels["flaw_type"] != "none").sum())), unsafe_allow_html=True)
        c[2].markdown(kpi("Flaw types", labels.loc[labels["flaw_type"] != "none", "flaw_type"].nunique(), 6), unsafe_allow_html=True)
        c[3].markdown(kpi("Severity levels", 5, 5, "0 = baseline … 4 = egregious"), unsafe_allow_html=True)
        st.write("")
        f1, f2 = st.columns(2)
        src = f1.multiselect("Source", sorted(labels["source"].unique()), default=sorted(labels["source"].unique()))
        sev = f2.multiselect("Severity", sorted(labels["severity"].unique()), default=sorted(labels["severity"].unique()))
        view = labels[labels["source"].isin(src) & labels["severity"].isin(sev)]
        st.dataframe(view, width="stretch", hide_index=True, height=260)
        pick = st.selectbox("Listen to a dataset recording", list(dict.fromkeys(view["file"].tolist())) or ["-"])
        if pick != "-" and (ROOT / pick).exists():
            st.audio(str(ROOT / pick))
    if EVAL.exists():
        ev = json.loads(EVAL.read_text(encoding="utf-8"))
        st.markdown("<div class='section-title'>Does it work? Evaluation against the labels</div>", unsafe_allow_html=True)
        e1, e2 = st.columns([1, 1.2])
        with e1:
            sv = ev["score_by_severity"]
            bar = go.Figure(go.Bar(x=[f"Severity {k}" for k in sv], y=list(sv.values()), text=list(sv.values()),
                                   textposition="outside", marker=dict(color=list(sv.values()), colorscale=[[0, VIOLET], [1, TEAL]])))
            plotly_dark(bar, 340).update_layout(title="Mean overall score by severity (should fall)", yaxis_range=[0, 108])
            st.plotly_chart(bar, width="stretch")
        with e2:
            rows = [{"Flaw type": FLAW_NAMES.get(k, k), "Labels": v["labels"], "Recall": v["recall"],
                     "Precision": v["precision"], "Mean IoU": v["mean_iou"]} for k, v in ev["per_flaw_type"].items()]
            st.markdown(f"**Temporal grounding** (hit = IoU ≥ {ev.get('iou_threshold', 0.3)})")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("IoU = overlap between the detected flaw region and the labeled region, divided by their union.")


def render_how():
    steps = [("🎧", "Inputs", "Participant audio + the transcript of the text they deliver."),
             ("🧩", "Forced alignment", "faster-whisper word timestamps matched to the transcript (difflib): every word gets a start/end time."),
             ("📊", "Acoustic features", "Every 20 ms: F0 pitch (pYIN), RMS energy, spectral centroid & flatness, 13 MFCCs, FFT. Per word: rate, pause (from silence), pitch range, loudness, clarity."),
             ("⚖️", "Word-by-word comparison", "Word i of the participant vs word i of the baseline: z = (participant − baseline) / spread of the baseline."),
             ("⏱️", "Temporal grounding", "Words with |z| > 1.8 (smoothed over 3 words) merge into flaw regions with exact start/end times."),
             ("🧾", "Explanations + rubric", "Every region gets a numeric cause. Rubric per criterion = max(0, 10 − 2·mean|z|). Same audio → same score.")]
    cols = st.columns(3)
    for i, (ico, title, body) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"<div class='glass' style='min-height:170px;margin-bottom:14px'><div style='font-size:1.6rem'>{ico}</div>"
                        f"<div style='font-family:Sora;font-weight:700;margin:6px 0;color:{TEAL}'>{i + 1}. {title}</div>"
                        f"<div class='muted'>{body}</div></div>", unsafe_allow_html=True)
    st.markdown("""
**Why a baseline of the same text?** Good delivery depends on the words: a dramatic pause before *"ask not"* is a strength,
the same pause mid-phrase is a flaw. Word-by-word comparison with an ideal delivery captures that.

**Fair across voices:** pitch in semitones relative to each speaker's own median, loudness relative to each speaker's own level,
so a student on a phone can be compared fairly with JFK in 1961.

**Reproducible:** no LLM and no randomness; every score is a fixed formula on z-scores.
""")


# ------------------------------------------------------------------------------------------- login
def login_page():
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("<div class='brand'>🎙️ OrateLens</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>Objective feedback on <span>how</span> a speech is delivered.</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Compare any performance word by word with an ideal delivery of the same text. "
                    "Find the exact seconds where pacing, pauses, pitch, volume or clarity break down, and see why, "
                    "with the numbers behind it.</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("".join(f"<span class='pill'>{p}</span>" for p in
                            ["Forced alignment", "F0 · MFCC · FFT", "Temporal grounding", "Explainable rubric", "Reproducible scores"]),
                    unsafe_allow_html=True)
        for ico, t, s in [("🎯", "Pinpoints the flaw", "Exact start–end time of every delivery failure, with playback."),
                          ("🧮", "Explains the cause", "“Speaking rate 4.6 words/s vs baseline 2.1 (2.2× faster)”."),
                          ("⚖️", "Same score, every time", "A fixed rubric removes judge-to-judge inconsistency.")]:
            st.markdown(f"<div class='feature'><div class='ico'>{ico}</div><div><b>{t}</b><span>{s}</span></div></div>",
                        unsafe_allow_html=True)
    with right:
        st.write("")
        with st.container(border=True):
            st.markdown("<div class='section-title'>Sign in</div>", unsafe_allow_html=True)
            role = st.radio("I am a", ["🎤 Participant", "⚖️ Judge"], horizontal=True)
            name = st.text_input("Your name", placeholder="e.g. Kanisha")
            code = ""
            if "Judge" in role:
                code = st.text_input("Judge access code", type="password", placeholder="Provided by the organisers")
            if st.button("Continue →", type="primary", width="stretch"):
                if not name.strip():
                    st.error("Please enter your name.")
                elif "Judge" in role and code.strip() != JUDGE_CODE:
                    st.error("That judge access code is not valid.")
                else:
                    st.session_state.user = {"name": name.strip(), "role": "judge" if "Judge" in role else "participant"}
                    st.rerun()
            st.caption(f"Prototype demo · judge code: **{JUDGE_CODE}**")


# ------------------------------------------------------------------------------------------- participant
def participant_page(user):
    st.markdown(f"<div class='brand'>🎙️ OrateLens</div><div class='hero-title' style='font-size:2.2rem'>"
                f"Hi {user['name']}, let's hear your speech.</div>", unsafe_allow_html=True)
    tab_new, tab_res, tab_prog, tab_how = st.tabs(["🎙️  New attempt", "📋  My analysis", "📈  My progress", "⚙️  How it works"])
    with tab_new:
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("<div class='section-title'>1 · Your delivery</div>", unsafe_allow_html=True)
                mode = st.radio("Source", ["Record now", "Upload a file", "Sample recording"], horizontal=True,
                                label_visibility="collapsed")
                chosen = None
                if mode == "Record now":
                    rec = st.audio_input("Read the text on the right aloud")
                    if rec is not None:
                        rec.name = "recording.wav"
                        chosen = rec
                elif mode == "Upload a file":
                    chosen = st.file_uploader("Upload your recording", type=["wav", "mp3", "m4a"])
                else:
                    f = st.selectbox("Sample", list(DEMO_SPEAKERS), format_func=lambda x: Path(x).stem)
                    chosen = ROOT / f
                if st.button("Analyse my delivery", type="primary", width="stretch", disabled=chosen is None):
                    baseline = "jfk"
                    text = st.session_state.get("transcript", "") or ""
                    ok = True
                    if st.session_state.get("bmode", "JFK").startswith("My own"):
                        if st.session_state.get("base_upload") is None or not text.strip():
                            st.error("Upload the ideal recording and paste its script to use your own baseline.")
                            ok = False
                        else:
                            bpath = to_wav(st.session_state.base_upload, prefix="baseline_")
                            baseline = custom_baseline(bpath, text, st.session_state.get("base_label") or "Custom baseline")
                    if ok:
                        path = chosen if isinstance(chosen, Path) else to_wav(chosen)
                        result = run_analysis(path, text or DEFAULT_TRANSCRIPT, baseline)
                        save_attempt(user["name"], result, path, baseline)
                        st.session_state.p_result, st.session_state.p_audio = result, str(path)
                        st.session_state.p_baseline = baseline
                        st.success(f"Done! Overall score **{result['overall']}/100**. Open **My analysis** to see the details.")
        with c2:
            with st.container(border=True):
                st.markdown("<div class='section-title'>2 · Baseline & text</div>", unsafe_allow_html=True)
                bopts = ["JFK · 1961 inaugural (default)"] + (["My own ideal recording"] if custom_baseline else [])
                bmode = st.radio("Baseline", bopts, horizontal=True, key="bmode")
                if bmode.startswith("JFK"):
                    st.session_state.transcript = st.text_area("Text to deliver", DEFAULT_TRANSCRIPT, height=200)
                    st.caption("Deliver this exact text. You're compared word by word with JFK's delivery of it.")
                    st.session_state.base_upload = None
                else:
                    st.session_state.base_label = st.text_input("Who is the baseline speaker?", "Coach")
                    st.session_state.base_upload = st.file_uploader(
                        "Ideal delivery of your script (coach / champion / your best take)", type=["wav", "mp3", "m4a"],
                        key="base_up")
                    st.session_state.transcript = st.text_area("Your script (exact words of the ideal recording)", "",
                                                               height=160, placeholder="Paste the script here...")
                    st.caption("Any script works: the ideal recording becomes the baseline and you're compared word by word with it.")
    with tab_res:
        res = st.session_state.get("p_result")
        if not res:
            st.info("Make an attempt in **New attempt** to see your analysis here.")
        else:
            st.caption(f"Compared with baseline: {res.get('baseline_label', 'JFK, 1961 inaugural address')}")
            render_analysis(res, st.session_state.p_audio, "p", st.session_state.get("p_baseline", "jfk"))
    with tab_prog:
        mine = [a for a in load_attempts() if a["participant"].lower() == user["name"].lower()]
        if not mine:
            st.info("Your progress appears here after your first attempt.")
        else:
            mine.sort(key=lambda a: a["ts"])
            latest, best = mine[-1], max(a["overall"] for a in mine)
            delta = latest["overall"] - mine[0]["overall"]
            c = st.columns(4)
            c[0].markdown(kpi("Latest score", latest["overall"], 100), unsafe_allow_html=True)
            c[1].markdown(kpi("Best score", best, 100), unsafe_allow_html=True)
            c[2].markdown(f"<div class='glass kpi'><div class='label'>Change since first</div><div class='value' "
                          f"style='color:{'#34D399' if delta >= 0 else '#F0506E'}'>{delta:+d}</div><div class='sub'>points</div></div>",
                          unsafe_allow_html=True)
            c[3].markdown(f"<div class='glass kpi'><div class='label'>Attempts</div><div class='value'>{len(mine)}</div>"
                          f"<div class='sub'>keep practising</div></div>", unsafe_allow_html=True)
            g1, g2 = st.columns([1.3, 1])
            with g1:
                x = [f"#{i + 1} · {a['time']}" for i, a in enumerate(mine)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=[a["overall"] for a in mine], name="Overall", mode="lines+markers",
                                         line=dict(color=TEAL, width=3), marker=dict(size=9)))
                for k, n in RUBRIC.items():
                    fig.add_trace(go.Scatter(x=x, y=[a["rubric"][k] * 10 for a in mine], name=n, mode="lines",
                                             line=dict(width=1.3, dash="dot"), visible="legendonly"))
                plotly_dark(fig, 360).update_layout(title="Overall score across attempts", yaxis_range=[0, 105],
                                                    legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")
            with g2:
                series = [("First attempt", mine[0]["rubric"])] + ([("Latest", latest["rubric"])] if len(mine) > 1 else [])
                st.plotly_chart(radar(series, 360), width="stretch")
            weakest = sorted(latest["rubric"].items(), key=lambda kv: kv[1])[:2]
            st.markdown("<div class='section-title'>🎯 Focus for your next attempt</div>", unsafe_allow_html=True)
            fc = st.columns(2)
            for col, (k, v) in zip(fc, weakest):
                col.markdown(f"<div class='glass'><b style='color:{score_color(v, 10)}'>{RUBRIC[k]} · {v}/10</b>"
                             f"<div class='muted' style='margin-top:6px'>{TIPS[k]}</div></div>", unsafe_allow_html=True)
    with tab_how:
        render_how()


# ------------------------------------------------------------------------------------------- judge
def seed_demo_attempts():
    attempts = load_attempts()
    have = {a["file"] for a in attempts}
    for f, name in DEMO_SPEAKERS.items():
        p = ROOT / f
        if p.exists() and p.name not in have:
            save_attempt(name, analyze(p), p)


def judge_page(user):
    st.markdown(f"<div class='brand'>🎙️ OrateLens · Judge panel</div><div class='hero-title' style='font-size:2.2rem'>"
                f"Welcome, {user['name']}.</div>", unsafe_allow_html=True)
    if not st.session_state.get("seeded"):
        with st.spinner("Loading participants..."):
            seed_demo_attempts()
        st.session_state.seeded = True
    attempts = load_attempts()
    latest = {}
    for a in sorted(attempts, key=lambda a: a["ts"]):
        latest[a["participant"]] = a
    board = sorted(latest.values(), key=lambda a: -a["overall"])

    tab_board, tab_cmp, tab_one, tab_data, tab_how = st.tabs(
        ["🏆  Leaderboard", "🕸️  Compare", "🔎  Inspect a participant", "🗂️  Dataset & evaluation", "⚙️  How it works"])
    with tab_board:
        c = st.columns(4)
        c[0].markdown(kpi("Participants", len(board), max(len(board), 1)), unsafe_allow_html=True)
        avg = int(round(np.mean([a["overall"] for a in board]))) if board else 0
        c[1].markdown(kpi("Average score", avg, 100), unsafe_allow_html=True)
        c[2].markdown(kpi("Top score", board[0]["overall"] if board else 0, 100), unsafe_allow_html=True)
        c[3].markdown(f"<div class='glass kpi'><div class='label'>Problem statement</div><div class='value' style='font-size:1.15rem'>"
                      f"PS1 · Speech delivery</div><div class='sub'>Baseline: JFK 1961 inaugural</div></div>", unsafe_allow_html=True)
        st.write("")
        for i, a in enumerate(board):
            with st.container(border=True):
                cols = st.columns([0.4, 1.6, 1, 3.4])
                cols[0].markdown(f"<div class='rank'>#{i + 1}</div>", unsafe_allow_html=True)
                cols[1].markdown(f"**{a['participant']}**<br><span class='muted'>{a['file']} · {a['time']}</span>",
                                 unsafe_allow_html=True)
                cols[2].markdown(f"<div class='kpi'><div class='value' style='color:{score_color(a['overall'], 100)}'>"
                                 f"{a['overall']}</div></div>", unsafe_allow_html=True)
                cols[3].markdown(" ".join(
                    f"<span class='pill'>{RUBRIC[k]} <b style='color:{score_color(v, 10)}'>{v}</b></span>"
                    for k, v in a["rubric"].items()) + "<br>" + " ".join(
                    f"<span class='badge' style='background:{FLAW_COLORS[k]};margin-right:4px'>{FLAW_NAMES[k]} ×{n}</span>"
                    for k, n in a.get("flaw_counts", {}).items() if n), unsafe_allow_html=True)
    with tab_cmp:
        names = [a["participant"] for a in board]
        pick = st.multiselect("Participants to compare", names, default=names[:3])
        if pick:
            st.plotly_chart(radar([(n, latest[n]["rubric"]) for n in pick], 460), width="stretch")
            df = pd.DataFrame([{"Participant": n, "Overall": latest[n]["overall"],
                                **{RUBRIC[k]: v for k, v in latest[n]["rubric"].items()}, "Flaw regions": latest[n]["regions"]}
                               for n in pick])
            st.dataframe(df, width="stretch", hide_index=True)
    with tab_one:
        if board:
            who = st.selectbox("Participant", [a["participant"] for a in board])
            a = latest[who]
            path = Path(a["audio_path"])
            if path.exists():
                b = a.get("baseline", "jfk")
                st.audio(str(path))
                st.caption(f"Baseline: {a.get('baseline_label', 'JFK, 1961 inaugural address')}")
                render_analysis(analyze(path, None, baseline=b), str(path), "j", b)
            else:
                st.warning("The audio for this attempt is no longer available.")
    with tab_data:
        render_dataset_eval()
    with tab_how:
        render_how()


# ------------------------------------------------------------------------------------------- router
user = st.session_state.get("user")
if user:
    with st.sidebar:
        st.markdown("<div class='brand'>🎙️ OrateLens</div>", unsafe_allow_html=True)
        st.markdown(f"**{user['name']}**  \n<span class='muted'>{'Judge' if user['role'] == 'judge' else 'Participant'}</span>",
                    unsafe_allow_html=True)
        st.divider()
        st.caption("PS1 · Counterpointing Speech Analytics & Temporal Flaw Finding · Verveathon 2K26")
        if st.button("Sign out", width="stretch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    if user["role"] == "judge":
        judge_page(user)
    else:
        participant_page(user)
else:
    login_page()
