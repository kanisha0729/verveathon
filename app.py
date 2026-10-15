"""OrateLens - objective, explainable feedback on how a speech is delivered.

Run:  streamlit run app.py
Demo logins:  Participant -> any name        Judge -> access code VERVE2026
"""
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.analyze import analyze, region_spectra, BASELINES
try:   # newer pipeline/analyze.py supports custom baselines; older versions still run
    from pipeline.analyze import custom_baseline
except ImportError:
    custom_baseline = None
try:   # recording sanity check (silent mic / wrong script) - newer pipeline only
    from pipeline.analyze import audio_check, NoSpeechError
except ImportError:
    audio_check = None

    class NoSpeechError(ValueError):
        pass

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data/labels.csv"
EVAL = ROOT / "data/eval.json"
UPLOADS = ROOT / "results/uploads"
ATTEMPTS = ROOT / "results/attempts.json"
BASELINE_AUDIO = BASELINES["jfk"]["audio"]
DEFAULT_TRANSCRIPT = BASELINES["jfk"]["transcript"].read_text(encoding="utf-8")
JUDGE_CODE = "VERVE2026"

YOU, IDEAL = "#F2B544", "#8FB3E0"          # amber = the participant, steel blue = the ideal delivery
FLAW_COLORS = {"rushed": "#EF8A62", "dragging": "#D9B26A", "long_pause": "#B49CF2",
               "low_volume": "#5FB8E8", "monotone": "#E86F8E", "unclear": "#A9B1C2"}
FLAW_NAMES = {"rushed": "Rushed", "dragging": "Dragging", "long_pause": "Long pause",
              "low_volume": "Low volume", "monotone": "Monotone", "unclear": "Unclear"}
RUBRIC = {"pacing": "Pacing", "pausing": "Pausing", "expressiveness": "Expressiveness",
          "volume": "Volume", "clarity": "Clarity"}
TIPS = {
    "pacing": "Slow down on the key phrases. Aim to stay within ±20% of the ideal speaking rate.",
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
P_PAGES = [("New attempt", ":material/mic:"), ("My analysis", ":material/monitoring:"),
           ("My progress", ":material/trending_up:"), ("How it works", ":material/info:")]
J_PAGES = [("Leaderboard", ":material/leaderboard:"), ("Compare", ":material/compare_arrows:"),
           ("Inspect", ":material/search:"), ("Dataset and evaluation", ":material/database:"),
           ("How it works", ":material/info:")]
MIC_HELP = """**How to fix the microphone (Windows + Chrome):**
1. Click the ⓘ icon left of `localhost:8501` in the address bar → **Microphone → Allow** → reload the page.
2. Windows **Settings → Privacy & security → Microphone** → turn on *Microphone access* and *Let desktop apps access your microphone*.
3. Windows **Settings → System → Sound → Input** → choose the right microphone and check the level bar moves when you talk.
4. Chrome: open `chrome://settings/content/microphone` and pick the same microphone.
5. Record again and press ▶ on the recording to hear yourself before analysing."""

st.set_page_config(page_title="OrateLens", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=Hanken+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#0E1626; --side:#0A111F; --panel:#142036; --panel-2:#1A2843; --seg:#1F2E4D; --line:rgba(236,232,223,.10);
        --line-2:rgba(236,232,223,.16); --chalk:#ECE8DF; --soft:#C3C7D1; --muted:#97A0B3;
        --you:#F2B544; --ideal:#8FB3E0; --good:#7BD3A4; --mid:#E6A35C; --weak:#E86F8E; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, label { font-family:'Hanken Grotesk', system-ui, sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent !important; height:2.5rem; }
.stApp { background:var(--ink); color:var(--chalk); }
.block-container { padding-top:2.6rem !important; padding-bottom:3rem !important; max-width:1240px; }
.serif { font-family:'Spectral', Georgia, serif; }
.muted { color:var(--muted); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background:var(--side); border-right:1px solid rgba(236,232,223,.08); }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top:1.4rem; }
.sb-brand { display:flex; align-items:center; gap:10px; padding:0 8px 18px; font-family:'Spectral', serif; font-size:23px; font-weight:600; }
section[data-testid="stSidebar"] .stButton { margin-bottom:-6px; }
section[data-testid="stSidebar"] .stButton button { justify-content:flex-start !important; height:44px; min-height:44px; padding:0 12px;
    border:none !important; border-radius:10px; background:transparent !important; color:var(--soft) !important; box-shadow:none; }
section[data-testid="stSidebar"] .stButton button > div, section[data-testid="stSidebar"] .stButton button > span { justify-content:flex-start !important; gap:12px; }
section[data-testid="stSidebar"] .stButton button p { font-size:15px; font-weight:500; color:var(--soft) !important; }
section[data-testid="stSidebar"] .stButton button:hover { background:rgba(236,232,223,.05) !important; }
section[data-testid="stSidebar"] .stButton button:hover p { color:var(--chalk) !important; }
section[data-testid="stSidebar"] .stButton button[kind="primary"], section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background:var(--panel-2) !important; color:var(--you) !important; }
section[data-testid="stSidebar"] .stButton button[kind="primary"] p, section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p {
    color:var(--you) !important; font-weight:600; }
.sb-user { display:flex; align-items:center; gap:12px; padding:16px 6px 14px; margin-top:22px; border-top:1px solid rgba(236,232,223,.08); }
.sb-user .av { width:40px; height:40px; border-radius:50%; background:#2A3A5C; color:var(--you); display:grid; place-items:center;
               font-family:'Spectral', serif; font-weight:600; font-size:18px; flex:none; }
.sb-user .av.judge { background:#23395A; color:var(--ideal); }
.sb-user b { display:block; font-weight:600; font-size:15px; } .sb-user span.r { font-size:13px; color:var(--muted); }
section[data-testid="stSidebar"] .st-key-signout button { justify-content:center !important; border:1px solid var(--line-2) !important; }
section[data-testid="stSidebar"] .st-key-signout button > div, section[data-testid="stSidebar"] .st-key-signout button > span { justify-content:center !important; }
section[data-testid="stSidebar"] .st-key-signout button p { color:var(--chalk) !important; font-weight:600; }

/* ---------- page header + chips ---------- */
.page-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:14px; margin-bottom:18px; }
.page-title { font-family:'Spectral', serif; font-weight:500; font-size:36px; letter-spacing:-.015em; line-height:1.15; }
.chips { display:flex; gap:10px; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px; border:1px solid var(--line-2); color:var(--soft); font-size:14px; }
.chip.ideal { border-color:rgba(143,179,224,.35); color:#B9CDE9; }
.chip .dash { width:18px; border-top:2px dashed var(--ideal); }
.section-title { font-family:'Spectral', serif; font-weight:500; font-size:22px; letter-spacing:-.01em; margin:.1rem 0 .55rem; }

/* ---------- panels ---------- */
div[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-panel"], div[class*="st-key-reg_"] {
    border-color:var(--line) !important; border-radius:16px !important; background:var(--panel); }
.glass { border:1px solid var(--line); border-radius:16px; padding:18px 20px; background:var(--panel); }
.kpi .label { font-size:14px; color:var(--muted); }
.kpi .value { font-family:'Spectral', serif; font-size:38px; font-weight:500; line-height:1.1; font-variant-numeric:lining-nums tabular-nums; }
.kpi .value .of { font-family:'Hanken Grotesk', sans-serif; font-size:14px; color:var(--muted); margin-left:4px; }
.kpi .sub { color:var(--muted); font-size:13px; }

/* ---------- score sheet + profile ---------- */
.sheet { display:grid; grid-template-columns:minmax(240px, 1fr) 2fr; border:1px solid var(--line); border-radius:16px; background:var(--panel); overflow:hidden; }
.sheet .total { padding:26px 28px; border-right:1px solid rgba(236,232,223,.08); display:flex; flex-direction:column; justify-content:center; gap:6px; }
.sheet .total .num { font-family:'Spectral', serif; font-size:84px; line-height:.95; font-weight:500; font-variant-numeric:tabular-nums; }
.sheet .total .num span { font-family:'Hanken Grotesk', sans-serif; font-size:17px; color:var(--muted); margin-left:6px; }
.sheet .total .verdict { font-family:'Spectral', serif; font-style:italic; font-size:20px; margin-top:4px; }
.sheet .rows { padding:12px 28px; display:flex; flex-direction:column; justify-content:center; }
.sheet .row { display:grid; grid-template-columns:9rem 1fr 3rem; align-items:center; gap:16px; padding:11px 0; border-bottom:1px solid rgba(236,232,223,.08); }
.sheet .row:last-child { border-bottom:none; }
.sheet .row .name { font-weight:600; }
.sheet .row .val { font-family:'Spectral', serif; font-size:20px; text-align:right; font-variant-numeric:tabular-nums; }
.track { height:7px; border-radius:7px; background:rgba(236,232,223,.08); overflow:hidden; }
.track i { display:block; height:7px; border-radius:7px; }
.profile { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); border:1px solid var(--line); border-radius:16px; background:rgba(20,32,54,.55); margin:16px 0 18px; }
.profile > div { padding:16px 18px; border-left:1px solid rgba(236,232,223,.08); display:flex; flex-direction:column; gap:3px; }
.profile > div:first-child { border-left:none; }
.profile .label { font-size:13px; color:var(--muted); }
.profile .v { font-family:'Spectral', serif; font-size:26px; line-height:1.2; color:var(--you); font-variant-numeric:tabular-nums; }
.profile .v small { font-family:'Hanken Grotesk', sans-serif; font-size:13px; color:var(--muted); margin-left:4px; }
.profile .ideal { font-size:13px; color:#A9C2E5; }
.legend { display:flex; justify-content:flex-end; gap:18px; font-size:13px; color:var(--muted); padding-top:10px; }
.legend span { display:inline-flex; align-items:center; gap:8px; }
.legend i { width:18px; border-top:3px solid var(--you); } .legend i.d { border-top:2px dashed var(--ideal); }

/* ---------- flaw regions ---------- */
.ftag { display:inline-flex; align-items:center; gap:8px; padding:5px 11px; border-radius:8px; font-size:13px; font-weight:700; white-space:nowrap;
        color:var(--c); border:1px solid color-mix(in srgb, var(--c) 50%, transparent); }
.ftag::before { content:""; width:8px; height:8px; border-radius:50%; background:var(--c); }
.rtime { font-weight:700; font-variant-numeric:tabular-nums; white-space:nowrap; }
.rwords { font-family:'Spectral', serif; font-size:18px; line-height:1.4; }
.rcause { font-size:14px; }
div[class*="st-key-py_"] button { border:1px solid rgba(242,181,68,.5) !important; background:transparent !important; }
div[class*="st-key-py_"] button p { color:var(--you) !important; font-weight:600; }
div[class*="st-key-pb_"] button { border:1px solid rgba(143,179,224,.45) !important; background:transparent !important; }
div[class*="st-key-pb_"] button p { color:#B9CDE9 !important; font-weight:600; }
.transcript { line-height:2.1; font-size:18px; font-family:'Spectral', serif; }
.transcript span.w { padding:2px 3px; border-radius:4px; }
.pause { display:inline-block; padding:0 7px; margin:0 3px; border-radius:5px; font-family:'Hanken Grotesk', sans-serif; font-size:12px;
         color:#C9B8F7; border:1px solid rgba(180,156,242,.5); font-weight:700; }

/* ---------- new attempt ---------- */
.script { max-height:340px; overflow:auto; padding:18px 20px; border-radius:12px; background:var(--ink); border:1px solid rgba(236,232,223,.06);
          font-family:'Spectral', Georgia, serif; font-size:19px; line-height:1.75; color:#E3DFD6; }
.meta-row { display:flex; gap:18px; flex-wrap:wrap; font-size:14px; color:var(--muted); margin-top:10px; }
.status { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-radius:10px; font-size:14px; margin:4px 0 6px; }
.status b { display:inline-flex; align-items:center; gap:10px; font-weight:600; }
.status b::before { content:""; width:8px; height:8px; border-radius:50%; background:currentColor; }
.status.ok { background:rgba(123,211,164,.08); border:1px solid rgba(123,211,164,.3); } .status.ok b { color:#A6E3C2; }
.status.bad { background:rgba(232,111,142,.08); border:1px solid rgba(232,111,142,.35); } .status.bad b { color:#F29AB1; }
.status span { color:var(--muted); }
.stTextArea textarea { font-family:'Spectral', Georgia, serif !important; font-size:17px !important; line-height:1.65 !important; }

/* ---------- segmented controls ---------- */
div[data-testid="stElementContainer"]:has(> div[data-testid="stButtonGroup"]) { width:100% !important; }
div[data-testid="stButtonGroup"] { width:100%; }
div[data-testid="stButtonGroup"] > div { width:100% !important; max-width:none !important; display:flex; gap:4px; padding:4px; border-radius:12px; background:var(--ink); border:1px solid rgba(236,232,223,.08); }
div[data-testid="stButtonGroup"] button { flex:1 1 0; height:40px; border:none !important; border-radius:9px !important; background:transparent !important; }
div[data-testid="stButtonGroup"] button p { color:var(--muted); font-weight:600; font-size:14px; }
div[data-testid="stButtonGroup"] button[aria-checked="true"], div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"], div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
    background:var(--seg) !important; }
div[data-testid="stButtonGroup"] button[aria-checked="true"] p, div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p, div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] p {
    color:var(--chalk) !important; font-weight:700; }

/* ---------- login ---------- */
.login-brand { min-height:calc(100vh - 7rem); box-sizing:border-box; display:flex; flex-direction:column; justify-content:space-between;
               padding:48px 56px; border-radius:22px; background:var(--side); border:1px solid rgba(236,232,223,.08); }
.login-brand .mid { display:flex; flex-direction:column; align-items:flex-start; gap:20px; }
.login-brand .name { font-family:'Spectral', serif; font-size:clamp(52px, 6vw, 72px); line-height:1; font-weight:600; letter-spacing:-.02em; }
.login-brand .tag { font-family:'Spectral', serif; font-style:italic; font-size:clamp(24px, 2.4vw, 32px); line-height:1.2; color:#C9C4B8; }
.login-brand .foot { font-size:14px; color:#8A93A6; }
.st-key-signin { max-width:400px; margin:0 auto; padding-top:18vh; }
.signin-title { font-family:'Spectral', serif; font-weight:500; font-size:40px; letter-spacing:-.015em; margin-bottom:6px; }
.st-key-signin div[data-testid="stButtonGroup"] > div { background:var(--panel); border-color:var(--line); }
.st-key-signin div[data-testid="stButtonGroup"] button { height:44px; }
.st-key-signin div[data-testid="stButtonGroup"] button[aria-checked="true"], .st-key-signin div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
.st-key-signin div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] { background:var(--you) !important; }
.st-key-signin div[data-testid="stButtonGroup"] button[aria-checked="true"] p, .st-key-signin div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p,
.st-key-signin div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] p { color:#1A1305 !important; }
.st-key-signin .stTextInput input { height:48px; font-size:16px; }
.demo-code { text-align:center; font-size:13px; color:#8A93A6; margin-top:6px; }

/* ---------- buttons ---------- */
div.stButton > button, div.stFormSubmitButton > button { border-radius:10px; font-weight:600; }
button[kind="primary"], [data-testid="stBaseButton-primary"] { color:#1A1305 !important; border:none !important; min-height:48px; }
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color:#1A1305 !important; font-weight:700; font-size:16px; }
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled { background:var(--panel-2) !important; opacity:1; }
button[kind="primary"]:disabled p, [data-testid="stBaseButton-primary"]:disabled p { color:var(--muted) !important; }
button:focus-visible { outline:2px solid var(--you) !important; outline-offset:2px; }

/* ---------- leaderboard ---------- */
.board { overflow-x:auto; border-radius:16px; border:1px solid var(--line); background:var(--panel); margin-top:18px; }
.board table { width:100%; min-width:860px; margin:0 !important; border:none !important; border-collapse:collapse; font-size:15px; font-variant-numeric:tabular-nums; }
.board th, .board td { border:none !important; background:transparent !important; }
.board th { text-align:left; color:var(--muted); font-size:13px; font-weight:600; padding:14px 12px; }
.board td { padding:15px 12px; border-top:1px solid rgba(236,232,223,.08) !important; }
.board th:first-child, .board td:first-child { padding-left:22px; } .board th.n, .board td.n { text-align:right; }
.board .rank { font-family:'Spectral', serif; font-size:24px; color:var(--muted); } .board .rank.first { color:var(--you); }
.board .score { display:inline-flex; align-items:center; gap:12px; }
.board .score b { font-family:'Spectral', serif; font-weight:500; font-size:24px; width:36px; }
.board .score .track { width:90px; height:6px; }

@media (max-width: 900px) {
  .sheet { grid-template-columns:1fr; } .sheet .total { border-right:none; border-bottom:1px solid rgba(236,232,223,.08); }
  .profile { grid-template-columns:repeat(2, minmax(0, 1fr)); } .profile > div:nth-child(odd) { border-left:none; }
  .login-brand { min-height:auto; } .st-key-signin { padding-top:2rem; }
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- helpers
def mmss(t):
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def score_color(v, top):
    f = v / top if top else 0
    return "#7BD3A4" if f >= 0.7 else ("#E6A35C" if f >= 0.4 else "#E86F8E")


def kpi(label, value, top=None, sub="", color=True):
    c = score_color(value, top) if (color and top) else "var(--chalk)"
    return (f"<div class='glass kpi'><div class='label'>{label}</div><div class='value' style='color:{c}'>{value}</div>"
            + (f"<div class='sub'>{sub}</div>" if sub else "") + "</div>")


def ftag(kind, extra=""):
    return f"<span class='ftag' style='--c:{FLAW_COLORS.get(kind, '#999')}'>{FLAW_NAMES.get(kind, kind)}{extra}</span>"


def verdict(overall):
    if overall >= 85:
        return "Polished delivery"
    if overall >= 70:
        return "Solid, with a few slips"
    if overall >= 50:
        return "Needs work in places"
    return "Needs significant work"


def logo(size=28):
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 26 26' aria-hidden='true'>"
            "<rect x='2' y='10' width='2.4' height='6' rx='1.2' fill='#8FB3E0'/><rect x='6.5' y='6' width='2.4' height='14' rx='1.2' fill='#8FB3E0'/>"
            "<rect x='11' y='3' width='2.4' height='20' rx='1.2' fill='#8FB3E0'/>"
            "<path d='M15.5 15 C18 15, 18.5 8, 21 8 S 23.5 13, 25 12' fill='none' stroke='#F2B544' stroke-width='2.4' stroke-linecap='round'/></svg>")


def chip(text, ideal=False):
    if ideal:
        return f"<span class='chip ideal'><span class='dash'></span>{text}</span>"
    return f"<span class='chip'>{text}</span>"


def page_header(title, chips=()):
    st.markdown(f"<div class='page-head'><div class='page-title'>{title}</div><div class='chips'>{''.join(chips)}</div></div>",
                unsafe_allow_html=True)


def seg(label, options, key, default=None):
    """Segmented control (falls back to a horizontal radio on older Streamlit)."""
    default = default if default in options else options[0]
    f = getattr(st, "segmented_control", None)
    if f is not None:
        return f(label, options, default=default, key=key, label_visibility="collapsed") or default
    return st.radio(label, options, index=options.index(default), key=key, horizontal=True, label_visibility="collapsed")


def ideal_name(label):
    return (label or "JFK, 1961").replace(" inaugural address", "")


def plotly_dark(fig, height):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Hanken Grotesk, sans-serif", color="#CFD3DC"),
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor="rgba(236,232,223,0.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(236,232,223,0.07)", zeroline=False)
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
    rec = {"participant": name, "file": result["file"], "audio_path": str(audio_path),
           "time": datetime.now().strftime("%d %b %H:%M"), "ts": time.time(),
           "overall": result["overall"], "rubric": result["rubric"],
           "regions": len(result["regions"]), "baseline": baseline,
           "baseline_label": result.get("baseline_label", "JFK, 1961 inaugural address"),
           "flaw_counts": {k: sum(1 for r in result["regions"] if r["flaw_type"] == k) for k in FLAW_NAMES}}
    attempts.append(rec)
    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.write_text(json.dumps(attempts, indent=1), encoding="utf-8")
    rec["n"] = sum(1 for a in attempts if a["participant"].lower() == name.lower())
    return rec


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
        kw = {"strict": True} if audio_check is not None else {}
        try:
            result = analyze(path, transcript, baseline=baseline,
                             progress=lambda name, frac: status.write(f"✓ {name}"), **kw)
        except NoSpeechError:
            status.update(label="Recording could not be scored", state="error", expanded=False)
            raise
        status.update(label="Analysis complete", state="complete", expanded=False)
    return result


def radar(series, height=380):
    cats = list(RUBRIC.values())
    fig = go.Figure()
    palette = [YOU, IDEAL, "#7BD3A4", "#E86F8E", "#B49CF2", "#EF8A62"]
    for i, (name, rub) in enumerate(series):
        vals = [rub[k] for k in RUBRIC] + [rub[list(RUBRIC)[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], name=name, fill="toself",
                                      line=dict(color=palette[i % len(palette)], width=2), opacity=0.75))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 10], gridcolor="rgba(255,255,255,0.1)"),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
                      legend=dict(orientation="h", y=-0.1))
    return plotly_dark(fig, height)


def short_cause(r):
    """One-line cause of a flaw region, from its numbers."""
    m, t = r["metrics"], r["flaw_type"]
    p, b = m["participant"], m["baseline"]
    if t == "long_pause":
        return f"{p:.1f} s silence where the ideal pauses {b:.1f} s"
    if t == "rushed":
        return f"{p:.1f} words/s against the ideal {b:.1f}, {p / max(b, 0.1):.1f}× faster"
    if t == "dragging":
        return f"{p:.1f} words/s against the ideal {b:.1f}, {b / max(p, 0.1):.1f}× slower"
    if t == "low_volume":
        return f"{p:+.1f} dB against the ideal {b:+.1f} dB, each from the speaker's own average"
    if t == "monotone":
        return f"Pitch range {p:.1f} semitones against the ideal {b:.1f}"
    if t == "unclear":
        return f"Recognition confidence {p:.2f} against the ideal {b:.2f}"
    return r.get("explanation", "")


# ------------------------------------------------------------------------------------------- analysis views
def render_scores(result):
    overall, n_reg = result["overall"], len(result["regions"])
    rows = "".join(
        f"<div class='row'><span class='name'>{name}</span>"
        f"<div class='track'><i style='width:{result['rubric'][k] * 10:.0f}%;background:{score_color(result['rubric'][k], 10)}'></i></div>"
        f"<span class='val'>{result['rubric'][k]:.1f}</span></div>"
        for k, name in RUBRIC.items())
    st.markdown(
        f"<div class='sheet'><div class='total'><span class='muted' style='font-size:14px'>Overall score</span>"
        f"<div class='num' style='color:{score_color(overall, 100)}'>{overall}<span>/ 100</span></div>"
        f"<div class='verdict'>{verdict(overall)}</div>"
        f"<span class='muted' style='font-size:14px'>{n_reg} flaw region{'s' if n_reg != 1 else ''}</span></div>"
        f"<div class='rows'>{rows}</div></div>", unsafe_allow_html=True)
    cov = result.get("coverage", 1.0)
    if cov < 0.85:
        st.warning(f"Only {cov:.0%} of the script was clearly recognised in this recording. Read the full text "
                   f"shown in the Script panel. Missing or unclear words lower the Clarity and Overall scores.")
    prof = result.get("profile")
    if prof and prof.get("participant"):
        p, b = prof["participant"], prof.get("baseline", {})
        items = [("Pace", "wpm", "wpm", 0), ("Pauses", "pauses", "", 0), ("Pitch range", "pitch_range", "st", 1),
                 ("Loudness range", "loudness_dynamics", "dB", 1), ("Clarity", "clarity", "", 2), ("Duration", "duration", "s", 1)]
        cells = "".join(
            f"<div><span class='label'>{label}</span><span class='v'>{p.get(key, 0):.{nd}f}<small>{unit}</small></span>"
            f"<span class='ideal'>Ideal {b.get(key, 0):.{nd}f}</span></div>"
            for label, key, unit, nd in items)
        st.markdown(f"<div class='profile'>{cells}</div>", unsafe_allow_html=True)


METRICS = {"Pitch": ("f0", "Semitones from your median"), "Loudness": ("energy", "dB from your median"),
           "Pace": ("rate", "Words per second")}


def render_metric_chart(result, metric):
    ov, regions = result["overlay"], result["regions"]
    key, ytitle = METRICS[metric]
    t = np.array(ov["t"], dtype=float)
    p = [np.nan if v is None else v for v in ov[f"{key}_participant"]]
    b = [np.nan if v is None else v for v in ov[f"{key}_baseline"]]
    fig = go.Figure()
    for r in regions:
        fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=FLAW_COLORS.get(r["flaw_type"], "#888"), opacity=0.16, line_width=0)
    fig.add_trace(go.Scatter(x=t, y=b, name="Ideal", line=dict(color=IDEAL, dash="dash", width=1.6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=p, name="You", line=dict(color=YOU, width=2.4),
                             hovertemplate="%{y:.2f}<extra>You</extra>"))
    if regions:
        vals = [v for v in p if not np.isnan(v)] or [0]
        top = max(vals) + 0.08 * (max(vals) - min(vals) + 1)
        fig.add_trace(go.Scatter(
            x=[(r["start"] + r["end"]) / 2 for r in regions], y=[top] * len(regions), mode="markers",
            marker=dict(size=11, symbol="triangle-down", color=[FLAW_COLORS.get(r["flaw_type"]) for r in regions]),
            hovertext=[f"<b>{FLAW_NAMES[r['flaw_type']]}</b> {mmss(r['start'])} to {mmss(r['end'])}<br>{short_cause(r)}"
                       for r in regions], hoverinfo="text"))
    ticks = np.arange(0, (t.max() if len(t) else 1) + 5, 5)
    fig.update_xaxes(tickvals=ticks, ticktext=[mmss(x) for x in ticks])
    plotly_dark(fig, 360).update_layout(showlegend=False, yaxis_title=ytitle, margin=dict(l=10, r=10, t=10, b=10),
                                        hovermode="closest")
    st.plotly_chart(fig, width="stretch")


def render_spectrum(result, audio_path, key, baseline="jfk"):
    regions = result["regions"]
    if not regions:
        st.markdown("<span class='muted'>No flaw regions to inspect.</span>", unsafe_allow_html=True)
        return
    idx = st.selectbox("Flaw region", list(range(len(regions))), key=f"spec_{key}",
                       format_func=lambda i: f"{mmss(regions[i]['start'])} to {mmss(regions[i]['end'])}, "
                                             f"{FLAW_NAMES[regions[i]['flaw_type']]}: {regions[i]['words'][:50]}")
    r = regions[idx]

    @st.cache_data(show_spinner="Computing spectra...")
    def spectra(path, s, e, bs, be, bname):
        return region_spectra(path, {"start": s, "end": e, "baseline_start": bs, "baseline_end": be}, baseline)

    sp = spectra(audio_path, r["start"], r["end"], r["baseline_start"], r["baseline_end"],
                 baseline if isinstance(baseline, str) else baseline["name"])
    f = go.Figure()
    f.add_trace(go.Scatter(x=sp["fft_baseline"]["freq"], y=sp["fft_baseline"]["db"], name="Ideal (same words)",
                           line=dict(color=IDEAL, dash="dash")))
    f.add_trace(go.Scatter(x=sp["fft_participant"]["freq"], y=sp["fft_participant"]["db"], name="You",
                           line=dict(color=YOU, width=2.2), fill="tonexty", fillcolor="rgba(242,181,68,0.08)"))
    plotly_dark(f, 360).update_layout(title="FFT magnitude spectrum of the region", xaxis_title="Frequency (Hz)",
                                      yaxis_title="Magnitude (dB)", showlegend=False)
    st.plotly_chart(f, width="stretch")
    h1, h2 = st.columns(2)
    for col, k, title in [(h1, "mfcc_participant", "MFCCs: you"), (h2, "mfcc_baseline", "MFCCs: ideal")]:
        with col:
            hm = go.Figure(go.Heatmap(z=sp[k], colorscale=[[0, "#0E1626"], [0.5, "#4C6A9A"], [1, "#F2B544"]], showscale=False))
            plotly_dark(hm, 280).update_layout(title=title, xaxis_title="Frame (20 ms)", yaxis_title="MFCC #")
            st.plotly_chart(hm, width="stretch")


def render_regions(result, audio_path, key):
    regions = sorted(result["regions"], key=lambda r: r["start"])
    st.markdown("<div class='section-title' style='margin-top:22px'>Flaw regions</div>", unsafe_allow_html=True)
    play = st.session_state.get(f"play_{key}")
    if play:
        st.markdown(f"<span class='muted'>Now playing: {play['label']}</span>", unsafe_allow_html=True)
        st.audio(play["path"], start_time=int(play["start"]))
    if not regions:
        st.markdown("<div class='glass muted'>No flaw regions. Every word stayed close to the ideal delivery.</div>",
                    unsafe_allow_html=True)
    for k, r in enumerate(regions):
        with st.container(border=True, key=f"reg_{key}_{k}"):
            a, b, c, d = st.columns([1.15, 0.95, 4.2, 2.1], vertical_alignment="center")
            a.markdown(ftag(r["flaw_type"]), unsafe_allow_html=True)
            b.markdown(f"<span class='rtime'>{mmss(r['start'])} – {mmss(r['end'])}</span>", unsafe_allow_html=True)
            words = r["words"] if len(r["words"]) <= 90 else r["words"][:88] + "…"
            if r["flaw_type"] == "long_pause":
                words = "‖ " + words
            c.markdown(f"<div class='rwords'>“{words}”</div><div class='muted rcause'>{short_cause(r)}</div>",
                       unsafe_allow_html=True)
            with d:
                x, y = st.columns(2)
                if x.button("Play you", key=f"py_{key}_{k}", width="stretch"):
                    st.session_state[f"play_{key}"] = {"path": audio_path, "start": max(0, r["start"] - 0.5),
                                                       "label": f"you, from {mmss(r['start'])}"}
                    st.rerun()
                if y.button("Play ideal", key=f"pb_{key}_{k}", width="stretch"):
                    st.session_state[f"play_{key}"] = {"path": result.get("baseline_audio", str(BASELINE_AUDIO)),
                                                       "start": max(0, r["baseline_start"] - 0.5),
                                                       "label": f"the ideal delivery, same words, from {mmss(r['baseline_start'])}"}
                    st.rerun()


def render_transcript(result):
    regions = result["regions"]
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
            parts.append(f"<span class='pause'>‖ {pause_at[i]['end'] - pause_at[i]['start']:.1f}s</span>")
        c = word_color.get(i)
        style = f"background:{c}2E;border-bottom:2px solid {c}" if c else ""
        parts.append(f"<span class='w' style='{style}' title='{mmss(w['start'])}'>{w['word']}</span>")
    st.markdown("<div class='section-title' style='margin-top:22px'>Transcript</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='glass transcript'>{' '.join(parts)}</div>", unsafe_allow_html=True)


def render_analysis(result, audio_path, key, baseline="jfk"):
    render_scores(result)
    with st.container(border=True, key=f"panel_chart_{key}"):
        left, right = st.columns([1.3, 1], vertical_alignment="center")
        with left:
            view = seg("View", ["Pitch", "Loudness", "Pace", "Spectrum"], key=f"view_{key}")
        right.markdown("<div class='legend'><span><i></i>You</span><span><i class='d'></i>Ideal</span></div>",
                       unsafe_allow_html=True)
        if view == "Spectrum":
            render_spectrum(result, audio_path, key, baseline)
        else:
            render_metric_chart(result, view)
    render_regions(result, audio_path, key)
    render_transcript(result)


def render_dataset_eval():
    labels = load_labels()
    st.markdown("<div class='section-title'>Contrastive dataset</div>", unsafe_allow_html=True)
    st.markdown("<span class='muted'>The same JFK text delivered ideally and with increasing flaws: four human "
                "recordings and synthetic versions with flaws injected at exactly known times.</span>",
                unsafe_allow_html=True)
    if len(labels):
        c = st.columns(4)
        c[0].markdown(kpi("Recordings", labels["file"].nunique(), sub="human and synthetic", color=False), unsafe_allow_html=True)
        c[1].markdown(kpi("Labeled flaws", int((labels["flaw_type"] != "none").sum()), sub="with start and end times", color=False),
                      unsafe_allow_html=True)
        c[2].markdown(kpi("Flaw types", labels.loc[labels["flaw_type"] != "none", "flaw_type"].nunique(), color=False),
                      unsafe_allow_html=True)
        c[3].markdown(kpi("Severity levels", 5, sub="0 is ideal, 4 is egregious", color=False), unsafe_allow_html=True)
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
        st.markdown("<div class='section-title' style='margin-top:18px'>Evaluation against the labels</div>", unsafe_allow_html=True)
        e1, e2 = st.columns([1, 1.2])
        with e1:
            sv = ev["score_by_severity"]
            bar = go.Figure(go.Bar(x=[f"Severity {k}" for k in sv], y=list(sv.values()), text=list(sv.values()),
                                   textposition="outside", marker=dict(color=IDEAL)))
            plotly_dark(bar, 340).update_layout(title="Mean overall score by severity (should fall)", yaxis_range=[0, 108])
            st.plotly_chart(bar, width="stretch")
        with e2:
            rows = [{"Flaw type": FLAW_NAMES.get(k, k), "Labels": v["labels"], "Recall": v["recall"],
                     "Precision": v["precision"], "Mean IoU": v["mean_iou"]} for k, v in ev["per_flaw_type"].items()]
            st.markdown(f"**Temporal grounding** (hit = IoU ≥ {ev.get('iou_threshold', 0.3)})")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("IoU = overlap between the detected flaw region and the labeled region, divided by their union.")


def render_how():
    steps = [("Inputs", "Your recording and the exact text you delivered."),
             ("Forced alignment", "faster-whisper word timestamps are matched to the text with difflib, so every word gets a start and end time."),
             ("Acoustic features", "Every 20 ms: pitch (pYIN), loudness (RMS), spectral centroid and flatness, 13 MFCCs and the FFT. "
                                   "Per word: pace, pause before it (measured from silence), pitch range, loudness and clarity."),
             ("Word-by-word comparison", "Word i of your delivery is compared with word i of the ideal one: z = (you − ideal) / spread of the ideal."),
             ("Temporal grounding", "Words with |z| above 1.8, smoothed over 3 words, merge into flaw regions with exact start and end times."),
             ("Explanation and score", "Each region gets its cause in numbers. Each criterion = 10 − 1.5 × mean|z| − 15 × share of the speech "
                                       "inside its flaw regions. The same audio always gets the same score.")]
    rows = "".join(f"<div style='display:grid;grid-template-columns:2.6rem 1fr;gap:12px;padding:14px 0;border-top:1px solid var(--line)'>"
                   f"<div style='font-family:Spectral,serif;font-size:1.7rem;line-height:1;color:var(--you)'>{i + 1}</div>"
                   f"<div><div style='font-weight:600;margin-bottom:3px'>{t}</div><div class='muted'>{body}</div></div></div>"
                   for i, (t, body) in enumerate(steps))
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        st.markdown(f"<div class='section-title'>From recording to feedback</div>{rows}", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'><div class='section-title' style='font-size:18px'>Why compare with the same text?</div>"
                    "<div class='muted'>Good delivery depends on the words. A pause before “ask not” is a strength; the same pause "
                    "mid-phrase is a flaw. Comparing word by word with an ideal reading captures that.</div>"
                    "<div class='section-title' style='font-size:18px;margin-top:1rem'>Fair across voices</div>"
                    "<div class='muted'>Pitch is measured in semitones from each speaker's own median and loudness from each "
                    "speaker's own level, so a student on a phone is compared fairly with JFK in 1961.</div>"
                    "<div class='section-title' style='font-size:18px;margin-top:1rem'>Reproducible</div>"
                    "<div class='muted'>No LLM and no randomness. Every score is a fixed formula on the z-scores.</div></div>",
                    unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- login
def login_page():
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown(f"<div class='login-brand'><div></div><div class='mid'>{logo(96)}<div class='name'>OrateLens</div>"
                    f"<div class='tag'>Every word, measured.</div></div><div class='foot'>Verveathon 2K26</div></div>",
                    unsafe_allow_html=True)
    with right:
        with st.container(key="signin"):
            st.markdown("<div class='signin-title'>Sign in</div>", unsafe_allow_html=True)
            role = seg("Role", ["Participant", "Judge"], key="role_pick")
            name = st.text_input("Name", placeholder="Your name")
            code = ""
            if role == "Judge":
                code = st.text_input("Judge code", type="password", placeholder="Access code")
            if st.button("Sign in", type="primary", width="stretch"):
                if not name.strip():
                    st.error("Enter your name to sign in.")
                elif role == "Judge" and code.strip() != JUDGE_CODE:
                    st.error("That judge code doesn't match.")
                else:
                    st.session_state.user = {"name": name.strip(), "role": "judge" if role == "Judge" else "participant"}
                    st.session_state.page = "Leaderboard" if role == "Judge" else "New attempt"
                    st.rerun()
            st.markdown(f"<div class='demo-code'>Demo judge code {JUDGE_CODE}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- participant
def page_new_attempt(user):
    own = bool(custom_baseline) and st.session_state.get("bmode") == "My own recording"
    page_header("New attempt", [chip(f"Ideal: {st.session_state.get('base_label') or 'my own'}" if own else "Ideal: JFK, 1961", ideal=True)])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True, key="panel_rec"):
            st.markdown("<div class='section-title'>Recording</div>", unsafe_allow_html=True)
            mode = seg("Source", ["Record", "Upload", "Sample"], key="src")
            chosen = None
            if mode == "Record":
                rec = st.audio_input("Record your delivery", label_visibility="collapsed")
                if rec is not None:
                    rec.name = "recording.wav"
                    chosen = rec
                    if audio_check is not None:
                        try:
                            chk = audio_check(rec.getvalue())
                        except Exception:
                            chk = None
                        if chk and chk["verdict"] == "ok":
                            st.markdown(f"<div class='status ok'><b>Mic check passed</b><span>{chk['speech_s']:.1f} s audible</span></div>",
                                        unsafe_allow_html=True)
                        elif chk:
                            st.markdown(f"<div class='status bad'><b>Mic check failed</b><span>{chk['message']}</span></div>",
                                        unsafe_allow_html=True)
                            with st.expander("How to fix the microphone", expanded=chk["verdict"] == "silent"):
                                st.markdown(MIC_HELP)
            elif mode == "Upload":
                chosen = st.file_uploader("Upload your recording", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
            else:
                f = st.selectbox("Sample", list(DEMO_SPEAKERS), label_visibility="collapsed",
                                 format_func=lambda x: f"{DEMO_SPEAKERS[x]} ({Path(x).stem.split('_', 2)[-1].replace('_', ' ')})")
                chosen = ROOT / f
            go_btn = st.button("Analyse delivery", type="primary", width="stretch", disabled=chosen is None)
    with c2:
        with st.container(border=True, key="panel_script"):
            h, s = st.columns([1, 1.1], vertical_alignment="center")
            h.markdown("<div class='section-title'>Script</div>", unsafe_allow_html=True)
            options = ["JFK, 1961 inaugural"] + (["My own recording"] if custom_baseline else [])
            s.selectbox("Ideal", options, key="bmode", label_visibility="collapsed")
            if not own:
                st.markdown(f"<div class='script'>{DEFAULT_TRANSCRIPT}</div>"
                            f"<div class='meta-row'><span>{len(re.findall(r"[A-Za-z0-9']+", DEFAULT_TRANSCRIPT))} words</span>"
                            f"<span>Read it exactly as written</span></div>", unsafe_allow_html=True)
            else:
                st.text_input("Ideal speaker", key="base_label", placeholder="e.g. Coach")
                st.file_uploader("Ideal recording", type=["wav", "mp3", "m4a"], key="base_up")
                st.text_area("Script of the ideal recording", key="own_script", height=200, placeholder="Paste the exact words")
    if go_btn:
        baseline, text = "jfk", DEFAULT_TRANSCRIPT
        if own:
            text = st.session_state.get("own_script", "") or ""
            if st.session_state.get("base_up") is None or not text.strip():
                with c1:
                    st.error("Add the ideal recording and its script to use your own ideal.")
                return
            bpath = to_wav(st.session_state.base_up, prefix="baseline_")
            baseline = custom_baseline(bpath, text, st.session_state.get("base_label") or "Custom ideal")
        with c1:
            path = chosen if isinstance(chosen, Path) else to_wav(chosen)
            try:
                result = run_analysis(path, text, baseline)
            except NoSpeechError as e:
                st.markdown(f"<div class='status bad'><b>Not scored</b><span>{e}</span></div>", unsafe_allow_html=True)
                with st.expander("How to fix the microphone"):
                    st.markdown(MIC_HELP)
                return
        rec = save_attempt(user["name"], result, path, baseline)
        st.session_state.p_result, st.session_state.p_audio = result, str(path)
        st.session_state.p_baseline, st.session_state.p_meta = baseline, rec
        st.session_state.page = "My analysis"
        st.rerun()


def page_my_analysis():
    res = st.session_state.get("p_result")
    if not res:
        page_header("My analysis")
        st.markdown("<div class='glass muted'>No analysis yet. Record or upload a delivery in New attempt.</div>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Start a new attempt", type="primary"):
            st.session_state.page = "New attempt"
            st.rerun()
        return
    meta = st.session_state.get("p_meta", {})
    chips = ([chip(f"Attempt {meta['n']}, {meta['time']}")] if meta.get("n") else []) + \
            [chip(f"Ideal: {ideal_name(res.get('baseline_label'))}", ideal=True)]
    page_header("My analysis", chips)
    render_analysis(res, st.session_state.p_audio, "p", st.session_state.get("p_baseline", "jfk"))


def page_progress(user):
    page_header("My progress")
    mine = [a for a in load_attempts() if a["participant"].lower() == user["name"].lower()]
    if not mine:
        st.markdown("<div class='glass muted'>Your score trend appears here after your first analysed attempt.</div>",
                    unsafe_allow_html=True)
        return
    mine.sort(key=lambda a: a["ts"])
    latest, best = mine[-1], max(a["overall"] for a in mine)
    delta = latest["overall"] - mine[0]["overall"]
    c = st.columns(4)
    c[0].markdown(kpi("Latest score", latest["overall"], 100), unsafe_allow_html=True)
    c[1].markdown(kpi("Best score", best, 100), unsafe_allow_html=True)
    c[2].markdown(f"<div class='glass kpi'><div class='label'>Change since first</div><div class='value' "
                  f"style='color:{'#7BD3A4' if delta >= 0 else '#E86F8E'}'>{delta:+d}</div></div>", unsafe_allow_html=True)
    c[3].markdown(kpi("Attempts", len(mine), color=False), unsafe_allow_html=True)
    st.write("")
    g1, g2 = st.columns([1.3, 1])
    with g1:
        x = [f"#{i + 1}, {a['time']}" for i, a in enumerate(mine)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[a["overall"] for a in mine], name="Overall", mode="lines+markers",
                                 line=dict(color=YOU, width=3), marker=dict(size=9)))
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
    st.markdown("<div class='section-title'>Focus for your next attempt</div>", unsafe_allow_html=True)
    fc = st.columns(2)
    for col, (k, v) in zip(fc, weakest):
        col.markdown(f"<div class='glass'><b>{RUBRIC[k]}</b> <span class='serif' style='color:{score_color(v, 10)}'>{v} / 10</span>"
                     f"<div class='muted' style='margin-top:6px'>{TIPS[k]}</div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- judge
def seed_demo_attempts():
    attempts = load_attempts()
    have = {a["file"] for a in attempts}
    for f, name in DEMO_SPEAKERS.items():
        p = ROOT / f
        if p.exists() and p.name not in have:
            save_attempt(name, analyze(p), p)


def judge_board():
    if not st.session_state.get("seeded"):
        with st.spinner("Loading participants..."):
            seed_demo_attempts()
        st.session_state.seeded = True
    latest = {}
    for a in sorted(load_attempts(), key=lambda a: a["ts"]):
        latest[a["participant"]] = a
    return latest, sorted(latest.values(), key=lambda a: -a["overall"])


def page_leaderboard(board):
    page_header("Leaderboard", [chip("Ideal: JFK, 1961", ideal=True)])
    flaws = Counter()
    for a in board:
        flaws.update({k: n for k, n in a.get("flaw_counts", {}).items() if n})
    common = FLAW_NAMES[flaws.most_common(1)[0][0]] if flaws else "None"
    c = st.columns(4)
    c[0].markdown(kpi("Participants", len(board), color=False), unsafe_allow_html=True)
    avg = int(round(np.mean([a["overall"] for a in board]))) if board else 0
    c[1].markdown(kpi("Average score", avg, 100), unsafe_allow_html=True)
    c[2].markdown(kpi("Top score", board[0]["overall"] if board else 0, 100), unsafe_allow_html=True)
    c[3].markdown(f"<div class='glass kpi'><div class='label'>Most common flaw</div><div class='value' style='font-size:30px;"
                  f"line-height:1.47;color:#F29AB1'>{common}</div></div>", unsafe_allow_html=True)
    head = ("<tr><th>Rank</th><th>Participant</th><th>Score</th>" + "".join(f"<th class='n'>{n}</th>" for n in RUBRIC.values())
            + "<th>Flaws</th></tr>")
    rows = []
    for i, a in enumerate(board):
        sc = score_color(a["overall"], 100)
        crit = "".join(f"<td class='n' style='color:{'#F29AB1' if v < 5 else 'var(--chalk)'}'>{v:.1f}</td>"
                       for v in (a["rubric"][k] for k in RUBRIC))
        rows.append(f"<tr><td><span class='rank{' first' if i == 0 else ''}'>{i + 1}</span></td>"
                    f"<td><b>{a['participant']}</b><div class='muted' style='font-size:13px'>{a['time']}</div></td>"
                    f"<td><span class='score'><b style='color:{sc}'>{a['overall']}</b>"
                    f"<span class='track'><i style='width:{a['overall']}%;background:{sc}'></i></span></span></td>"
                    f"{crit}<td class='muted'>{a.get('regions', 0)}</td></tr>")
    st.markdown(f"<div class='board'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>",
                unsafe_allow_html=True)


def page_compare(board, latest):
    page_header("Compare")
    names = [a["participant"] for a in board]
    pick = st.multiselect("Participants to compare", names, default=names[:3])
    if pick:
        st.plotly_chart(radar([(n, latest[n]["rubric"]) for n in pick], 460), width="stretch")
        df = pd.DataFrame([{"Participant": n, "Overall": latest[n]["overall"],
                            **{RUBRIC[k]: v for k, v in latest[n]["rubric"].items()}, "Flaw regions": latest[n]["regions"]}
                           for n in pick])
        st.dataframe(df, width="stretch", hide_index=True)


def page_inspect(board, latest):
    page_header("Inspect")
    if not board:
        st.markdown("<div class='glass muted'>No participants yet.</div>", unsafe_allow_html=True)
        return
    who = st.selectbox("Participant", [a["participant"] for a in board])
    a = latest[who]
    path = Path(a["audio_path"])
    if not path.exists():
        st.warning("The audio for this attempt is no longer available.")
        return
    b = a.get("baseline", "jfk")
    st.audio(str(path))
    render_analysis(analyze(path, None, baseline=b), str(path), "j", b)


# ------------------------------------------------------------------------------------------- router
def sidebar(user, pages):
    with st.sidebar:
        st.markdown(f"<div class='sb-brand'>{logo(28)}OrateLens</div>", unsafe_allow_html=True)
        current = st.session_state.get("page", pages[0][0])
        for name, icon in pages:
            if st.button(name, icon=icon, key=f"nav_{name}", width="stretch",
                         type="primary" if name == current else "secondary"):
                st.session_state.page = name
                st.rerun()
        judge = user["role"] == "judge"
        st.markdown(f"<div class='sb-user'><span class='av{' judge' if judge else ''}'>{user['name'][:1].upper()}</span>"
                    f"<span><b>{user['name']}</b><span class='r'>{'Judge' if judge else 'Participant'}</span></span></div>",
                    unsafe_allow_html=True)
        if st.button("Sign out", icon=":material/logout:", key="signout", width="stretch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


user = st.session_state.get("user")
if not user:
    login_page()
elif user["role"] == "judge":
    if st.session_state.get("page") not in dict(J_PAGES):
        st.session_state.page = "Leaderboard"
    sidebar(user, J_PAGES)
    latest, board = judge_board()
    page = st.session_state.page
    if page == "Leaderboard":
        page_leaderboard(board)
    elif page == "Compare":
        page_compare(board, latest)
    elif page == "Inspect":
        page_inspect(board, latest)
    elif page == "Dataset and evaluation":
        page_header("Dataset and evaluation")
        render_dataset_eval()
    else:
        page_header("How it works")
        render_how()
else:
    if st.session_state.get("page") not in dict(P_PAGES):
        st.session_state.page = "New attempt"
    sidebar(user, P_PAGES)
    page = st.session_state.page
    if page == "New attempt":
        page_new_attempt(user)
    elif page == "My analysis":
        page_my_analysis()
    elif page == "My progress":
        page_progress(user)
    else:
        page_header("How it works")
        render_how()
