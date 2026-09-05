# =============================================================================
# SETUP
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
# streamlit run ml_showcase.py
#
# API keys: copy .env.example → .env  (never commit secrets)
# First YOLO / MiDaS run downloads model weights (~90MB) into cache.
# =============================================================================
"""
AI / ML Playground — interactive showcase for teaching classical ML → CV → GenAI → tooling.
Run: streamlit run ml_showcase.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Load .env early (local only; .env is gitignored)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# Optional deps — never crash if missing
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

try:
    from ultralytics import YOLO as UltralyticsYOLO
except ImportError:
    UltralyticsYOLO = None

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None


def _refresh_optional_imports() -> None:
    """Re-try optional imports after user installs packages mid-session."""
    global UltralyticsYOLO, torch, hf_pipeline, Image, ImageDraw, ImageFont
    if UltralyticsYOLO is None:
        try:
            from ultralytics import YOLO as _Y

            UltralyticsYOLO = _Y
        except ImportError:
            pass
    if torch is None:
        try:
            import torch as _t

            torch = _t
        except ImportError:
            pass
    if hf_pipeline is None:
        try:
            from transformers import pipeline as _p

            hf_pipeline = _p
        except ImportError:
            pass
    if Image is None:
        try:
            from PIL import Image as _I, ImageDraw as _D, ImageFont as _F

            Image, ImageDraw, ImageFont = _I, _D, _F
        except ImportError:
            pass


SAMPLE_IMAGE_URLS = {
    "Bus (YOLO classic)": "https://ultralytics.com/images/bus.jpg",
    "Person (Zidane)": "https://ultralytics.com/images/zidane.jpg",
    "Room interior": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=640&q=80",
    "City street": "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=640&q=80",
}

# ---------------------------------------------------------------------------
# Constants & page config
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Home",
    "📈 Logistic Regression",
    "🧠 Word Embeddings",
    "👁️ Depth Estimation",
    "🎯 YOLO Vision",
    "✨ 3D Gaussian Splatting",
    "🍄 Mario",
    "🤖 OpenRouter AI",
    "🧰 Cursor & Agents",
    "🤗 Hugging Face",
    "📊 Kaggle",
    "☁️ Google Colab",
]

MARIO_URL = "https://sup-udh.github.io/mario/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
]

OPENROUTER_EMBED_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "qwen/qwen3-embedding-0.6b",
    "perplexity/pplx-embed-v1-0.6b",
]

OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1"

QUICK_PROMPTS = [
    "Explain logistic regression like I’m a first-year CSE student.",
    "What’s the difference between precision and recall?",
    "Explain embeddings with a king/queen analogy.",
    "Compare supervised vs reinforcement learning using Mario.",
    "When should I use Colab vs Kaggle vs a local laptop?",
]

st.set_page_config(
    page_title="AI / ML Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_openrouter_key() -> str:
    """Resolve API key from env / .env / Streamlit secrets — never hardcode."""
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENROUTER_API_KEY", "") or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "page_idx": 0,
        "presentation_mode": False,
        "lr_threshold": 0.5,
        "embed_words": "king queen man woman apple orange car bus dog cat",
        "embed_mode": "Demo Mode",
        "agent_goal": "Find the average of these numbers and explain it: 12, 18, 7, 23, 15",
        "agent_step": 0,
        "openrouter_context": "",
        "gs_seed": 42,
        "or_messages": [],
        "or_pending_prompt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def nav_to(idx: int) -> None:
    st.session_state.page_idx = int(np.clip(idx, 0, len(PAGES) - 1))


def next_page() -> None:
    nav_to(st.session_state.page_idx + 1)


def prev_page() -> None:
    nav_to(st.session_state.page_idx - 1)


def reset_demo() -> None:
    st.session_state.lr_threshold = 0.5
    st.session_state.agent_step = 0
    st.session_state.gs_seed = int(time.time()) % 10_000
    st.rerun()


# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
def inject_css(presentation: bool) -> None:
    font_scale = 1.18 if presentation else 1.0
    sidebar_css = (
        """
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        .block-container { padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
        """
        if presentation
        else ""
    )
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: {16 * font_scale}px;
}}

.stApp {{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(56, 189, 248, 0.12), transparent 55%),
    radial-gradient(900px 500px at 90% 0%, rgba(167, 139, 250, 0.10), transparent 50%),
    radial-gradient(800px 500px at 50% 100%, rgba(52, 211, 153, 0.06), transparent 55%),
    #07090f;
  color: #e8edf7;
}}

h1, h2, h3 {{
  letter-spacing: -0.02em;
  font-weight: 650 !important;
}}

.hero-title {{
  font-size: {2.6 * font_scale}rem;
  font-weight: 700;
  margin: 0 0 0.25rem 0;
  background: linear-gradient(120deg, #f8fafc 0%, #7dd3fc 45%, #c4b5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}

.hero-sub {{
  color: #94a3b8;
  font-size: {1.05 * font_scale}rem;
  margin-bottom: 1.5rem;
}}

.glass-card {{
  background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 16px;
  padding: 1.15rem 1.25rem;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.25);
  height: 100%;
  transition: border-color 0.2s ease, transform 0.2s ease;
}}

.glass-card:hover {{
  border-color: rgba(125, 211, 252, 0.45);
  transform: translateY(-2px);
}}

.glass-card h3 {{
  margin: 0 0 0.4rem 0;
  font-size: 1.05rem;
  color: #f1f5f9;
}}

.glass-card p {{
  margin: 0;
  color: #94a3b8;
  font-size: 0.92rem;
  line-height: 1.45;
}}

.path-step {{
  display: inline-block;
  padding: 0.45rem 0.9rem;
  margin: 0.2rem;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.25);
  background: rgba(15,23,42,0.65);
  color: #e2e8f0;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
}}

.path-arrow {{
  color: #64748b;
  margin: 0 0.15rem;
  font-family: 'IBM Plex Mono', monospace;
}}

.metric-chip {{
  display: inline-block;
  padding: 0.35rem 0.7rem;
  margin: 0.15rem;
  border-radius: 8px;
  background: rgba(30, 41, 59, 0.85);
  border: 1px solid rgba(100, 116, 139, 0.35);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
}}

.tp {{ color: #34d399; }}
.fp {{ color: #fbbf24; }}
.tn {{ color: #60a5fa; }}
.fn {{ color: #f87171; }}

.section-label {{
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: #64748b;
  margin-bottom: 0.35rem;
}}

.warn-box {{
  background: rgba(251, 191, 36, 0.08);
  border: 1px solid rgba(251, 191, 36, 0.35);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  color: #fde68a;
  margin: 0.75rem 0;
}}

.info-box {{
  background: rgba(56, 189, 248, 0.08);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  color: #bae6fd;
  margin: 0.75rem 0;
}}

.footer-note {{
  text-align: center;
  color: #64748b;
  font-size: 0.85rem;
  margin-top: 2.5rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(148,163,184,0.15);
}}

.flow-box {{
  text-align: center;
  padding: 0.65rem 0.5rem;
  border-radius: 10px;
  border: 1px solid rgba(148,163,184,0.25);
  background: rgba(15,23,42,0.7);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  color: #e2e8f0;
}}

div[data-testid="stMetric"] {{
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 12px;
  padding: 0.6rem 0.85rem;
}}

/* Streamlit widgets polish */
.stSlider label, .stSelectbox label, .stTextInput label, .stTextArea label {{
  color: #cbd5e1 !important;
}}

{sidebar_css}
</style>
""",
        unsafe_allow_html=True,
    )


def card_html(title: str, body: str, emoji: str = "") -> str:
    return f"""
<div class="glass-card">
  <h3>{emoji} {title}</h3>
  <p>{body}</p>
</div>
"""


def flow_vertical(steps: list[str]) -> None:
    html = ""
    for i, s in enumerate(steps):
        html += f'<div class="flow-box">{s}</div>'
        if i < len(steps) - 1:
            html += '<div style="text-align:center;color:#64748b;padding:0.15rem 0;">↓</div>'
    st.markdown(html, unsafe_allow_html=True)


def flow_horizontal(steps: list[str]) -> None:
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="path-step">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="path-arrow">→</span>')
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared plot styling
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,9,15,0.4)",
    font=dict(color="#e2e8f0", family="Space Grotesk"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(148,163,184,0.2)"),
)


def style_fig(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)")
    return fig


# ---------------------------------------------------------------------------
# Presentation chrome
# ---------------------------------------------------------------------------
def render_presentation_nav() -> None:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 1, 1])
    with c1:
        if st.button("← Previous", use_container_width=True, key="prev_btn"):
            prev_page()
            st.rerun()
    with c2:
        if st.button("Next →", use_container_width=True, key="next_btn"):
            next_page()
            st.rerun()
    with c3:
        st.markdown(
            f"<div style='text-align:center;color:#94a3b8;padding-top:0.5rem;'>"
            f"{PAGES[st.session_state.page_idx]} · {st.session_state.page_idx + 1}/{len(PAGES)}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c4:
        if st.button("R · Reset", use_container_width=True, key="reset_btn"):
            reset_demo()
    with c5:
        st.session_state.presentation_mode = st.toggle(
            "Present",
            value=st.session_state.presentation_mode,
            key="pres_toggle",
        )


# ===========================================================================
# 1. HOME
# ===========================================================================
def page_home() -> None:
    st.markdown('<div class="hero-title">AI / ML Playground</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">From classical ML → Computer Vision → Generative AI → AI tools</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("CLASSICAL ML", "Interactive logistic regression: boundaries, probabilities, thresholds.", "📈"),
        ("Computer Vision", "Depth estimation & YOLO detection labs with real or demo modes.", "👁️"),
        ("Generative AI", "Word embeddings, OpenRouter chat, conceptual 3D Gaussian splats.", "✨"),
        ("AI TOOLING", "Cursor, agents, Hugging Face, Kaggle, and Colab — the practical stack.", "🧰"),
    ]
    for col, (t, b, e) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(card_html(t, b, e), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Learning path</div>', unsafe_allow_html=True)
    flow_horizontal(
        ["Data", "ML", "Deep Learning", "Computer Vision", "Generative AI", "AI Agents"]
    )

    if not st.session_state.presentation_mode:
        st.markdown("### What you'll explore")
        with st.expander("Section guide", expanded=True):
            st.markdown(
                """
| Section | What you'll see |
|---|---|
| **Logistic Regression** | Live decision boundary, probability heatmap, threshold trade-offs |
| **Word Embeddings** | PCA/3D maps of word vectors & cosine similarity |
| **Depth Estimation** | Image → depth map → 3D surface (demo or API) |
| **YOLO Vision** | Object detection / seg / pose when Ultralytics is installed |
| **Gaussian Splatting** | Conceptual 3D Gaussian scene explorer |
| **Mario** | Embedded browser Mario + RL concepts |
| **OpenRouter** | Live LLM playground (needs API key) |
| **Cursor & Agents** | How modern AI coding & agent loops work |
| **HF / Kaggle / Colab** | Where students train, share, and ship models |
"""
            )

    st.markdown("<br>", unsafe_allow_html=True)
    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("▶ Start with Logistic Regression", type="primary", use_container_width=True):
            nav_to(1)
            st.rerun()
    with cta2:
        if st.button("🤖 Open OpenRouter chat", use_container_width=True):
            nav_to(PAGES.index("🤖 OpenRouter AI"))
            st.rerun()


# ===========================================================================
# 2. LOGISTIC REGRESSION
# ===========================================================================
@st.cache_data(show_spinner=False)
def generate_classification_data(
    n_samples: int, class_sep: float, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    # flip_y approximates label noise; scale with noise slider
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=min(0.4, max(0.0, noise)),
        random_state=seed,
    )
    return X, y


@st.cache_data(show_spinner=False)
def fit_logistic(X: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray, float]:
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    return clf.coef_.copy(), clf.intercept_.copy(), float(clf.C)


def predict_proba_grid(
    coef: np.ndarray, intercept: np.ndarray, xx: np.ndarray, yy: np.ndarray
) -> np.ndarray:
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = grid @ coef.T + intercept
    # sigmoid
    prob = 1.0 / (1.0 + np.exp(-z.ravel()))
    return prob.reshape(xx.shape)


def page_logistic() -> None:
    st.markdown("## 📈 Logistic Regression Lab")
    if not st.session_state.presentation_mode:
        st.caption(
            "The model predicts a **probability**. A threshold turns that probability into a class."
        )

    with st.sidebar if not st.session_state.presentation_mode else st.container():
        st.markdown("#### Dataset & model")
        n_samples = st.slider("Samples", 50, 800, 300, 25, key="lr_n")
        class_sep = st.slider("Class separation", 0.5, 3.0, 1.5, 0.1, key="lr_sep")
        noise = st.slider("Label noise (flip_y)", 0.0, 0.35, 0.05, 0.01, key="lr_noise")
        seed = st.slider("Random seed", 0, 100, 42, key="lr_seed")
        C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01, key="lr_C")
        threshold = st.slider(
            "Classification threshold",
            0.01,
            0.99,
            float(st.session_state.lr_threshold),
            0.01,
            key="lr_thr_slider",
        )
        st.session_state.lr_threshold = threshold

    X, y = generate_classification_data(n_samples, class_sep, noise, seed)
    coef, intercept, _ = fit_logistic(X, y, C)

    # probabilities for points
    z_pts = X @ coef.T + intercept
    probs = (1.0 / (1.0 + np.exp(-z_pts.ravel()))).ravel()
    y_pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("Threshold", f"{threshold:.2f}")

    # Grid for heatmap
    pad = 0.8
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    res = 120
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, res),
        np.linspace(y_min, y_max, res),
    )
    zz = predict_proba_grid(coef, intercept, xx, yy)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, res),
            y=np.linspace(y_min, y_max, res),
            z=zz,
            colorscale=[
                [0.0, "#0f172a"],
                [0.5, "#334155"],
                [1.0, "#7c3aed"],
            ],
            opacity=0.75,
            contours=dict(showlines=False),
            colorbar=dict(title="P(y=1|x)", thickness=14),
            name="Probability",
            hovertemplate="P(y=1)=%{z:.3f}<extra></extra>",
        )
    )
    # Decision boundary at current threshold
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, res),
            y=np.linspace(y_min, y_max, res),
            z=zz,
            showscale=False,
            contours=dict(
                start=threshold,
                end=threshold,
                size=0.001,
                coloring="lines",
            ),
            line=dict(width=3, color="#f8fafc"),
            name="Decision boundary",
            hoverinfo="skip",
        )
    )

    # Points colored by true class, symbol by predicted
    for cls, color in [(0, "#38bdf8"), (1, "#f472b6")]:
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"True class {cls}",
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(
                        width=2,
                        color=np.where(y_pred[mask] == cls, color, "#fbbf24"),
                    ),
                    symbol=np.where(y_pred[mask] == cls, "circle", "x"),
                ),
                customdata=np.c_[probs[mask], y_pred[mask]],
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>"
                    "P(y=1)=%{customdata[0]:.3f}<br>"
                    "pred=%{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Decision boundary + probability field",
        xaxis_title="Feature x₁",
        yaxis_title="Feature x₂",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    style_fig(fig, height=560 if st.session_state.presentation_mode else 500)
    st.plotly_chart(fig, use_container_width=True)

    # Confusion visual + coefficients
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Confusion matrix (at current threshold)")
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                colorscale="Blues",
                text=[[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]],
                texttemplate="%{text}",
                textfont=dict(size=16, color="#f8fafc"),
                showscale=False,
                hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
            )
        )
        style_fig(cm_fig, height=280)
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown(
            f"""
<span class="metric-chip tp">TP {tp}</span>
<span class="metric-chip fp">FP {fp}</span>
<span class="metric-chip tn">TN {tn}</span>
<span class="metric-chip fn">FN {fn}</span>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Model coefficients")
        st.code(
            f"w₁ = {coef[0, 0]:.4f}\nw₂ = {coef[0, 1]:.4f}\nb  = {intercept[0]:.4f}\n"
            f"C  = {C:.4f}",
            language="text",
        )
        st.markdown(
            """
<div class="info-box">
The model does <b>not</b> directly predict a class. It predicts a probability
<code>P(y=1 | x)</code>, and the threshold converts that probability into a class.
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("What’s actually happening? (sigmoid)", expanded=not st.session_state.presentation_mode):
        st.markdown(
            r"""
1. Linear score: \( z = w_1 x_1 + w_2 x_2 + b \)
2. Probability via sigmoid: \( P(y=1\mid x) = \sigma(z) = \dfrac{1}{1+e^{-z}} \)
3. Decision rule:
   - if \( P(y=1\mid x) > \) **threshold** → predict class **1**
   - otherwise → predict class **0**

Move the threshold: lower it → more class-1 predictions (↑ recall, often ↓ precision).
Yellow **×** markers are misclassified points under the current threshold.
"""
        )
        # Mini sigmoid plot
        zs = np.linspace(-6, 6, 200)
        sig = 1 / (1 + np.exp(-zs))
        sfig = go.Figure()
        sfig.add_trace(go.Scatter(x=zs, y=sig, mode="lines", name="sigmoid", line=dict(color="#7dd3fc", width=3)))
        sfig.add_hline(y=threshold, line_dash="dash", line_color="#f472b6", annotation_text=f"thr={threshold:.2f}")
        sfig.update_layout(title="σ(z) and threshold", xaxis_title="z", yaxis_title="P(y=1)")
        style_fig(sfig, height=280)
        st.plotly_chart(sfig, use_container_width=True)

    st.session_state.openrouter_context = (
        f"Logistic regression demo: C={C}, threshold={threshold}, "
        f"accuracy={acc:.3f}, precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}, "
        f"coef=[{coef[0,0]:.3f},{coef[0,1]:.3f}], intercept={intercept[0]:.3f}."
    )
    if st.button("Ask OpenRouter to explain this demo", key="lr_ask_ai"):
        st.session_state.or_pending_prompt = (
            "Explain this live logistic-regression demo to a first-year CSE student:\n\n"
            + st.session_state.openrouter_context
            + "\n\nCover: probability vs class, threshold trade-offs, and what the coefficients mean."
        )
        nav_to(PAGES.index("🤖 OpenRouter AI"))
        st.rerun()


# ===========================================================================
# 3. WORD EMBEDDINGS
# ===========================================================================
def _token_seed(word: str) -> int:
    return int(hashlib.md5(word.lower().encode()).hexdigest()[:8], 16)


def demo_embeddings(words: list[str], dim: int = 64) -> np.ndarray:
    """Deterministic pseudo-embeddings with semantic-ish clustering heuristics."""
    rng_base = {}
    clusters = {
        "royalty": {"king", "queen", "prince", "princess", "royal"},
        "gender_m": {"man", "boy", "he", "king", "prince"},
        "gender_f": {"woman", "girl", "she", "queen", "princess"},
        "fruit": {"apple", "orange", "banana", "grape", "mango", "fruit"},
        "vehicle": {"car", "bus", "truck", "train", "bike", "vehicle"},
        "animal": {"dog", "cat", "bird", "horse", "fish", "animal"},
    }
    # Fixed cluster centroids
    centroids = {}
    for i, name in enumerate(clusters):
        rng = np.random.default_rng(1000 + i)
        centroids[name] = rng.normal(0, 1, dim)

    mats = []
    for w in words:
        rng = np.random.default_rng(_token_seed(w))
        base = rng.normal(0, 0.35, dim)
        wl = w.lower()
        for name, members in clusters.items():
            if wl in members:
                base = base + 0.85 * centroids[name]
        # tiny char n-gram signal for nearest-neighbor fun
        for ch in set(wl):
            base[_token_seed(ch) % dim] += 0.15
        base = base / (np.linalg.norm(base) + 1e-9)
        mats.append(base)
    return np.vstack(mats)


class EmbeddingProvider:
    """Abstraction for real embedding APIs — swap in a real endpoint later."""

    name = "base"

    def embed(self, texts: list[str], dim: int = 64) -> np.ndarray:
        raise NotImplementedError


class DemoEmbeddingProvider(EmbeddingProvider):
    name = "demo"

    def embed(self, texts: list[str], dim: int = 64) -> np.ndarray:
        return demo_embeddings(texts, dim=dim)


class OpenAICompatEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings (OpenRouter, OpenAI, etc.)."""

    name = "openai_compat"

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model

    def embed(self, texts: list[str], dim: int = 64) -> np.ndarray:
        if requests is None:
            raise RuntimeError("requests package is required for API mode")
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter asks for these attribution headers
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "http://localhost:8501"
            headers["X-Title"] = "AI ML Playground"
        resp = requests.post(
            url,
            headers=headers,
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
        data = resp.json()["data"]
        data = sorted(data, key=lambda d: d["index"])
        return np.array([d["embedding"] for d in data], dtype=float)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def page_embeddings() -> None:
    st.markdown("## 🧠 Word Embedding Playground")
    if not st.session_state.presentation_mode:
        st.caption("Embeddings represent words as vectors. Similar meanings → closer vectors.")

    words_raw = st.text_input(
        "Words (space-separated)",
        value=st.session_state.embed_words,
        key="embed_words_input",
    )
    st.session_state.embed_words = words_raw
    words = [w.strip() for w in words_raw.replace(",", " ").split() if w.strip()]
    if len(words) < 2:
        st.warning("Enter at least two words.")
        return

    or_key = get_openrouter_key()
    default_mode = "OpenRouter API" if or_key else "Demo Mode"
    mode = st.radio(
        "Embedding source",
        ["Demo Mode", "OpenRouter API", "Custom API"],
        index=["Demo Mode", "OpenRouter API", "Custom API"].index(default_mode),
        horizontal=True,
        key="embed_mode_radio",
    )
    dim = st.slider("Embedding dimension (demo only)", 16, 128, 64, 8, key="embed_dim")

    provider: EmbeddingProvider
    if mode == "Demo Mode":
        provider = DemoEmbeddingProvider()
        st.markdown(
            '<div class="info-box">Demo Mode uses deterministic pseudo-embeddings so the viz always works offline.</div>',
            unsafe_allow_html=True,
        )
    elif mode == "OpenRouter API":
        embed_model = st.selectbox(
            "OpenRouter embedding model",
            OPENROUTER_EMBED_MODELS,
            key="or_embed_model",
        )
        if not or_key:
            st.markdown(
                '<div class="warn-box">Set <code>OPENROUTER_API_KEY</code> in <code>.env</code> for live embeddings. Falling back to Demo.</div>',
                unsafe_allow_html=True,
            )
            provider = DemoEmbeddingProvider()
        else:
            provider = OpenAICompatEmbeddingProvider(
                or_key, OPENROUTER_EMBED_URL, embed_model
            )
            st.markdown(
                f'<div class="info-box">Using OpenRouter embeddings · <code>{embed_model}</code></div>',
                unsafe_allow_html=True,
            )
    else:
        api_key = (
            os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or or_key
            or ""
        )
        base_url = st.text_input(
            "Embeddings base URL",
            value=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        )
        model = st.text_input(
            "Embedding model",
            value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        if not api_key:
            st.markdown(
                '<div class="warn-box">Set an embedding API key. Falling back to Demo.</div>',
                unsafe_allow_html=True,
            )
            provider = DemoEmbeddingProvider()
        else:
            provider = OpenAICompatEmbeddingProvider(api_key, base_url, model)

    try:
        with st.spinner("Computing embeddings…"):
            emb = provider.embed(words, dim=dim)
    except Exception as e:
        st.error(f"API embedding failed — using Demo Mode. ({e})")
        emb = DemoEmbeddingProvider().embed(words, dim=dim)

    # PCA
    n_comp2 = min(2, emb.shape[0], emb.shape[1])
    pca2 = PCA(n_components=n_comp2, random_state=0)
    xy = pca2.fit_transform(emb)
    if n_comp2 == 1:
        xy = np.c_[xy, np.zeros(len(words))]

    tab2d, tab3d, tabsim = st.tabs(["2D PCA", "3D PCA", "Similarity"])

    with tab2d:
        fig = go.Figure(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers+text",
                text=words,
                textposition="top center",
                marker=dict(size=14, color=np.arange(len(words)), colorscale="Turbo", line=dict(width=1, color="#0f172a")),
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
            )
        )
        fig.update_layout(title=f"PCA projection · dim={emb.shape[1]} · provider={provider.name}")
        style_fig(fig, height=520 if st.session_state.presentation_mode else 460)
        st.plotly_chart(fig, use_container_width=True)
        var = pca2.explained_variance_ratio_.sum() * 100
        st.caption(f"Explained variance (2D): {var:.1f}%")

    with tab3d:
        n_comp3 = min(3, emb.shape[0], emb.shape[1])
        if n_comp3 < 3:
            st.info("Need at least 3 words/dims for a meaningful 3D PCA.")
        else:
            pca3 = PCA(n_components=3, random_state=0)
            xyz = pca3.fit_transform(emb)
            fig3 = go.Figure(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers+text",
                    text=words,
                    marker=dict(size=6, color=np.arange(len(words)), colorscale="Turbo"),
                )
            )
            fig3.update_layout(title="3D PCA embedding space", scene=dict(bgcolor="rgba(7,9,15,0.5)"))
            style_fig(fig3, height=560)
            st.plotly_chart(fig3, use_container_width=True)

    with tabsim:
        c1, c2 = st.columns(2)
        w1 = c1.selectbox("Word A", words, index=0, key="sim_a")
        w2 = c2.selectbox("Word B", words, index=min(1, len(words) - 1), key="sim_b")
        i1, i2 = words.index(w1), words.index(w2)
        sim = cosine_sim(emb[i1], emb[i2])
        st.metric("Cosine similarity", f"{sim:.3f}")

        # Nearest neighbors for word A
        sims = [(words[j], cosine_sim(emb[i1], emb[j])) for j in range(len(words)) if j != i1]
        sims.sort(key=lambda t: -t[1])
        st.markdown(f"**Nearest neighbors of `{w1}`**")
        nn_df = pd.DataFrame(sims[:5], columns=["word", "cosine"])
        st.dataframe(nn_df, use_container_width=True, hide_index=True)

        # Similarity heatmap
        S = np.zeros((len(words), len(words)))
        for i in range(len(words)):
            for j in range(len(words)):
                S[i, j] = cosine_sim(emb[i], emb[j])
        hfig = px.imshow(S, x=words, y=words, color_continuous_scale="Viridis", zmin=-1, zmax=1)
        hfig.update_layout(title="Pairwise cosine similarity")
        style_fig(hfig, height=420)
        st.plotly_chart(hfig, use_container_width=True)

    st.markdown(
        """
<div class="info-box">
Embeddings represent words as vectors. Similar meanings tend to produce vectors that are closer together
(higher cosine similarity). Classic analogy: <code>king − man + woman ≈ queen</code>.
</div>
""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# 4. DEPTH ESTIMATION
# ===========================================================================
def _pil_from_upload(upload) -> Optional[Any]:
    if Image is None:
        st.error("Install Pillow: `pip install Pillow`")
        return None
    try:
        return Image.open(upload).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_sample_image_bytes(url: str) -> Optional[bytes]:
    if requests is None:
        return None
    try:
        r = requests.get(url, timeout=45, headers={"User-Agent": "AI-ML-Playground/1.0"})
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def load_named_sample(name: str) -> Optional[Any]:
    """Load a curated sample photo, with a procedural fallback."""
    url = SAMPLE_IMAGE_URLS.get(name)
    if url and Image is not None:
        data = fetch_sample_image_bytes(url)
        if data:
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                pass
    return _sample_scene_image(480)


def _box_blur(depth: np.ndarray, k: int) -> np.ndarray:
    """Safe integral-image box blur (fixes odd/even kernel sizing)."""
    k = max(3, int(k) | 1)
    pad = k // 2
    padded = np.pad(depth, pad, mode="edge")
    # summed-area table with leading zeros
    sat = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    sat = sat.cumsum(axis=0).cumsum(axis=1)
    h, w = depth.shape
    out = (
        sat[k : h + k, k : w + k]
        - sat[0:h, k : w + k]
        - sat[k : h + k, 0:w]
        + sat[0:h, 0:w]
    )
    return out / float(k * k)


def demo_depth_map(img_arr: np.ndarray) -> np.ndarray:
    """Visually convincing depth-like map from image processing (not a real model)."""
    gray = img_arr.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radial = 1.0 - np.sqrt(((yy - cy) / (cy + 1e-6)) ** 2 + ((xx - cx) / (cx + 1e-6)) ** 2)
    radial = np.clip(radial, 0, 1)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = (gx + gy) / (float(gray.max()) + 1e-6)
    gnorm = gray / (float(gray.max()) + 1e-6)
    # Darker / centered regions → nearer (presentation-friendly prior)
    depth = 0.50 * (1.0 - gnorm) + 0.35 * radial + 0.15 * (1.0 - np.clip(edges, 0, 1))
    depth = _box_blur(depth, max(5, min(h, w) // 35))
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    return depth.astype(np.float32)


@st.cache_resource(show_spinner=True)
def load_midas_small():
    """MiDaS small via torch.hub. Nested hub loads need trust_repo patched."""
    _refresh_optional_imports()
    if torch is None:
        return None
    try:
        import torch.hub as _hub

        _orig_load = _hub.load

        def _trusted_load(repo_or_dir, model, *args, **kwargs):
            kwargs.setdefault("trust_repo", True)
            return _orig_load(repo_or_dir, model, *args, **kwargs)

        _hub.load = _trusted_load  # type: ignore[method-assign]
        try:
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            transform = transforms.small_transform
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            return {"model": model, "transform": transform, "device": device, "kind": "torch"}
        finally:
            _hub.load = _orig_load  # type: ignore[method-assign]
    except Exception:
        return None


@st.cache_resource(show_spinner=True)
def load_midas_onnx():
    """Offline-friendly MiDaS small ONNX (downloaded once into local cache)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if requests is None:
        return None
    cache_dir = Path.home() / ".cache" / "ml_showcase"
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / "midas_v21_small.onnx"
    # Hugging Face ONNX mirror of MiDaS v2.1 small
    hf_url = (
        "https://huggingface.co/onnxmodels/midas-V2.1-small/"
        "resolve/main/model-small.onnx"
    )
    try:
        if not onnx_path.exists() or onnx_path.stat().st_size < 1_000_000:
            r = requests.get(hf_url, timeout=120, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1_000_000:
                onnx_path.write_bytes(r.content)
            else:
                return None
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return {"session": sess, "kind": "onnx"}
    except Exception:
        return None


def _run_midas_onnx(image: Any, bundle: dict) -> np.ndarray:
    sess = bundle["session"]
    img = np.array(image).astype(np.float32)
    h0, w0 = img.shape[:2]
    # MiDaS small typically 256
    net = 256
    resized = np.array(Image.fromarray(img.astype(np.uint8)).resize((net, net), Image.BILINEAR)).astype(
        np.float32
    )
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = resized / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x})[0]
    depth = np.squeeze(out).astype(np.float32)
    depth = np.array(Image.fromarray(depth).resize((w0, h0), Image.BILINEAR)).astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    return depth


@st.cache_resource(show_spinner=False)
def get_hf_depth_pipe(model_id: str):
    _refresh_optional_imports()
    if hf_pipeline is None:
        return None
    try:
        return hf_pipeline("depth-estimation", model=model_id)
    except Exception:
        return None


def _run_midas(image: Any, bundle: dict) -> np.ndarray:
    model, transform, device = bundle["model"], bundle["transform"], bundle["device"]
    img_arr = np.array(image)
    sample = transform(img_arr)
    if isinstance(sample, dict):
        batch = sample["image"]
    else:
        batch = sample
    if hasattr(batch, "ndim") and batch.ndim == 3:
        batch = batch.unsqueeze(0)
    batch = batch.to(device)
    with torch.no_grad():
        pred = model(batch)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        depth = pred.squeeze().float().cpu().numpy()
    if Image is not None:
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize(image.size, Image.BILINEAR)
        depth = np.array(depth_img).astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    return depth


def run_depth_estimation(image: Any, model_name: str) -> tuple[np.ndarray, str, float]:
    """
    Returns (depth_map HxW float 0-1, status, elapsed_sec).
    Preference: torch MiDaS → ONNX MiDaS → transformers → HF API → classical demo.
    """
    t0 = time.perf_counter()
    img_arr = np.array(image)
    notes: list[str] = []

    hf_ids = {
        "Depth Anything": "LiheYoung/depth-anything-small-hf",
        "Depth Anything V2": "depth-anything/Depth-Anything-V2-Small-hf",
        "MiDaS": "Intel/dpt-hybrid-midas",
    }

    # 1) Local MiDaS (torch.hub)
    if model_name in ("MiDaS", "Depth Anything", "Depth Anything V2"):
        bundle = load_midas_small()
        if bundle is not None:
            try:
                depth = _run_midas(image, bundle)
                elapsed = time.perf_counter() - t0
                label = "Local MiDaS_small (torch.hub)"
                if model_name != "MiDaS":
                    label += f" · stand-in for {model_name}"
                return depth, label, elapsed
            except Exception as e:
                notes.append(f"MiDaS torch failed: {e}")

        # 1b) ONNX MiDaS
        onnx_b = load_midas_onnx()
        if onnx_b is not None and Image is not None:
            try:
                depth = _run_midas_onnx(image, onnx_b)
                elapsed = time.perf_counter() - t0
                return depth, "Local MiDaS small (ONNX)", elapsed
            except Exception as e:
                notes.append(f"MiDaS ONNX failed: {e}")

    # 2) Local transformers pipeline
    if hf_pipeline is not None and model_name in hf_ids:
        pipe = get_hf_depth_pipe(hf_ids[model_name])
        if pipe is not None:
            try:
                out = pipe(image)
                depth = out["depth"] if isinstance(out, dict) else out
                depth = np.array(depth).astype(np.float32)
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
                if depth.shape[:2] != (image.size[1], image.size[0]) and Image is not None:
                    depth = np.array(
                        Image.fromarray(depth).resize(image.size, Image.BILINEAR)
                    ).astype(np.float32)
                    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
                elapsed = time.perf_counter() - t0
                return depth, f"Local transformers · {model_name}", elapsed
            except Exception as e:
                notes.append(f"transformers failed: {e}")

    # 3) HF Inference API
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token and requests is not None and model_name in hf_ids and Image is not None:
        try:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            api_url = f"https://api-inference.huggingface.co/models/{hf_ids[model_name]}"
            r = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {hf_token}"},
                data=buf.getvalue(),
                timeout=90,
            )
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and "image" in ctype:
                depth_img = Image.open(io.BytesIO(r.content)).convert("L")
                depth = (
                    np.array(depth_img.resize(image.size, Image.BILINEAR)).astype(np.float32)
                    / 255.0
                )
                elapsed = time.perf_counter() - t0
                return depth, f"HF Inference API · {model_name}", elapsed
            notes.append(f"HF API {r.status_code}")
        except Exception as e:
            notes.append(f"HF API error: {e}")

    depth = demo_depth_map(img_arr)
    elapsed = time.perf_counter() - t0
    extra = (" · " + "; ".join(notes[:2])) if notes else ""
    status = f"Demo depth visualization — not model inference{extra}"
    return depth, status, elapsed


def _sample_scene_image(size: int = 420) -> Any:
    """Procedural indoor/outdoor-ish scene when network samples fail."""
    if Image is None:
        return None
    rng = np.random.default_rng(7)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # floor / wall gradient
    for y in range(size):
        t = y / size
        img[y, :] = [
            int(40 + 50 * t),
            int(45 + 55 * t),
            int(55 + 40 * (1 - t)),
        ]
    # "window" (bright far plane)
    img[40:140, 260:380] = [210, 220, 235]
    # "sofa" near
    img[240:360, 60:220] = [120, 70, 55]
    # "person-like" blob
    yy, xx = np.ogrid[0:size, 0:size]
    mask = ((xx - 300) ** 2) / 35**2 + ((yy - 220) ** 2) / 90**2 < 1
    img[mask] = [200, 160, 130]
    # noise texture
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _sample_gradient_image(size: int = 256) -> Any:
    return _sample_scene_image(size)


def page_depth() -> None:
    _refresh_optional_imports()
    st.markdown("## 👁️ Depth Estimation")
    if not st.session_state.presentation_mode:
        st.caption("Estimate relative depth from a single RGB image.")

    model_name = st.selectbox(
        "Model",
        ["MiDaS", "Depth Anything", "Depth Anything V2"],
        key="depth_model",
    )
    sample = st.selectbox(
        "Image source",
        ["Sample photos", "Upload", "Procedural scene"],
        key="depth_src",
    )
    image = None
    if sample == "Upload":
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            image = _pil_from_upload(up)
    elif sample == "Procedural scene":
        image = _sample_scene_image(480)
    else:
        pick = st.selectbox("Sample", list(SAMPLE_IMAGE_URLS.keys()), key="depth_sample_name")
        with st.spinner("Loading sample…"):
            image = load_named_sample(pick)

    if image is None:
        st.info("Upload an image or pick a sample to begin.")
        return

    max_side = 512
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1 and Image is not None:
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    with st.spinner("Estimating depth…"):
        depth, status, elapsed = run_depth_estimation(image, model_name)
    img_arr = np.array(image)

    live = not (
        "Demo" in status or "not model" in status.lower() or "fallback" in status.lower()
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model_name.split()[0])
    c2.metric("Resolution", f"{img_arr.shape[1]}×{img_arr.shape[0]}")
    c3.metric("Time", f"{elapsed*1000:.0f} ms")
    c4.metric("Status", "Live" if live else "Demo")

    if live:
        st.markdown(f'<div class="info-box">{status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">{status}</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(img_arr, use_container_width=True)
    with right:
        st.markdown("**Depth map**")
        dfig = px.imshow(depth, color_continuous_scale="Magma")
        dfig.update_layout(coloraxis_showscale=True, margin=dict(l=0, r=0, t=0, b=0))
        style_fig(dfig, height=360)
        dfig.update_xaxes(visible=False)
        dfig.update_yaxes(visible=False, autorange="reversed")
        st.plotly_chart(dfig, use_container_width=True)

    st.markdown("**Original + depth overlay**")
    # Proper colormap blend
    try:
        import matplotlib.cm as cm

        colored = (cm.magma(depth)[:, :, :3] * 255).astype(np.uint8)
        mix = (0.45 * img_arr.astype(float) + 0.55 * colored.astype(float)).clip(0, 255).astype(
            np.uint8
        )
    except Exception:
        tint = np.stack([depth * 255, depth * 80, (1 - depth) * 120], axis=-1)
        mix = (0.45 * img_arr.astype(float) + 0.55 * tint).clip(0, 255).astype(np.uint8)
    st.image(mix, use_container_width=True)

    st.markdown("**3D depth surface**")
    step = max(1, min(depth.shape) // 70)
    zs = depth[::step, ::step]
    fig3 = go.Figure(
        data=[go.Surface(z=zs, colorscale="Magma", showscale=True, opacity=0.95)]
    )
    fig3.update_layout(
        title="Depth as 3D surface",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="depth",
            bgcolor="rgba(7,9,15,0.5)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.45),
        ),
    )
    style_fig(fig3, height=520)
    st.plotly_chart(fig3, use_container_width=True)

    st.session_state.openrouter_context = (
        f"Depth estimation with {model_name}: status='{status}', "
        f"resolution={img_arr.shape[1]}x{img_arr.shape[0]}, time={elapsed*1000:.0f}ms."
    )

    with st.expander("How to enable / upgrade models"):
        st.markdown(
            """
**Already good on most laptops after:** `pip install ultralytics` (pulls PyTorch)  
Then this page uses **MiDaS_small** via `torch.hub`.

For exact Depth Anything weights:
```bash
pip install transformers
```
Optional cloud API: set `HF_TOKEN` in `.env`.
"""
        )
        st.caption(f"torch={'yes' if torch is not None else 'no'} · transformers={'yes' if hf_pipeline is not None else 'no'}")


# ===========================================================================
# 5. YOLO VISION
# ===========================================================================
@st.cache_resource(show_spinner=True)
def load_yolo_model(model_type: str):
    _refresh_optional_imports()
    if UltralyticsYOLO is None:
        return None
    weights = {
        "YOLO detect": "yolov8n.pt",
        "YOLO": "yolov8n.pt",
        "YOLO segmentation": "yolov8n-seg.pt",
        "YOLO pose": "yolov8n-pose.pt",
    }.get(model_type, "yolov8n.pt")
    try:
        model = UltralyticsYOLO(weights)
        # warm-up tiny predict to surface download errors early
        return model
    except Exception:
        return None


def run_yolo(image: Any, model_type: str, conf: float = 0.25) -> dict[str, Any]:
    """Run YOLO if ultralytics is available. Never fakes detections as real."""
    t0 = time.perf_counter()
    _refresh_optional_imports()
    model = load_yolo_model(model_type)
    if model is None:
        return {
            "ok": False,
            "reason": "Ultralytics not installed or weights failed to load.",
            "install": "pip install ultralytics",
            "elapsed": time.perf_counter() - t0,
        }
    try:
        results = model.predict(
            source=np.array(image),
            verbose=False,
            conf=conf,
            imgsz=640,
        )
        r0 = results[0]
        detections = []
        names = r0.names
        if r0.boxes is not None and len(r0.boxes):
            for box in r0.boxes:
                cls_id = int(box.cls.item())
                conf_v = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy().ravel().tolist()
                detections.append(
                    {
                        "class": names.get(cls_id, str(cls_id)),
                        "confidence": conf_v,
                        "xyxy": xyxy,
                    }
                )
        plotted = r0.plot()[:, :, ::-1]  # BGR → RGB
        elapsed = time.perf_counter() - t0
        avg_conf = (
            float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
        )
        return {
            "ok": True,
            "detections": detections,
            "image": plotted,
            "elapsed": elapsed,
            "avg_conf": avg_conf,
            "count": len(detections),
        }
    except Exception as e:
        return {
            "ok": False,
            "reason": f"Inference error: {e}",
            "install": "pip install ultralytics",
            "elapsed": time.perf_counter() - t0,
        }


def _draw_boxes(img_arr: np.ndarray, dets: list[dict], demo: bool = False) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return img_arr
    im = Image.fromarray(img_arr.copy())
    draw = ImageDraw.Draw(im)
    colors = [(56, 189, 248), (244, 114, 182), (52, 211, 153), (251, 191, 36)]
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{d['class']} {d['confidence']*100:.0f}%"
        if demo:
            label = f"[demo] {label}"
        tw = 8 * len(label) + 8
        draw.rectangle([x1, max(0, y1 - 18), x1 + tw, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(7, 9, 15))
    return np.array(im)


def _yolo_demo_overlay(img_arr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Clearly labelled demo boxes — NOT presented as real YOLO."""
    h, w = img_arr.shape[:2]
    demo = [
        {"class": "person", "confidence": 0.98, "xyxy": [w * 0.12, h * 0.18, w * 0.38, h * 0.92]},
        {"class": "car", "confidence": 0.94, "xyxy": [w * 0.42, h * 0.48, w * 0.88, h * 0.88]},
        {"class": "dog", "confidence": 0.91, "xyxy": [w * 0.55, h * 0.68, w * 0.74, h * 0.95]},
    ]
    return _draw_boxes(img_arr, demo, demo=True), demo


def page_yolo() -> None:
    _refresh_optional_imports()
    st.markdown("## 🎯 YOLO Vision Lab")
    if not st.session_state.presentation_mode:
        st.caption("Object detection / segmentation / pose with Ultralytics YOLOv8.")

    model_type = st.selectbox(
        "Model type",
        ["YOLO detect", "YOLO segmentation", "YOLO pose"],
        key="yolo_type",
    )
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05, key="yolo_conf")
    src = st.selectbox(
        "Source",
        ["Sample photos", "Upload", "Webcam (optional)"],
        key="yolo_src",
    )

    image = None
    if src == "Upload":
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], key="yolo_up")
        if up:
            image = _pil_from_upload(up)
    elif src == "Webcam (optional)":
        cam = st.camera_input("Capture a frame")
        if cam:
            image = _pil_from_upload(cam)
    else:
        pick = st.selectbox(
            "Sample",
            ["Bus (YOLO classic)", "Person (Zidane)", "City street", "Room interior"],
            key="yolo_sample_name",
        )
        with st.spinner("Loading sample…"):
            image = load_named_sample(pick)

    avail = UltralyticsYOLO is not None
    if not avail:
        st.markdown(
            '<div class="warn-box">Install for real detection: <code>pip install ultralytics</code> then rerun.</div>',
            unsafe_allow_html=True,
        )

    if image is None:
        st.info("Provide an image to run detection.")
        return

    # Keep laptop-friendly size
    max_side = 960
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    with st.spinner("Running YOLO…"):
        result = run_yolo(image, model_type, conf=conf)
    img_arr = np.array(image)

    if result["ok"]:
        st.markdown(
            '<div class="info-box">Real YOLO inference (Ultralytics YOLOv8)</div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects detected", result["count"])
        m2.metric("Avg confidence", f"{result['avg_conf']*100:.1f}%")
        m3.metric("Inference time", f"{result['elapsed']*1000:.0f} ms")
        st.image(result["image"], use_container_width=True, caption="Annotated detections")
        if result["detections"]:
            rows = [
                {
                    "class": d["class"],
                    "confidence": f"{d['confidence']*100:.0f}%",
                }
                for d in sorted(result["detections"], key=lambda x: -x["confidence"])
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            # confidence bars
            bar = go.Figure(
                go.Bar(
                    x=[d["confidence"] for d in result["detections"]],
                    y=[d["class"] for d in result["detections"]],
                    orientation="h",
                    marker_color="#38bdf8",
                )
            )
            bar.update_layout(title="Confidence by detection", xaxis_range=[0, 1])
            style_fig(bar, height=max(220, 40 * len(result["detections"])))
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("No objects above the confidence threshold — try lowering it.")
        st.session_state.openrouter_context = (
            f"YOLO {model_type}: detected {result['count']} objects, "
            f"avg conf={result['avg_conf']:.2f}, time={result['elapsed']*1000:.0f}ms. "
            f"Classes: {', '.join(sorted({d['class'] for d in result['detections']})) or 'none'}."
        )
    else:
        st.markdown(
            f'<div class="warn-box"><b>Real YOLO unavailable.</b> {result["reason"]}<br>'
            f'Install: <code>{result.get("install", "pip install ultralytics")}</code><br>'
            f"Below is a <b>labelled demo overlay</b> — not real detections.</div>",
            unsafe_allow_html=True,
        )
        demo_img, demo_dets = _yolo_demo_overlay(img_arr)
        st.image(demo_img, use_container_width=True, caption="Demo mode — illustrative boxes only")
        st.dataframe(
            pd.DataFrame(
                [
                    {"class": d["class"], "confidence": f"{d['confidence']*100:.0f}%"}
                    for d in demo_dets
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


# ===========================================================================
# 6. GAUSSIAN SPLATTING
# ===========================================================================
@st.cache_data(show_spinner=False)
def synth_gaussian_scene(seed: int, density: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = [
        ((0, 0, 0), (1.2, 0.4, 0.4), "#7dd3fc"),
        ((2.5, 0.5, -0.5), (0.5, 1.0, 0.5), "#c4b5fd"),
        ((-1.5, 1.2, 1.0), (0.6, 0.6, 0.9), "#34d399"),
        ((1.0, -1.5, 0.8), (0.8, 0.3, 0.7), "#f472b6"),
        ((0.2, 2.0, -1.2), (0.4, 0.4, 0.4), "#fbbf24"),
    ]
    rows = []
    per = max(20, density // len(clusters))
    for (cx, cy, cz), (sx, sy, sz), color in clusters:
        pts = rng.normal(0, 1, size=(per, 3))
        pts[:, 0] = pts[:, 0] * sx + cx
        pts[:, 1] = pts[:, 1] * sy + cy
        pts[:, 2] = pts[:, 2] * sz + cz
        opac = rng.uniform(0.35, 0.95, per)
        for i in range(per):
            rows.append(
                {
                    "x": pts[i, 0],
                    "y": pts[i, 1],
                    "z": pts[i, 2],
                    "color": color,
                    "opacity": float(opac[i]),
                    "scale": float(rng.uniform(2, 8)),
                }
            )
    return pd.DataFrame(rows)


def page_gaussian() -> None:
    st.markdown("## ✨ 3D Gaussian Splatting")
    st.markdown(
        '<div class="warn-box"><b>Conceptual Gaussian Splat Visualization</b> — '
        "synthetic Gaussians, not a real .ply / .splat renderer.</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Traditional 3D**")
        flow_vertical(["Mesh", "Triangles", "Textures"])
    with col_b:
        st.markdown("**Gaussian Splatting**")
        flow_vertical(["Scene", "Millions of 3D Gaussians", "Differentiable rasterization"])

    c1, c2, c3, c4 = st.columns(4)
    density = c1.slider("Density", 200, 4000, 1200, 100, key="gs_density")
    point_size = c2.slider("Point size", 1, 12, 4, key="gs_size")
    opacity = c3.slider("Opacity", 0.1, 1.0, 0.75, 0.05, key="gs_opacity")
    cam_dist = c4.slider("Camera distance", 3.0, 15.0, 6.0, 0.5, key="gs_cam")

    if st.button("Generate New Scene"):
        st.session_state.gs_seed = int(time.time()) % 10_000

    df = synth_gaussian_scene(st.session_state.gs_seed, density)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=df["color"],
                    opacity=opacity,
                    line=dict(width=0),
                ),
                hovertemplate="Gaussian splat<br>scale≈%{marker.size}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Synthetic Gaussian cloud (rotate / zoom)",
        scene=dict(
            bgcolor="rgba(7,9,15,0.55)",
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=cam_dist / 6, y=cam_dist / 6, z=cam_dist / 5)),
            aspectmode="data",
        ),
    )
    style_fig(fig, height=560 if st.session_state.presentation_mode else 500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Attributes of each Gaussian", expanded=not st.session_state.presentation_mode):
        st.markdown(
            """
| Attribute | Role |
|---|---|
| **Position** (μ) | Where the splat sits in 3D |
| **Scale** | Anisotropic size of the ellipsoid |
| **Rotation** | Orientation of the covariance |
| **Color** (SH) | View-dependent appearance |
| **Opacity** (α) | How strongly it contributes when blended |
"""
        )

    with st.expander("Optional: load a real splat / point cloud"):
        st.markdown(
            "If you have `plyfile` / Open3D installed, you could extend this section to load `.ply` point clouds. "
            "Full `.splat` rasterization typically needs a dedicated WebGL viewer."
        )
        up = st.file_uploader("Optional .ply (xyz only)", type=["ply"], key="ply_up")
        if up is not None:
            st.info("PLY parsing is optional; install `plyfile` and we can display XYZ points here.")
            try:
                import plyfile  # type: ignore

                plydata = plyfile.PlyData.read(io.BytesIO(up.read()))
                verts = plydata["vertex"]
                xs, ys, zs = verts["x"], verts["y"], verts["z"]
                pfig = go.Figure(
                    go.Scatter3d(
                        x=xs[:: max(1, len(xs) // 8000)],
                        y=ys[:: max(1, len(xs) // 8000)],
                        z=zs[:: max(1, len(xs) // 8000)],
                        mode="markers",
                        marker=dict(size=2, color="#7dd3fc", opacity=0.7),
                    )
                )
                style_fig(pfig, height=480)
                st.plotly_chart(pfig, use_container_width=True)
            except ImportError:
                st.warning("Install plyfile to preview: `pip install plyfile`")
            except Exception as e:
                st.error(f"Could not parse PLY: {e}")


# ===========================================================================
# 7. MARIO
# ===========================================================================
def page_mario() -> None:
    st.markdown("## 🍄 AI + Games")
    st.markdown(
        """
<div class="info-box">
This is a browser-based Super Mario Bros experience. Try controlling it with
<b>Arrow Keys</b>, <b>X</b> to jump, and <b>Z</b> to run.
</div>
""",
        unsafe_allow_html=True,
    )

    # iframe embed — may be blocked by X-Frame-Options
    try:
        import streamlit.components.v1 as components

        components.html(
            f"""
            <div style="border:1px solid rgba(148,163,184,0.25);border-radius:12px;overflow:hidden;">
              <iframe src="{MARIO_URL}" width="100%" height="480" style="border:0;background:#000;"
                allow="gamepad *; fullscreen *"></iframe>
            </div>
            <p style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem;">
              If the game appears blank, embedding is blocked — use the link below.
            </p>
            """,
            height=540,
        )
    except Exception:
        pass

    st.link_button("Open Mario in a new browser tab", MARIO_URL, type="primary")

    st.markdown("### Why is Mario interesting for AI?")
    st.markdown(
        """
Classic **reinforcement learning** playground:

- **State** — screen pixels / positions / velocities  
- **Action** — left, right, jump, run  
- **Reward** — progress right, coins, surviving  
- **Policy** — mapping from state → action  
- **Environment** — the Mario simulator itself  
"""
    )

    st.markdown('<div class="section-label">RL loop</div>', unsafe_allow_html=True)
    flow_horizontal(["STATE", "AGENT", "ACTION", "ENVIRONMENT", "REWARD", "STATE"])


# ===========================================================================
# 8. OPENROUTER
# ===========================================================================
def openrouter_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> tuple[bool, Any]:
    api_key = get_openrouter_key()
    if not api_key:
        return False, (
            "Missing OPENROUTER_API_KEY. Add it to a local `.env` file "
            "(see `.env.example`) or export it in your shell, then restart Streamlit."
        )
    if requests is None:
        return False, "Install requests: pip install requests"
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "AI ML Playground",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            },
            timeout=120,
            stream=stream,
        )
        if r.status_code != 200:
            body = r.text[:600] if not stream else ""
            if stream:
                try:
                    body = r.json().get("error", {}).get("message", r.text[:600])
                except Exception:
                    body = r.text[:600]
            return False, f"API error {r.status_code}: {body}"
        if not stream:
            data = r.json()
            return True, data["choices"][0]["message"]["content"]
        return True, r
    except Exception as e:
        return False, f"Request failed: {e}"


def _stream_openrouter(response) -> Generator[str, None, None]:
    """Yield text deltas from an OpenRouter SSE response."""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    yield piece
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue


def page_openrouter() -> None:
    st.markdown("## 🤖 OpenRouter AI Playground")
    if not st.session_state.presentation_mode:
        st.caption("One API key → many models. Great for demos without juggling vendor accounts.")

    key_present = bool(get_openrouter_key())
    if not key_present:
        st.markdown(
            '<div class="warn-box">No <code>OPENROUTER_API_KEY</code> found. '
            "Copy <code>.env.example</code> → <code>.env</code>, add your key, restart Streamlit.</div>",
            unsafe_allow_html=True,
        )
    else:
        masked = get_openrouter_key()
        tip = f"...{masked[-4:]}" if len(masked) > 8 else "set"
        st.markdown(
            f'<div class="info-box">OpenRouter connected <code>({tip})</code> — chat streams live.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Model & sampling", expanded=not st.session_state.presentation_mode):
        model = st.selectbox("Model", OPENROUTER_MODELS, key="or_model")
        c1, c2, c3 = st.columns(3)
        temperature = c1.slider("Temperature", 0.0, 1.5, 0.7, 0.05, key="or_temp")
        max_tokens = c2.slider("Max tokens", 64, 4096, 800, 64, key="or_tokens")
        use_stream = c3.toggle("Stream reply", value=True, key="or_stream")
        system_prompt = st.text_area(
            "System prompt",
            value=(
                "You are a clear, friendly ML teaching assistant for first-year CSE students. "
                "Use short paragraphs, concrete examples, and light analogies. "
                "Prefer intuition before formulas."
            ),
            height=90,
            key="or_sys",
        )

    st.markdown('<div class="section-label">Quick prompts</div>', unsafe_allow_html=True)
    qcols = st.columns(len(QUICK_PROMPTS))
    for i, (col, prompt) in enumerate(zip(qcols, QUICK_PROMPTS)):
        label = prompt if len(prompt) < 42 else prompt[:39] + "…"
        if col.button(label, key=f"qp_{i}", use_container_width=True):
            st.session_state.or_pending_prompt = prompt

    b1, b2, b3 = st.columns([1, 1, 1])
    explain_viz = b1.button("Explain this visualization", key="or_explain", use_container_width=True)
    clear_chat = b2.button("Clear chat", key="or_clear", use_container_width=True)
    if b3.button("Jump from Logistic Regression context", key="or_from_lr", use_container_width=True):
        ctx = st.session_state.get("openrouter_context") or ""
        if ctx:
            st.session_state.or_pending_prompt = (
                f"Explain this live demo state to a student:\n\n{ctx}"
            )
        else:
            st.session_state.or_pending_prompt = (
                "Explain logistic regression decision boundaries and thresholds simply."
            )

    if clear_chat:
        st.session_state.or_messages = []
        st.rerun()

    if explain_viz:
        ctx = st.session_state.get("openrouter_context") or "the current ML playground demos"
        st.session_state.or_pending_prompt = (
            f"Explain this visualization/demo context to a student in plain language:\n\n{ctx}"
        )

    # Chat transcript
    for msg in st.session_state.or_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.or_pending_prompt
    user_text = st.chat_input("Ask an AI anything…")
    if pending:
        user_text = pending
        st.session_state.or_pending_prompt = None

    if user_text:
        st.session_state.or_messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        history = [{"role": "system", "content": system_prompt}] + list(st.session_state.or_messages)
        # keep last N turns + system to stay laptop-friendly
        if len(history) > 17:
            history = [history[0]] + history[-16:]

        with st.chat_message("assistant"):
            if use_stream and key_present:
                ok, result = openrouter_chat(
                    history, model, temperature, max_tokens, stream=True
                )
                if not ok:
                    st.error(result)
                    st.session_state.or_messages.pop()  # drop failed user turn? keep it
                else:
                    try:
                        answer = st.write_stream(_stream_openrouter(result))
                    except Exception as e:
                        st.error(f"Stream failed: {e}")
                        answer = None
                    if answer:
                        st.session_state.or_messages.append(
                            {"role": "assistant", "content": answer}
                        )
            else:
                with st.spinner("Calling OpenRouter…"):
                    ok, text = openrouter_chat(
                        history, model, temperature, max_tokens, stream=False
                    )
                if ok:
                    st.markdown(text)
                    st.session_state.or_messages.append({"role": "assistant", "content": text})
                else:
                    st.error(text)

    if not st.session_state.or_messages and not st.session_state.presentation_mode:
        st.markdown(
            """
<div class="glass-card">
  <h3>Try this live</h3>
  <p>Pick a quick prompt above, or ask: <em>“Quiz me on confusion matrices.”</em>
  Switch models mid-demo to show how gateways unlock many providers with one key.</p>
</div>
""",
            unsafe_allow_html=True,
        )


# ===========================================================================
# 9–10. CURSOR & AGENTS (combined nav item)
# ===========================================================================
def page_cursor_agents() -> None:
    tab_c, tab_a = st.tabs(["🧰 Cursor", "🤖 AI Agents"])

    with tab_c:
        st.markdown("## Cursor")
        st.markdown(
            """
**An AI-powered code editor** that understands your codebase and helps you write, modify,
debug and explore code.
"""
        )
        st.markdown('<div class="section-label">Workflow</div>', unsafe_allow_html=True)
        flow_horizontal(
            ["IDE", "AI understands code", "Prompt", "Code changes", "Run", "Debug", "Iterate"]
        )

        if not st.session_state.presentation_mode:
            st.markdown("### Core ideas")
            cols = st.columns(3)
            ideas = [
                ("Chat", "Ask questions about files, errors, and design."),
                ("Inline edit", "Select code → instruct → accept/reject diffs."),
                ("Composer / Agent", "Multi-file changes with tool use & iteration."),
                ("Codebase context", "@files, @folders, rules, and retrieval."),
                ("Rules", "Persistent project guidance for the AI."),
                ("Debug & refactor", "Trace bugs; reshape APIs without guesswork."),
            ]
            for i, (t, b) in enumerate(ideas):
                with cols[i % 3]:
                    st.markdown(card_html(t, b), unsafe_allow_html=True)

        st.markdown("### 5 prompts students can try")
        for p in [
            "Explain this code.",
            "Find the bug in this function.",
            "Refactor this into a reusable function.",
            "Add tests.",
            "Implement this feature without changing the existing API.",
        ]:
            st.code(p, language="text")

    with tab_a:
        st.markdown("## AI Agents")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Chatbot**")
            flow_vertical(["User", "Prompt", "Response"])
        with c2:
            st.markdown("**Agent**")
            flow_vertical(["Goal", "Reason", "Choose tool", "Execute", "Observe", "Repeat"])

        st.markdown(
            """
Agents can use **tools**, **APIs**, **files**, **browsers**, **code execution**, and **databases**.
"""
        )
        st.markdown(
            '<div class="warn-box">Agents are not magic. They are systems that combine models '
            "with tools, state and control loops.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Toy agent visualization")
        goal = st.text_input("Goal", value=st.session_state.agent_goal, key="agent_goal_input")
        st.session_state.agent_goal = goal

        steps = ["USER", "AGENT", "TOOL", "RESULT", "AGENT", "ANSWER"]
        # Parse numbers for a real tiny tool
        import re

        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", goal)]
        tool_result = None
        answer = None
        if nums:
            avg = sum(nums) / len(nums)
            tool_result = f"mean({nums}) = {avg:.4f}"
            answer = (
                f"The average of {len(nums)} numbers is **{avg:.4f}**. "
                "I used a calculator tool, then explained the result."
            )
        else:
            tool_result = "(no numbers found — tool returns N/A)"
            answer = "I couldn't find numbers to average. Please include a list of values."

        if st.button("Run toy agent"):
            st.session_state.agent_step = len(steps)

        # Animate-ish reveal
        reveal = st.session_state.agent_step
        if reveal == 0:
            st.caption("Press **Run toy agent** to step through the loop.")
        flow_horizontal(steps)
        detail = [
            ("USER", goal),
            ("AGENT", "Plan: extract numbers → call average tool → explain."),
            ("TOOL", "calculator.mean(...)"),
            ("RESULT", tool_result),
            ("AGENT", "Compose a student-friendly explanation."),
            ("ANSWER", answer),
        ]
        for i, (s, text) in enumerate(detail):
            if reveal == 0 or i < reveal:
                st.markdown(f"**{s}** — {text}")


# ===========================================================================
# 11. HUGGING FACE
# ===========================================================================
def page_hf() -> None:
    st.markdown("## 🤗 Hugging Face")
    st.markdown(
        "Hugging Face is an ecosystem for **sharing, discovering and using** machine learning models and datasets."
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, (t, b) in zip(
        [c1, c2, c3, c4],
        [
            ("Model Hub", "Pretrained checkpoints for NLP, CV, audio, multimodal."),
            ("Datasets", "Ready-to-stream datasets with one-liners."),
            ("Spaces", "Host interactive ML demos in the browser."),
            ("Transformers", "Unified Python API for models & pipelines."),
        ],
    ):
        with col:
            st.markdown(card_html(t, b), unsafe_allow_html=True)

    st.markdown("### Pipeline example")
    st.code(
        """from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("I love machine learning!")""",
        language="python",
    )

    st.markdown('<div class="section-label">Inference flow</div>', unsafe_allow_html=True)
    flow_horizontal(["Model", "Pipeline", "Input", "Inference", "Output"])

    st.markdown("### Where HF shows up")
    st.markdown(
        """
- **NLP** — classification, NER, translation, summarization  
- **Computer vision** — depth, detection, segmentation  
- **Audio** — ASR, TTS  
- **Multimodal** — vision-language models  
- **Embeddings** — sentence / image vectors  
- **Generative** — diffusion & LLMs  

In this playground, HF can power **depth estimation** (Transformers or Inference API) and inspire the **embedding** section’s API mode.
"""
    )


# ===========================================================================
# 12. KAGGLE
# ===========================================================================
def page_kaggle() -> None:
    st.markdown("## 📊 Kaggle")
    st.markdown(
        "Kaggle is a platform for **datasets**, **competitions**, **notebooks**, **model experiments**, and learning."
    )
    st.markdown('<div class="section-label">Workflow</div>', unsafe_allow_html=True)
    flow_horizontal(
        ["Dataset", "Explore", "Clean", "Visualize", "Train", "Evaluate", "Submit"]
    )

    st.markdown("### Typical beginner workflow")
    st.markdown(
        """
1. Find dataset  
2. Read documentation  
3. Load data  
4. EDA  
5. Clean data  
6. Feature engineering  
7. Train baseline  
8. Evaluate  
9. Improve  
10. Submit / share notebook  
"""
    )

    st.markdown("### Tiny pandas example")
    demo = pd.DataFrame(
        {
            "feature_a": np.random.default_rng(0).normal(size=8),
            "feature_b": np.random.default_rng(1).normal(size=8),
            "target": np.random.default_rng(2).integers(0, 2, size=8),
        }
    )
    st.code(
        "import pandas as pd\n"
        "df = pd.read_csv('train.csv')\n"
        "df.head()\n"
        "df.describe()\n"
        "df['target'].value_counts()",
        language="python",
    )
    st.dataframe(demo, use_container_width=True)


# ===========================================================================
# 13. COLAB
# ===========================================================================
def page_colab() -> None:
    st.markdown("## ☁️ Google Colab")
    st.markdown(
        "Colab is a browser-based Jupyter notebook environment that lets you run Python "
        "**without setting up a local environment**."
    )
    st.markdown('<div class="section-label">Flow</div>', unsafe_allow_html=True)
    flow_horizontal(["Browser", "Notebook", "Python", "GPU/CPU", "ML experiment"])

    st.markdown("### What to know")
    bullets = [
        ("Notebooks", "Documents made of code + markdown cells."),
        ("Cells", "Run pieces interactively; iterate fast."),
        ("GPU", "Runtime → Change runtime type → GPU for deep learning."),
        ("pip install", "Use `!pip install ...` inside a cell."),
        ("Google Drive", "Mount Drive to persist datasets/checkpoints."),
        ("Sharing", "Share like a Doc — great for teaching."),
        ("Runtime reset", "Machines are ephemeral; save to Drive."),
    ]
    for t, b in bullets:
        st.markdown(f"- **{t}** — {b}")

    st.code(
        """!pip install transformers
import torch
print(torch.cuda.is_available())""",
        language="python",
    )

    st.markdown("### When to use what")
    st.markdown(
        """
| Tool | Best for |
|---|---|
| **Local laptop** | Small experiments, this Streamlit playground |
| **Colab** | GPU / quick experiments in the browser |
| **Kaggle** | Datasets & competitions |
| **Hugging Face** | Models & datasets sharing |
| **Cursor** | Coding / development with AI |
"""
    )


# ===========================================================================
# Main
# ===========================================================================
PAGE_RENDERERS: dict[str, Callable[[], None]] = {
    "🏠 Home": page_home,
    "📈 Logistic Regression": page_logistic,
    "🧠 Word Embeddings": page_embeddings,
    "👁️ Depth Estimation": page_depth,
    "🎯 YOLO Vision": page_yolo,
    "✨ 3D Gaussian Splatting": page_gaussian,
    "🍄 Mario": page_mario,
    "🤖 OpenRouter AI": page_openrouter,
    "🧰 Cursor & Agents": page_cursor_agents,
    "🤗 Hugging Face": page_hf,
    "📊 Kaggle": page_kaggle,
    "☁️ Google Colab": page_colab,
}


def main() -> None:
    _init_state()

    # Keyboard-ish controls via query / buttons (Streamlit has limited key events)
    render_presentation_nav()
    inject_css(st.session_state.presentation_mode)

    if not st.session_state.presentation_mode:
        with st.sidebar:
            st.markdown("### AI / ML Playground")
            st.caption("From classical ML → CV → GenAI → tools")
            choice = st.radio(
                "Navigate",
                PAGES,
                index=st.session_state.page_idx,
                key="nav_radio",
            )
            st.session_state.page_idx = PAGES.index(choice)
            st.divider()
            if get_openrouter_key():
                st.markdown(
                    '<div class="metric-chip tp">OpenRouter · ready</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="metric-chip fn">OpenRouter · no key</div>',
                    unsafe_allow_html=True,
                )
            st.caption("Keys load from `.env` — never hardcode them.")
            st.divider()
            st.markdown("**Shortcuts**")
            st.caption("Use Previous / Next · Reset clears demo state.")
            st.divider()
            st.markdown(
                f'<div class="metric-chip">page {st.session_state.page_idx+1}/{len(PAGES)}</div>',
                unsafe_allow_html=True,
            )
    else:
        # In presentation mode, page comes from session index only
        pass

    page_name = PAGES[st.session_state.page_idx]
    try:
        PAGE_RENDERERS[page_name]()
    except Exception:
        st.error("Something went wrong rendering this section — the app stayed up.")
        st.code(traceback.format_exc(), language="text")

    st.markdown(
        '<div class="footer-note">Built as an interactive AI/ML learning playground.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
