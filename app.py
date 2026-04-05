"""
Bone Cancer Detection System — Professional Medical AI Dashboard
Radiology Workstation UI built with Streamlit
"""
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os, io, time, json, base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from model import load_checkpoint
from dataset import get_transforms
from gradcam import GradCAM, create_heatmap_overlay
import config
import hdfs_manager
import cassandra_db

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bone Cancer Detection System", page_icon="🦴",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS Theme ────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,.stApp{font-family:'Inter',sans-serif!important;background:#050505!important;color:#EAEAEA!important}
.stApp::before{content:'';position:fixed;top:0;left:0;right:0;bottom:0;
  background:radial-gradient(ellipse 600px 400px at 15% 20%,rgba(255,255,255,.012) 0%,transparent 70%),
  radial-gradient(ellipse 500px 500px at 80% 70%,rgba(255,255,255,.008) 0%,transparent 70%);
  pointer-events:none;z-index:0;animation:bgP 12s ease-in-out infinite alternate}
@keyframes bgP{0%{opacity:.6}100%{opacity:1}}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A0A0A,#080808)!important;border-right:1px solid rgba(255,255,255,.04)!important}
section[data-testid="stSidebar"] .stRadio>label{color:#888!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:2px!important;font-weight:600!important}
section[data-testid="stSidebar"] .stRadio>div>label{background:transparent!important;border:1px solid rgba(255,255,255,.04)!important;border-radius:8px!important;padding:12px 16px!important;margin-bottom:4px!important;transition:all .3s ease!important;color:#A3A3A3!important;font-weight:500!important}
section[data-testid="stSidebar"] .stRadio>div>label:hover{background:rgba(255,255,255,.03)!important;border-color:rgba(255,255,255,.1)!important;color:#EAEAEA!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0A0A0A}::-webkit-scrollbar-thumb{background:#222;border-radius:3px}
h1,h2,h3,h4,h5,h6{font-family:'Inter',sans-serif!important;color:#EAEAEA!important;font-weight:700!important}
.hero-header{text-align:center;padding:30px 20px 10px}
.hero-title{font-size:36px;font-weight:800;color:#EAEAEA;letter-spacing:-.5px;margin:0;text-shadow:0 0 40px rgba(255,255,255,.06)}
.hero-subtitle{font-size:13px;color:#666;letter-spacing:4px;text-transform:uppercase;margin-top:8px}
.hero-divider{width:60px;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.15),transparent);margin:16px auto 0}
.glass-card{background:#0E0E0E;border:1px solid rgba(255,255,255,.04);border-radius:12px;padding:24px;margin-bottom:16px;transition:all .4s ease;position:relative;overflow:hidden}
.glass-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent)}
.glass-card:hover{border-color:rgba(255,255,255,.08);box-shadow:0 4px 30px rgba(0,0,0,.4);transform:translateY(-1px)}
.metric-card{background:#0E0E0E;border:1px solid rgba(255,255,255,.04);border-radius:12px;padding:20px;text-align:center;transition:all .3s ease}
.metric-card:hover{border-color:rgba(255,255,255,.08);box-shadow:0 0 30px rgba(255,255,255,.02)}
.metric-value{font-size:32px;font-weight:800;color:#EAEAEA;line-height:1;margin-bottom:6px}
.metric-label{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:2px;font-weight:600}
.result-badge{display:inline-block;padding:8px 20px;border-radius:20px;font-size:14px;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.badge-normal{background:rgba(76,175,80,.12);color:#66BB6A;border:1px solid rgba(76,175,80,.2)}
.badge-benign{background:rgba(255,193,7,.12);color:#FFD54F;border:1px solid rgba(255,193,7,.2)}
.badge-malignant{background:rgba(244,67,54,.12);color:#EF5350;border:1px solid rgba(244,67,54,.2)}
.interpretation-box{background:#0A0A0A;border:1px solid rgba(255,255,255,.04);border-left:3px solid;border-radius:8px;padding:16px 20px;margin:12px 0;font-size:13px;line-height:1.8;color:#B0B0B0}
.interp-normal{border-left-color:#4CAF50}.interp-benign{border-left-color:#FFC107}.interp-malignant{border-left-color:#F44336}
.confidence-bar-bg{background:#1A1A1A;border-radius:6px;height:8px;overflow:hidden}
.confidence-bar-fill{height:100%;border-radius:6px;transition:width 1.5s ease}
.conf-fill-normal{background:linear-gradient(90deg,#2E7D32,#4CAF50)}.conf-fill-benign{background:linear-gradient(90deg,#F57F17,#FFC107)}.conf-fill-malignant{background:linear-gradient(90deg,#C62828,#F44336)}
.section-header{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:3px;font-weight:600;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,.04)}
.info-table{width:100%;border-collapse:separate;border-spacing:0}
.info-table tr:hover{background:rgba(255,255,255,.02)}.info-table td{padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.03);font-size:13px}
.info-table td:first-child{color:#666;font-weight:500;width:40%;text-transform:uppercase;letter-spacing:1px;font-size:11px}
.info-table td:last-child{color:#EAEAEA;font-weight:600}
.scan-ring{width:50px;height:50px;border:2px solid rgba(255,255,255,.05);border-top:2px solid rgba(255,255,255,.3);border-radius:50%;margin:0 auto 16px;animation:scanSpin 1.2s linear infinite}
@keyframes scanSpin{to{transform:rotate(360deg)}}
.scan-text{font-size:12px;color:#555;letter-spacing:3px;text-transform:uppercase;animation:scanPulse 1.5s ease-in-out infinite}
@keyframes scanPulse{0%,100%{opacity:.4}50%{opacity:1}}
.status-dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:8px;animation:dotPulse 2s ease-in-out infinite}
.dot-active{background:#4CAF50;box-shadow:0 0 8px rgba(76,175,80,.4)}
@keyframes dotPulse{0%,100%{opacity:1}50%{opacity:.4}}
.history-item{background:#0A0A0A;border:1px solid rgba(255,255,255,.03);border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;font-size:12px}
.stButton>button{background:linear-gradient(135deg,#1A1A1A,#0E0E0E)!important;color:#EAEAEA!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:8px!important;padding:10px 28px!important;font-weight:600!important;font-family:'Inter',sans-serif!important;letter-spacing:.5px!important;transition:all .3s ease!important;font-size:13px!important}
.stButton>button:hover{background:linear-gradient(135deg,#222,#1A1A1A)!important;border-color:rgba(255,255,255,.15)!important;box-shadow:0 4px 20px rgba(0,0,0,.4)!important;transform:translateY(-1px)!important}
.stFileUploader>div{background:#0E0E0E!important;border:1px dashed rgba(255,255,255,.08)!important;border-radius:12px!important}
.stTabs [data-baseweb="tab-list"]{gap:0;background:#0A0A0A;border-radius:8px;padding:4px;border:1px solid rgba(255,255,255,.03)}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#666!important;border-radius:6px!important;font-size:12px!important;font-weight:600!important;letter-spacing:.5px!important;padding:8px 16px!important}
.stTabs [aria-selected="true"]{background:rgba(255,255,255,.05)!important;color:#EAEAEA!important}
.stTabs [data-baseweb="tab-highlight"]{background:transparent!important}
.stTabs [data-baseweb="tab-border"]{display:none!important}
.stDownloadButton>button{background:#0E0E0E!important;color:#A3A3A3!important;border:1px solid rgba(255,255,255,.06)!important;border-radius:8px!important;font-size:12px!important}
.footer-bar{text-align:center;padding:20px;margin-top:40px;border-top:1px solid rgba(255,255,255,.03);font-size:11px;color:#333;letter-spacing:2px;text-transform:uppercase}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
</style>""", unsafe_allow_html=True)

# ── Plotly dark theme helper ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,14,14,1)',
    font=dict(family='Inter', color='#A3A3A3', size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.05)'),
)
CLASS_COLORS = {'normal': '#4CAF50', 'benign': '#FFC107', 'malignant': '#F44336'}

# ── Dataset class availability check (must be before sidebar) ────────────────
@st.cache_data
def get_class_availability():
    """Return dict of class → image count across train/val/test splits."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    availability = {}
    for cls in config.CLASS_NAMES:
        total = 0
        for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                total += sum(1 for f in os.listdir(cls_dir)
                             if os.path.splitext(f)[1].lower() in EXTS)
        availability[cls] = total
    return availability

CLASS_AVAILABILITY = get_class_availability()
MISSING_CLASSES    = [c for c, n in CLASS_AVAILABILITY.items() if n == 0]
ACTIVE_CLASSES     = [c for c, n in CLASS_AVAILABILITY.items() if n > 0]

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero-header"><div class="hero-title">Bone Cancer Detection System</div>
<div class="hero-subtitle">AI-Based X-Ray Analysis</div><div class="hero-divider"></div></div>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="padding:16px 0 20px;text-align:center"><div style="font-size:28px;margin-bottom:4px">🦴</div>
    <div style="font-size:11px;color:#555;letter-spacing:3px;text-transform:uppercase;font-weight:600">Radiology AI</div></div>""", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", [
        "🔬  Detection Dashboard",
        "📊  Batch Analysis",
        "📈  Model Performance",
        "📋  Prediction History",
        "🧠  Model Information",
    ])
    st.markdown("---")

    # ── System status ──────────────────────────────────────────────────────
    cass_ok = cassandra_db.is_connected()
    cass_dot   = "dot-active" if cass_ok  else "dot-error"
    cass_label = "CASSANDRA ONLINE" if cass_ok else "CASSANDRA OFFLINE"
    cass_color = "#4CAF50" if cass_ok else "#F44336"

    hdfs_stats = hdfs_manager.get_hdfs_stats()
    total_files = sum(v["count"] for v in hdfs_stats.values())

    st.markdown(f"""<div style="padding:12px 0">
    <div style="display:flex;align-items:center;margin-bottom:6px">
      <div class="status-dot dot-active"></div>
      <span style="font-size:10px;color:#555;letter-spacing:1px">MODEL ONLINE</span>
    </div>
    <div style="display:flex;align-items:center;margin-bottom:10px">
      <div class="status-dot" style="background:{cass_color};box-shadow:0 0 8px {cass_color}44;display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:8px;"></div>
      <span style="font-size:10px;color:#555;letter-spacing:1px">{cass_label}</span>
    </div>
    <div style="font-size:10px;color:#333;letter-spacing:1px;text-transform:uppercase;line-height:1.9">
      Model: ResNet50<br>Active Classes: {len(ACTIVE_CLASSES)}/3<br>Input: 224×224
    </div></div>""", unsafe_allow_html=True)

    # Missing class warning in sidebar
    if MISSING_CLASSES:
        missing_str = " · ".join(c.capitalize() for c in MISSING_CLASSES)
        st.markdown(f"""<div style="background:rgba(255,193,7,.06);border:1px solid rgba(255,193,7,.2);
        border-left:3px solid #FFC107;border-radius:6px;padding:10px 12px;margin-top:8px">
        <div style="font-size:10px;color:#FFC107;font-weight:700;letter-spacing:1px;margin-bottom:4px">⚠ MISSING DATA</div>
        <div style="font-size:10px;color:#888;line-height:1.6">No training images for:<br>
        <span style="color:#FFD54F;font-weight:600">{missing_str}</span><br>
        Model is effectively {len(ACTIVE_CLASSES)}-class only.</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── HDFS stats panel ───────────────────────────────────────────────────
    st.markdown("""<div style="font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:10px">HDFS Storage</div>""", unsafe_allow_html=True)
    hdfs_rows = [
        ("Raw Images",    hdfs_stats["raw"]["count"],         "📁"),
        ("Processed",     hdfs_stats["processed"]["count"],   "🖼"),
        ("Predictions",   hdfs_stats["predictions"]["count"], "📄"),
        ("Logs",          hdfs_stats["logs"]["count"],        "📝"),
    ]
    for label, count, icon in hdfs_rows:
        st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
        padding:5px 0;border-bottom:1px solid rgba(255,255,255,.03)">
        <span style="font-size:11px;color:#666">{icon} {label}</span>
        <span style="font-size:11px;color:#EAEAEA;font-weight:600">{count}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:10px;color:#444;margin-top:6px;text-align:right">
    Total: {total_files} files</div>""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    ckpts = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith("best_") and f.endswith(".pth")]
    if not ckpts:
        return None, None
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, sorted(ckpts)[-1])
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    # Detect how many output classes the checkpoint was trained with
    sd         = ckpt.get("model_state", {})
    last_w_key = [k for k in sd if "classifier" in k and k.endswith(".weight")][-1]
    ckpt_classes = sd[last_w_key].shape[0]   # e.g. 3 for old checkpoint, 2 for new

    model = load_checkpoint(ckpt_path, device="cpu")
    model.eval()
    return model, ckpt_classes

model, CKPT_NUM_CLASSES = load_model()
if model is None:
    CKPT_NUM_CLASSES = config.NUM_CLASSES
# True when the saved checkpoint matches the current config
CKPT_MATCHES_CONFIG = (CKPT_NUM_CLASSES == config.NUM_CLASSES)

# Class names that match what the LOADED CHECKPOINT was actually trained with.
# The old checkpoint has 3 classes; new retrained model will have 2.
_ALL_CLASS_NAMES = ["normal", "benign", "malignant"]
MODEL_CLASS_NAMES = (
    config.CLASS_NAMES if CKPT_MATCHES_CONFIG
    else _ALL_CLASS_NAMES[:CKPT_NUM_CLASSES]
    if CKPT_NUM_CLASSES <= len(_ALL_CLASS_NAMES)
    else config.CLASS_NAMES
)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("prediction_history", []),
    ("img_bytes", None),       # raw bytes of last uploaded image
    ("img_name", None),
    ("last_result", None),     # dict with prediction, confidence, probs, heatmap_bytes
    ("batch_results", None),   # list of dicts from batch run
    ("eval_metrics", None),    # dict from test-set eval
    ("hdfs_inited", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Initialise HDFS on first load
if not st.session_state.hdfs_inited:
    hdfs_manager.init_hdfs()
    st.session_state.hdfs_inited = True

# ── Helpers ──────────────────────────────────────────────────────────────────
def run_prediction(image):
    transform = get_transforms("test")
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(img_t), dim=1)[0]
    return MODEL_CLASS_NAMES[probs.argmax().item()], probs

def generate_heatmap(image):
    """Generate Grad-CAM heatmap overlay."""
    transform = get_transforms("test")
    img_t = transform(image).unsqueeze(0)
    img_t.requires_grad = True
    cam_gen = GradCAM(model)
    cam = cam_gen.generate(img_t)
    return create_heatmap_overlay(image, cam, alpha=0.45)

def get_interpretation(prediction, confidence):
    interps = {
        "normal": ("No signs of bone tumor detected.",
            "Bone density and structural integrity appear within normal parameters. "
            "No irregular mass formation or abnormal calcification patterns identified. "
            "Standard follow-up schedule recommended."),
        "benign": ("Possible benign bone tumor detected.",
            "Localized bone density variation identified with characteristics consistent with benign neoplasm. "
            "Well-defined margins and regular structure observed. "
            "Clinical correlation and periodic monitoring recommended. Consider follow-up imaging in 3–6 months."),
        "malignant": ("High probability of malignant bone tumor detected.",
            "Abnormal bone density and irregular structure identified. "
            "Findings suggest aggressive growth pattern with poorly defined margins. "
            "Immediate clinical review strongly recommended. Biopsy and advanced imaging (MRI/CT) advised."),
    }
    summary, detail = interps.get(prediction, interps["normal"])
    if confidence < 60:
        detail += " Note: Confidence below diagnostic threshold — manual radiologist review essential."
    return summary, detail

def make_probability_chart(probs, missing_classes=None):
    """Bar chart for class probabilities. Greys out missing/untrained classes."""
    missing_classes = missing_classes or []
    names, vals, colors, texts = [], [], [], []
    for i, cls in enumerate(MODEL_CLASS_NAMES):
        val = float(probs[i]) * 100
        names.append(cls.capitalize())
        vals.append(val)
        if cls in missing_classes:
            colors.append("rgba(80,80,80,0.3)")
            texts.append(f"N/A")
        else:
            colors.append(CLASS_COLORS[cls])
            texts.append(f"{val:.1f}%")

    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, text=texts,
                           textposition='outside', textfont=dict(color='#EAEAEA', size=13)))

    # Annotation for each missing class
    annotations = []
    for i, cls in enumerate(MODEL_CLASS_NAMES):
        if cls in missing_classes:
            annotations.append(dict(x=cls.capitalize(), y=max(vals)*0.5,
                text="No Data", showarrow=False,
                font=dict(color="rgba(255,193,7,0.6)", size=10), textangle=-90))

    fig.update_layout(**PLOTLY_LAYOUT, title=None, yaxis_title='Probability (%)',
                      yaxis_range=[0, 115], height=300, showlegend=False,
                      bargap=0.4, annotations=annotations)
    return fig

def make_confidence_breakdown(probs, missing_classes=None):
    """
    Three individual bullet-gauge indicators — one per class.
    Completely distinct from the bar chart.
    """
    from plotly.subplots import make_subplots
    missing_classes = missing_classes or []

    all_cls = ["normal", "benign", "malignant"]
    # Build values for all 3 display slots; model may only have 2 outputs
    vals, colors, labels = [], [], []
    for cls in all_cls:
        if cls in MODEL_CLASS_NAMES:
            idx = MODEL_CLASS_NAMES.index(cls)
            val = round(float(probs[idx]) * 100, 1)
        else:
            val = 0.0
        is_missing = cls in missing_classes or cls not in MODEL_CLASS_NAMES
        vals.append(val)
        colors.append("rgba(60,60,60,0.5)" if is_missing else CLASS_COLORS.get(cls, "#EAEAEA"))
        labels.append("N/A" if is_missing else f"{val:.1f}%")

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=[c.capitalize() for c in all_cls],
    )

    for col_idx, (cls, val, color, label, is_missing) in enumerate(
        zip(all_cls, vals, colors, labels,
            [c in missing_classes or c not in MODEL_CLASS_NAMES for c in all_cls])
    ):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val if not is_missing else 0,
                number=dict(
                    suffix="%" if not is_missing else "",
                    valueformat=".1f",
                    font=dict(
                        color="rgba(100,100,100,0.6)" if is_missing else color,
                        size=22,
                    ),
                ),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=0,
                              tickcolor="rgba(0,0,0,0)",
                              tickfont=dict(color="#333", size=9)),
                    bar=dict(color=color, thickness=0.7),
                    bgcolor="rgba(20,20,20,1)",
                    borderwidth=1,
                    bordercolor="rgba(255,255,255,0.05)",
                    steps=[
                        dict(range=[0, 33],  color="rgba(244,67,54,0.05)"),
                        dict(range=[33, 66], color="rgba(255,193,7,0.05)"),
                        dict(range=[66, 100],color="rgba(76,175,80,0.05)"),
                    ],
                    threshold=dict(
                        line=dict(color=color, width=2),
                        thickness=0.8,
                        value=val if not is_missing else 0,
                    ),
                ),
                title=dict(
                    text="No Data" if is_missing else "",
                    font=dict(color="#FFC107", size=11),
                ),
            ),
            row=1, col=col_idx + 1,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,14,14,1)",
        font=dict(family="Inter", color="#A3A3A3", size=12),
        height=240,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(color="#666", size=11, family="Inter")

    return fig

def make_gauge_chart(confidence, prediction):
    """Create Plotly gauge for confidence."""
    color = CLASS_COLORS.get(prediction, '#EAEAEA')
    fig = go.Figure(go.Indicator(mode="gauge+number", value=confidence,
        number=dict(suffix="%", font=dict(color='#EAEAEA', size=36)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor='#555', tickfont=dict(color='#555')),
                   bar=dict(color=color), bgcolor='#1A1A1A',
                   bordercolor='rgba(255,255,255,0.04)',
                   steps=[dict(range=[0, 40], color='rgba(244,67,54,0.08)'),
                          dict(range=[40, 70], color='rgba(255,193,7,0.08)'),
                          dict(range=[70, 100], color='rgba(76,175,80,0.08)')])))
    layout = {**PLOTLY_LAYOUT, 'height': 220, 'margin': dict(l=30, r=30, t=30, b=10)}
    fig.update_layout(**layout)
    return fig

# ── Retrain notice (shown on every page if checkpoint is stale) ───────────────
if not CKPT_MATCHES_CONFIG:
    st.markdown(f"""<div style="background:rgba(66,165,245,.07);border:1px solid rgba(66,165,245,.2);
    border-left:4px solid #42A5F5;border-radius:8px;padding:12px 18px;margin-bottom:16px;font-size:12px;color:#888;line-height:1.8">
    <span style="color:#42A5F5;font-weight:700">🔄 Retrain Required</span> — The saved checkpoint has
    <span style="color:#EAEAEA;font-weight:600">{CKPT_NUM_CLASSES} output classes</span> but the config now defines
    <span style="color:#EAEAEA;font-weight:600">{config.NUM_CLASSES} classes ({', '.join(config.CLASS_NAMES)})</span>.
    Predictions are currently running on the old checkpoint.
    Run <code style="background:#1A1A1A;padding:1px 8px;border-radius:3px;color:#CE93D8">python main.py --mode train --device cpu</code>
    to train the improved 2-class model (Adam · OneCycleLR · label smoothing · stronger augmentation).
    Expected accuracy: <span style="color:#4CAF50;font-weight:600">~85–92%</span>.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Detection Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔬  Detection Dashboard":
    col_up, col_res = st.columns([1, 1], gap="large")
    with col_up:
        st.markdown('<div class="section-header">X-Ray Image Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png","bmp","tiff"], label_visibility="collapsed")

        # Persist new upload into session state
        if uploaded_file is not None:
            new_bytes = uploaded_file.read()
            if new_bytes != st.session_state.img_bytes:
                # New image uploaded — clear previous results
                st.session_state.img_bytes = new_bytes
                st.session_state.img_name  = uploaded_file.name
                st.session_state.last_result = None

        # Use persisted image if available
        if st.session_state.img_bytes:
            image = Image.open(io.BytesIO(st.session_state.img_bytes)).convert("RGB")
            st.image(image, caption=st.session_state.img_name or "Uploaded X-Ray", use_container_width=True)
            btn_label = "⬡  Re-analyze" if st.session_state.last_result else "⬡  Analyze X-Ray"
            predict_btn = st.button(btn_label, use_container_width=True)
            if st.session_state.last_result:
                if st.button("✕  Clear Image", use_container_width=True):
                    st.session_state.img_bytes   = None
                    st.session_state.img_name    = None
                    st.session_state.last_result = None
                    st.rerun()
        else:
            st.markdown("""<div class="glass-card" style="text-align:center;padding:50px 20px">
            <div style="font-size:40px;margin-bottom:12px;opacity:.15">📂</div>
            <div style="font-size:13px;color:#555;font-weight:500">Upload a bone X-ray image to begin analysis</div>
            <div style="font-size:11px;color:#333;margin-top:8px">Supported: JPG · PNG · BMP · TIFF</div></div>""", unsafe_allow_html=True)
            predict_btn = False
            image = None

    with col_res:
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

        # Run analysis when button pressed
        if image is not None and predict_btn:
            if model is None:
                st.error("Model checkpoint not found. Train the model first.")
            else:
                ph = st.empty()
                ph.markdown("""<div style="text-align:center;padding:30px"><div class="scan-ring"></div>
                <div class="scan-text">Analyzing bone structure</div></div>""", unsafe_allow_html=True)
                time.sleep(1.2)
                prediction, probs = run_prediction(image)
                confidence = probs.max().item() * 100

                # Grad-CAM
                heatmap_bytes = None
                try:
                    heatmap_img = generate_heatmap(image)
                    buf = io.BytesIO()
                    heatmap_img.save(buf, format="PNG")
                    heatmap_bytes = buf.getvalue()
                except Exception:
                    pass

                ph.empty()
                st.session_state.last_result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probs": probs,
                    "heatmap_bytes": heatmap_bytes,
                }
                st.session_state.prediction_history.insert(0, {
                    "file": st.session_state.img_name, "prediction": prediction,
                    "confidence": f"{confidence:.1f}%"
                })
                st.session_state.prediction_history = st.session_state.prediction_history[:10]

                # ── Big Data integration ──────────────────────────────────
                from datetime import datetime
                ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                probs_dict = {MODEL_CLASS_NAMES[i]: round(float(probs[i])*100, 2)
                              for i in range(len(MODEL_CLASS_NAMES))}
                img_name = st.session_state.img_name or "unknown.jpg"

                # 1. Save raw image to HDFS
                hdfs_manager.save_raw_image(st.session_state.img_bytes, img_name)
                # 2. Save processed (224×224) image to HDFS
                hdfs_manager.save_processed_image(image, img_name)
                # 3. Save prediction JSON to HDFS
                hdfs_manager.save_prediction_json(img_name, prediction, confidence, probs_dict, ts)
                # 4. Insert into Cassandra
                cassandra_db.insert_prediction(img_name, prediction, confidence, probs_dict, ts)

        # Display persisted results (survives page switches)
        res = st.session_state.last_result
        if res:
            prediction = res["prediction"]
            confidence = res["confidence"]
            probs      = res["probs"]

            # Missing-class notice
            if MISSING_CLASSES:
                missing_str = " · ".join(c.capitalize() for c in MISSING_CLASSES)
                active_str  = " · ".join(c.capitalize() for c in ACTIVE_CLASSES)
                st.markdown(f"""<div style="background:rgba(255,193,7,.05);border:1px solid rgba(255,193,7,.15);
                border-left:3px solid #FFC107;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:12px;color:#888;line-height:1.7">
                <span style="color:#FFC107;font-weight:700">⚠ Limited Dataset</span> — No training data for
                <span style="color:#FFD54F;font-weight:600">{missing_str}</span>.
                Predictions are restricted to: <span style="color:#EAEAEA;font-weight:600">{active_str}</span>.
                Add benign X-ray images to <code style="background:#1A1A1A;padding:1px 6px;border-radius:3px">data/train/benign/</code> and retrain to unlock full 3-class detection.
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="glass-card" style="text-align:center">
            <div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Diagnosis Result</div>
            <span class="result-badge badge-{prediction}">{prediction}</span>
            <div style="font-size:10px;color:#555;margin-top:8px">{len(ACTIVE_CLASSES)}-class active model</div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(make_gauge_chart(confidence, prediction), use_container_width=True, config={'displayModeBar': False})

            summary, detail = get_interpretation(prediction, confidence)
            st.markdown(f"""<div class="glass-card">
            <div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Clinical Interpretation</div>
            <div style="font-size:14px;font-weight:600;color:#EAEAEA;margin-bottom:8px">{summary}</div>
            <div class="interpretation-box interp-{prediction}">{detail}</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Diagnostic Visualizations</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["🔥 Grad-CAM Heatmap", "📊 Probability Chart", "📋 Confidence Breakdown"])
            with t1:
                if res["heatmap_bytes"]:
                    hcol1, hcol2 = st.columns(2)
                    with hcol1:
                        st.image(image.resize((224, 224)), caption="Original X-Ray", use_container_width=True)
                    with hcol2:
                        st.image(Image.open(io.BytesIO(res["heatmap_bytes"])), caption="Grad-CAM Activation Map", use_container_width=True)
                    st.markdown("""<div style="font-size:11px;color:#555;text-align:center;margin-top:8px">
                    Red/yellow regions indicate areas the model focused on for its prediction</div>""", unsafe_allow_html=True)
                else:
                    st.info("Heatmap unavailable for this checkpoint.")
            with t2:
                st.plotly_chart(make_probability_chart(probs, MISSING_CLASSES), use_container_width=True, config={'displayModeBar': False})
            with t3:
                st.plotly_chart(make_confidence_breakdown(probs, MISSING_CLASSES), use_container_width=True, config={'displayModeBar': False})
        elif image is None or not predict_btn:
            if not res:
                st.markdown("""<div class="glass-card" style="text-align:center;padding:60px 20px">
                <div style="font-size:40px;margin-bottom:12px;opacity:.1">🔬</div>
                <div style="font-size:13px;color:#444;font-weight:500">Awaiting X-ray scan input</div>
                <div style="font-size:11px;color:#333;margin-top:8px">Upload an image and click Analyze</div></div>""", unsafe_allow_html=True)

    # History
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown('<div class="section-header">Recent Predictions</div>', unsafe_allow_html=True)
        h_html = ""
        for item in st.session_state.prediction_history:
            h_html += f"""<div class="history-item"><span style="color:#A3A3A3">{item['file']}</span>
            <span><span class="result-badge badge-{item['prediction']}" style="font-size:10px;padding:4px 10px">{item['prediction']}</span>
            <span style="color:#666;margin-left:10px;font-size:11px">{item['confidence']}</span></span></div>"""
        st.markdown(f'<div class="glass-card">{h_html}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Batch Analysis":
    st.markdown('<div class="section-header">Batch X-Ray Analysis</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload multiple X-ray images", type=["jpg","jpeg","png","bmp","tiff"],
                                     accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        st.markdown(f"""<div class="glass-card" style="text-align:center"><div class="metric-value">{len(uploaded_files)}</div>
        <div class="metric-label">Images Loaded</div></div>""", unsafe_allow_html=True)

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            run_batch = st.button("⬡  Analyze All Images", use_container_width=True)
        with bcol2:
            if st.button("✕  Clear Results", use_container_width=True):
                st.session_state.batch_results = None
                st.rerun()

        if run_batch:
            if model is None:
                st.error("Model checkpoint not found.")
            else:
                results = []
                prog   = st.progress(0)
                status = st.empty()
                for idx, f in enumerate(uploaded_files):
                    status.markdown(f"""<div style="text-align:center;font-size:12px;color:#555;letter-spacing:2px;text-transform:uppercase">
                    Scanning {idx+1}/{len(uploaded_files)} — {f.name}</div>""", unsafe_allow_html=True)
                    img = Image.open(f).convert("RGB")
                    pred, probs = run_prediction(img)
                    conf = probs.max().item() * 100
                    results.append({"Image": f.name, "Prediction": pred.upper(),
                                    "Confidence (%)": round(conf, 2),
                                    "Normal (%)": round(float(probs[0])*100, 2),
                                    "Benign (%)": round(float(probs[1])*100, 2),
                                    "Malignant (%)": round(float(probs[2])*100, 2)})
                    prog.progress((idx+1) / len(uploaded_files))
                status.empty(); prog.empty()
                st.session_state.batch_results = results

    # Display persisted batch results
    results = st.session_state.batch_results
    if results:
        n_n = sum(1 for r in results if r["Prediction"] == "NORMAL")
        n_b = sum(1 for r in results if r["Prediction"] == "BENIGN")
        n_m = sum(1 for r in results if r["Prediction"] == "MALIGNANT")

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl, clr in [(c1, len(results), "Total Scans", "#EAEAEA"),
                                   (c2, n_n, "Normal", "#66BB6A"),
                                   (c3, n_b, "Benign", "#FFD54F"),
                                   (c4, n_m, "Malignant", "#EF5350")]:
            with col:
                st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{clr}">{val}</div>
                <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(labels=["Normal", "Benign", "Malignant"], values=[n_n, n_b, n_m],
            marker=dict(colors=['#4CAF50', '#FFC107', '#F44336']), hole=0.5,
            textfont=dict(color='#EAEAEA', size=13)))
        layout_pie = {**PLOTLY_LAYOUT, 'height': 300, 'title': 'Batch Distribution',
                      'title_font': dict(size=13, color='#555')}
        fig_pie.update_layout(**layout_pie)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇  Download Report (CSV)", data=df.to_csv(index=False),
                           file_name="batch_analysis_report.csv", mime="text/csv", use_container_width=True)
    elif not uploaded_files:
        st.markdown("""<div class="glass-card" style="text-align:center;padding:60px 20px">
        <div style="font-size:40px;margin-bottom:12px;opacity:.12">📊</div>
        <div style="font-size:13px;color:#444;font-weight:500">Upload multiple X-ray images for batch analysis</div></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown('<div class="section-header">Model Performance Dashboard</div>', unsafe_allow_html=True)

    # Dataset availability banner
    if MISSING_CLASSES:
        col_avail = st.columns(len(config.CLASS_NAMES))
        for i, cls in enumerate(MODEL_CLASS_NAMES):
            count = CLASS_AVAILABILITY[cls]
            active = count > 0
            clr  = CLASS_COLORS.get(cls, '#EAEAEA') if active else 'rgba(80,80,80,0.4)'
            icon = "✓" if active else "✗"
            label = f"{count} images" if active else "No data"
            with col_avail[i]:
                st.markdown(f"""<div class="metric-card" style="border-color:{'rgba(255,255,255,.04)' if active else 'rgba(255,193,7,.15)'}">
                <div style="font-size:20px;font-weight:800;color:{clr}">{icon}</div>
                <div style="font-size:13px;color:{'#EAEAEA' if active else '#FFC107'};font-weight:600;margin:4px 0">{cls.capitalize()}</div>
                <div style="font-size:10px;color:#555">{label}</div>
                </div>""", unsafe_allow_html=True)
        if MISSING_CLASSES:
            st.markdown(f"""<div style="background:rgba(255,193,7,.05);border:1px solid rgba(255,193,7,.15);
            border-radius:8px;padding:10px 16px;font-size:12px;color:#888;margin:8px 0 4px">
            <span style="color:#FFC107;font-weight:700">⚠ Evaluation note:</span>
            Benign class has 0 images — confusion matrix row for benign will be all zeros.
            Metrics reflect <span style="color:#EAEAEA;font-weight:600">{len(ACTIVE_CLASSES)}-class performance only</span>.
            </div>""", unsafe_allow_html=True)
    st.markdown("")

    # ── Run Evaluation on Test Set ─────────────────────────────────────────────
    def _run_eval_batched(prog_bar, status_txt):
        """
        Batched evaluation using DataLoader — much faster than single-image loop.
        num_workers=0 avoids Windows multiprocessing issues inside Streamlit.
        """
        from torch.utils.data import DataLoader
        from dataset import BoneCancerDataset
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, confusion_matrix, roc_auc_score)

        dataset = BoneCancerDataset(root=config.TEST_DIR, split="test")
        loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                             num_workers=0, pin_memory=False)

        total   = len(dataset)
        done    = 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in loader:
                logits = model(images)
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())

                done += len(labels)
                prog_bar.progress(done / total)
                status_txt.markdown(
                    f'<div style="text-align:center;font-size:12px;color:#555;'
                    f'letter-spacing:2px;text-transform:uppercase">'
                    f'Processing {done} / {total} images</div>',
                    unsafe_allow_html=True)

        if not all_preds:
            return None

        cm = confusion_matrix(all_labels, all_preds).tolist()
        metrics = {
            "accuracy":         accuracy_score(all_labels, all_preds),
            "f1":               f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "precision":        precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall":           recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "confusion_matrix": cm,
            "total_images":     len(all_preds),
        }
        try:
            metrics["auc_roc"] = float(roc_auc_score(
                all_labels, np.array(all_probs), multi_class="ovr", average="weighted"))
        except Exception:
            metrics["auc_roc"] = None
        return metrics

    ec1, ec2 = st.columns([2, 1])
    with ec1:
        run_eval_btn = st.button("⬡  Evaluate on Test Set", use_container_width=True)
    with ec2:
        if st.button("✕  Clear Results", use_container_width=True):
            st.session_state.eval_metrics = None
            st.rerun()

    if run_eval_btn:
        if model is None:
            st.error("No model checkpoint found.")
        elif not os.path.isdir(config.TEST_DIR):
            st.error(f"Test directory not found: {config.TEST_DIR}")
        else:
            prog_bar   = st.progress(0)
            status_txt = st.empty()
            try:
                result = _run_eval_batched(prog_bar, status_txt)
                prog_bar.empty()
                status_txt.empty()
                if result is None:
                    st.error("No test images found. Check data/test/ directory structure.")
                else:
                    st.session_state.eval_metrics = result
                    st.rerun()
            except Exception as e:
                prog_bar.empty()
                status_txt.empty()
                st.error(f"Evaluation error: {e}")

    # Load from file as fallback if no session metrics
    metrics = st.session_state.eval_metrics
    if metrics is None:
        metrics_path = os.path.join(config.RESULTS_DIR, "test_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

    if metrics:
        total = metrics.get("total_images", "—")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, lbl, val in [
            (c1, "Images Evaluated", str(total)),
            (c2, "Accuracy",  f"{metrics.get('accuracy', 0)*100:.1f}%"),
            (c3, "F1 Score",  f"{metrics.get('f1', 0):.4f}"),
            (c4, "Precision", f"{metrics.get('precision', 0):.4f}"),
            (c5, "Recall",    f"{metrics.get('recall', 0):.4f}"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

        if metrics.get("auc_roc"):
            st.markdown(f"""<div class="glass-card" style="text-align:center;margin-top:12px">
            <div class="metric-value">{metrics['auc_roc']:.4f}</div>
            <div class="metric-label">AUC-ROC (Weighted OvR)</div></div>""", unsafe_allow_html=True)

        st.markdown("")
        cm = metrics.get("confusion_matrix")
        if cm:
            names = [c.capitalize() for c in config.CLASS_NAMES]
            cm_arr = np.array(cm)

            # Confusion matrix
            fig_cm = go.Figure(go.Heatmap(z=cm_arr, x=names, y=names,
                colorscale=[[0,'#0E0E0E'],[0.5,'#1565C0'],[1,'#42A5F5']],
                text=cm_arr, texttemplate="%{text}", textfont=dict(color='#EAEAEA', size=16),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
            layout_cm = {**PLOTLY_LAYOUT, 'height': 400, 'title': 'Confusion Matrix',
                         'title_font': dict(size=13, color='#555'),
                         'xaxis_title': 'Predicted Label', 'yaxis_title': 'True Label',
                         'yaxis': dict(autorange='reversed', gridcolor='rgba(255,255,255,0.03)')}
            fig_cm.update_layout(**layout_cm)
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

            # Per-class metrics bar chart
            per_class = []
            for i, cls in enumerate(MODEL_CLASS_NAMES):
                tp = cm_arr[i, i]
                fn = cm_arr[i, :].sum() - tp
                fp = cm_arr[:, i].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1v  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                per_class.append({"Class": cls.capitalize(),
                                  "Precision": round(prec*100, 1),
                                  "Recall":    round(rec*100,  1),
                                  "F1":        round(f1v*100,  1)})
            df_pc = pd.DataFrame(per_class)
            fig_pc = go.Figure()
            for metric_name, color in [("Precision","#42A5F5"),("Recall","#FFC107"),("F1","#4CAF50")]:
                fig_pc.add_trace(go.Bar(name=metric_name, x=df_pc["Class"], y=df_pc[metric_name],
                    marker_color=color, text=[f"{v}%" for v in df_pc[metric_name]],
                    textposition='outside', textfont=dict(color='#EAEAEA', size=11)))
            fig_pc.update_layout(**PLOTLY_LAYOUT, barmode='group', height=350,
                title="Per-Class Metrics", title_font=dict(size=13, color='#555'),
                yaxis_title="Score (%)", yaxis_range=[0, 115],
                legend=dict(font=dict(color='#A3A3A3')))
            st.plotly_chart(fig_pc, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("""<div class="glass-card" style="text-align:center;padding:50px 20px">
        <div style="font-size:40px;margin-bottom:12px;opacity:.12">📈</div>
        <div style="font-size:13px;color:#444;font-weight:500">No evaluation data yet</div>
        <div style="font-size:11px;color:#333;margin-top:8px">Click "Evaluate on Test Set" above to run live evaluation</div></div>""", unsafe_allow_html=True)

    # Training history interactive charts
    history_path = os.path.join(config.LOG_DIR, "train_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        if history:
            epochs = [h["epoch"] for h in history]
            t_loss = [h["train"]["loss"] for h in history]
            v_loss = [h["val"]["loss"] for h in history]
            t_acc  = [h["train"]["acc"]  for h in history]
            v_acc  = [h["val"]["accuracy"] for h in history]
            st.markdown('<div class="section-header">Training History</div>', unsafe_allow_html=True)
            cl, cr = st.columns(2)
            with cl:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs, y=t_loss, name='Train Loss',
                    line=dict(color='#F44336', width=2), fill='tozeroy', fillcolor='rgba(244,67,54,0.05)'))
                fig_loss.add_trace(go.Scatter(x=epochs, y=v_loss, name='Val Loss',
                    line=dict(color='#42A5F5', width=2), fill='tozeroy', fillcolor='rgba(66,165,245,0.05)'))
                fig_loss.update_layout(**PLOTLY_LAYOUT, height=320, title="Loss Curve",
                    title_font=dict(size=13, color='#555'), legend=dict(font=dict(color='#A3A3A3')))
                st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})
            with cr:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=epochs, y=t_acc, name='Train Acc',
                    line=dict(color='#4CAF50', width=2), fill='tozeroy', fillcolor='rgba(76,175,80,0.05)'))
                fig_acc.add_trace(go.Scatter(x=epochs, y=v_acc, name='Val Acc',
                    line=dict(color='#FFC107', width=2), fill='tozeroy', fillcolor='rgba(255,193,7,0.05)'))
                fig_acc.update_layout(**PLOTLY_LAYOUT, height=320, title="Accuracy Curve",
                    title_font=dict(size=13, color='#555'), legend=dict(font=dict(color='#A3A3A3')))
                st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

    # OSA Convergence
    osa_log = os.path.join(config.LOG_DIR, "osa_log.json")
    if os.path.exists(osa_log):
        with open(osa_log) as f:
            osa_data = json.load(f)
        if isinstance(osa_data, list):
            fig_osa = go.Figure(go.Scatter(y=osa_data, mode='lines+markers',
                line=dict(color='#CE93D8', width=2), marker=dict(size=5, color='#CE93D8')))
            fig_osa.update_layout(**PLOTLY_LAYOUT, height=300, title="OSA Convergence",
                title_font=dict(size=13, color='#555'),
                xaxis_title="Iteration", yaxis_title="Best Fitness")
            st.plotly_chart(fig_osa, use_container_width=True, config={'displayModeBar': False})

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Prediction History
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Prediction History":
    st.markdown('<div class="section-header">Prediction History — Big Data Layer</div>', unsafe_allow_html=True)

    # ── Data source tabs ──────────────────────────────────────────────────────
    src_tab1, src_tab2, src_tab3 = st.tabs(["🗄  Cassandra Database", "📁  HDFS Prediction Files", "📊  Storage Analytics"])

    # ── TAB 1 : Cassandra ─────────────────────────────────────────────────────
    with src_tab1:
        cass_connected = cassandra_db.is_connected()

        if cass_connected:
            st.markdown("""<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
            <div style="width:8px;height:8px;border-radius:50%;background:#4CAF50;box-shadow:0 0 10px #4CAF5088"></div>
            <span style="font-size:12px;color:#4CAF50;letter-spacing:1px">CASSANDRA CONNECTED — bone_cancer_db.prediction_logs</span></div>""",
            unsafe_allow_html=True)

            # Stats row
            stats = cassandra_db.get_prediction_stats()
            if stats:
                cs1, cs2, cs3, cs4 = st.columns(4)
                for col, lbl, val, clr in [
                    (cs1, "Total Records", stats.get("total",     0), "#EAEAEA"),
                    (cs2, "Normal",        stats.get("normal",    0), "#4CAF50"),
                    (cs3, "Benign",        stats.get("benign",    0), "#FFC107"),
                    (cs4, "Malignant",     stats.get("malignant", 0), "#F44336"),
                ]:
                    with col:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-value" style="color:{clr}">{val}</div>
                        <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)
                st.markdown("")

            # Records table
            records = cassandra_db.get_recent_predictions(limit=100)
            if records:
                df_cass = pd.DataFrame(records)[["timestamp","image_name","prediction","confidence","normal_prob","benign_prob","malignant_prob"]]
                df_cass.columns = ["Timestamp","Image","Prediction","Confidence (%)","Normal (%)","Benign (%)","Malignant (%)"]
                st.dataframe(df_cass, use_container_width=True, hide_index=True)
                st.download_button("⬇  Export Cassandra Records (CSV)", data=df_cass.to_csv(index=False),
                                   file_name="cassandra_prediction_logs.csv", mime="text/csv", use_container_width=True)
            else:
                st.markdown("""<div class="glass-card" style="text-align:center;padding:40px">
                <div style="font-size:13px;color:#444">No records yet — run a prediction to log to Cassandra</div></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""<div class="glass-card" style="border-left:3px solid #F44336;padding:20px">
            <div style="font-size:13px;color:#F44336;font-weight:600;margin-bottom:8px">Cassandra Not Connected</div>
            <div style="font-size:12px;color:#888;line-height:1.8">
            To enable Cassandra logging, start a local instance:<br>
            <code style="background:#1A1A1A;padding:2px 8px;border-radius:4px;color:#CE93D8">
            docker run -d --name cassandra -p 9042:9042 cassandra:4.1</code><br><br>
            Then install the driver:<br>
            <code style="background:#1A1A1A;padding:2px 8px;border-radius:4px;color:#CE93D8">
            pip install cassandra-driver</code><br><br>
            The app continues to work fully — predictions are stored in HDFS (local files) as fallback.
            </div></div>""", unsafe_allow_html=True)

    # ── TAB 2 : HDFS files ────────────────────────────────────────────────────
    with src_tab2:
        hdfs_preds = hdfs_manager.list_predictions(limit=100)
        if hdfs_preds:
            # Summary metrics
            h1, h2, h3, h4 = st.columns(4)
            n_n = sum(1 for r in hdfs_preds if r.get("prediction") == "normal")
            n_b = sum(1 for r in hdfs_preds if r.get("prediction") == "benign")
            n_m = sum(1 for r in hdfs_preds if r.get("prediction") == "malignant")
            for col, lbl, val, clr in [(h1,len(hdfs_preds),"Total","#EAEAEA"),(h2,n_n,"Normal","#4CAF50"),
                                        (h3,n_b,"Benign","#FFC107"),(h4,n_m,"Malignant","#F44336")]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{clr}">{val}</div>
                    <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)
            st.markdown("")

            # Records table
            rows_html = ""
            for r in hdfs_preds:
                pred  = r.get("prediction","—")
                badge = f'<span class="result-badge badge-{pred}" style="font-size:10px;padding:3px 10px">{pred}</span>'
                rows_html += f"""<tr>
                  <td style="color:#666;font-size:11px">{r.get('timestamp','—')}</td>
                  <td style="color:#A3A3A3;font-size:12px">{r.get('image','—')}</td>
                  <td>{badge}</td>
                  <td style="color:#EAEAEA;font-weight:600;font-size:12px">{r.get('confidence','—')}%</td>
                  <td style="font-size:11px;color:#555">prediction_{r.get('id',0):04d}.json</td>
                </tr>"""

            st.markdown(f"""<div class="glass-card" style="padding:0;overflow:hidden">
            <table style="width:100%;border-collapse:collapse">
              <thead><tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                <th style="padding:12px 16px;text-align:left;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px">Timestamp</th>
                <th style="padding:12px 16px;text-align:left;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px">Image</th>
                <th style="padding:12px 16px;text-align:left;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px">Prediction</th>
                <th style="padding:12px 16px;text-align:left;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px">Confidence</th>
                <th style="padding:12px 16px;text-align:left;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px">HDFS File</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table></div>""", unsafe_allow_html=True)

            df_hdfs = pd.DataFrame(hdfs_preds)
            st.download_button("⬇  Export HDFS Records (CSV)", data=df_hdfs.to_csv(index=False),
                               file_name="hdfs_prediction_records.csv", mime="text/csv", use_container_width=True)
        else:
            st.markdown("""<div class="glass-card" style="text-align:center;padding:50px">
            <div style="font-size:40px;margin-bottom:12px;opacity:.1">📁</div>
            <div style="font-size:13px;color:#444">No HDFS prediction files yet</div>
            <div style="font-size:11px;color:#333;margin-top:6px">Run a prediction on the Detection Dashboard to populate HDFS</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 3 : Storage Analytics ─────────────────────────────────────────────
    with src_tab3:
        hdfs_stats = hdfs_manager.get_hdfs_stats()

        st.markdown('<div class="section-header">HDFS Namespace Usage</div>', unsafe_allow_html=True)
        ac1, ac2, ac3, ac4 = st.columns(4)
        ns_map = [
            (ac1, "Raw Images",   hdfs_stats["raw"]),
            (ac2, "Processed",    hdfs_stats["processed"]),
            (ac3, "Predictions",  hdfs_stats["predictions"]),
            (ac4, "Logs",         hdfs_stats["logs"]),
        ]
        for col, label, stat in ns_map:
            with col:
                st.markdown(f"""<div class="metric-card">
                <div style="font-size:22px;font-weight:800;color:#EAEAEA">{stat['count']}</div>
                <div style="font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-top:4px">{label}</div>
                <div style="font-size:10px;color:#444;margin-top:2px">{stat['size_mb']} MB</div>
                </div>""", unsafe_allow_html=True)

        # Bar chart — file counts per namespace
        ns_names  = ["Raw Images", "Processed", "Predictions", "Logs"]
        ns_counts = [hdfs_stats[k]["count"] for k in ["raw","processed","predictions","logs"]]
        ns_sizes  = [hdfs_stats[k]["size_mb"] for k in ["raw","processed","predictions","logs"]]

        st.markdown("")
        fig_hdfs = go.Figure()
        fig_hdfs.add_trace(go.Bar(name="Files", x=ns_names, y=ns_counts,
            marker_color=['#42A5F5','#4CAF50','#FFC107','#CE93D8'],
            text=ns_counts, textposition='outside', textfont=dict(color='#EAEAEA', size=13)))
        layout_hdfs = {**PLOTLY_LAYOUT, 'height': 300, 'title': 'HDFS File Count per Namespace',
                       'title_font': dict(size=13, color='#555'), 'yaxis_title': 'Files',
                       'showlegend': False, 'bargap': 0.5}
        fig_hdfs.update_layout(**layout_hdfs)
        st.plotly_chart(fig_hdfs, use_container_width=True, config={'displayModeBar': False})

        # HDFS path tree (visual)
        st.markdown('<div class="section-header">HDFS Directory Structure</div>', unsafe_allow_html=True)
        tree_rows = [
            ("hdfs/",               "#EAEAEA", "0px"),
            ("├── raw_images/",     "#42A5F5", "16px"),
            ("├── processed_images/","#4CAF50","16px"),
            ("├── predictions/",    "#FFC107", "16px"),
            ("└── logs/",           "#CE93D8", "16px"),
        ]
        tree_html = "".join(
            f'<div style="font-family:monospace;font-size:13px;color:{clr};padding:4px 0 4px {indent};'
            f'border-bottom:1px solid rgba(255,255,255,.03)">{name}'
            f'<span style="float:right;font-size:10px;color:#444">'
            f'{hdfs_stats.get(name.strip("├─└/ ").replace("_images","").replace("images","raw").replace("processed_images","processed").replace("predictions","predictions").replace("logs","logs"), {}).get("count","") if "/" in name and name.strip("├─└/ ") in ["raw_images","processed_images","predictions","logs"] else ""}'
            f'</span></div>'
            for name, clr, indent in tree_rows
        )
        # Simpler clean tree
        tree_lines = [
            ("hdfs/",                "#EAEAEA", "0",   ""),
            ("├── raw_images/",      "#42A5F5", "16px", f'{hdfs_stats["raw"]["count"]} files · {hdfs_stats["raw"]["size_mb"]} MB'),
            ("├── processed_images/","#4CAF50", "16px", f'{hdfs_stats["processed"]["count"]} files · {hdfs_stats["processed"]["size_mb"]} MB'),
            ("├── predictions/",     "#FFC107", "16px", f'{hdfs_stats["predictions"]["count"]} files · {hdfs_stats["predictions"]["size_mb"]} MB'),
            ("└── logs/",            "#CE93D8", "16px", f'{hdfs_stats["logs"]["count"]} files · {hdfs_stats["logs"]["size_mb"]} MB'),
        ]
        tree_html2 = "".join(
            f'<div style="font-family:monospace;font-size:13px;color:{clr};padding:6px 16px 6px {indent};'
            f'border-bottom:1px solid rgba(255,255,255,.03);display:flex;justify-content:space-between">'
            f'<span>{name}</span><span style="font-size:10px;color:#444">{info}</span></div>'
            for name, clr, indent, info in tree_lines
        )
        st.markdown(f'<div class="glass-card" style="padding:0;overflow:hidden">{tree_html2}</div>', unsafe_allow_html=True)

        # Cassandra schema
        st.markdown('<div class="section-header">Cassandra Schema</div>', unsafe_allow_html=True)
        cass_schema = [
            ("Keyspace",  "bone_cancer_db"),
            ("Table",     "prediction_logs"),
            ("id",        "UUID  PRIMARY KEY"),
            ("image_name","TEXT"),
            ("prediction","TEXT  (normal | benign | malignant)"),
            ("confidence","FLOAT"),
            ("normal_prob / benign_prob / malignant_prob", "FLOAT"),
            ("timestamp", "TEXT"),
        ]
        rows_s = "".join(
            f'<tr><td style="padding:10px 16px;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid rgba(255,255,255,.03);width:40%">{k}</td>'
            f'<td style="padding:10px 16px;font-size:12px;color:#EAEAEA;font-weight:600;border-bottom:1px solid rgba(255,255,255,.03)">{v}</td></tr>'
            for k, v in cass_schema
        )
        st.markdown(f"""<div class="glass-card" style="padding:0;overflow:hidden">
        <div style="padding:12px 16px;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,.05)">
          {'🟢 Connected' if cassandra_db.is_connected() else '🔴 Offline — using HDFS fallback'}</div>
        <table style="width:100%;border-collapse:collapse"><tbody>{rows_s}</tbody></table></div>""",
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Information
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Model Information":
    st.markdown('<div class="section-header">Model Architecture & Configuration</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown(f"""<div class="glass-card"><div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">Architecture</div>
        <table class="info-table">
        <tr><td>Backbone</td><td>{config.BACKBONE.upper()}</td></tr>
        <tr><td>Pretrained</td><td>{"ImageNet V2" if config.PRETRAINED else "No"}</td></tr>
        <tr><td>FC Hidden Units</td><td>{config.DEFAULT_FC_UNITS}</td></tr>
        <tr><td>Dropout</td><td>{config.DEFAULT_DROPOUT}</td></tr>
        <tr><td>Num Classes</td><td>{config.NUM_CLASSES}</td></tr>
        <tr><td>Input Size</td><td>{config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]}×{config.CHANNELS}</td></tr>
        <tr><td>Freeze Backbone</td><td>{"Yes" if config.FREEZE_BACKBONE else "No"}</td></tr>
        </table></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="glass-card"><div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">Training Configuration</div>
        <table class="info-table">
        <tr><td>Epochs</td><td>{config.NUM_EPOCHS}</td></tr>
        <tr><td>Batch Size</td><td>{config.BATCH_SIZE}</td></tr>
        <tr><td>Learning Rate</td><td>{config.DEFAULT_LR}</td></tr>
        <tr><td>Weight Decay</td><td>{config.DEFAULT_WEIGHT_DECAY}</td></tr>
        <tr><td>Momentum</td><td>{config.DEFAULT_MOMENTUM}</td></tr>
        <tr><td>Early Stop Patience</td><td>{config.EARLY_STOP_PATIENCE}</td></tr>
        <tr><td>Augmentation</td><td>{"Enabled" if config.AUGMENT_TRAIN else "Disabled"}</td></tr>
        </table></div>""", unsafe_allow_html=True)
    with cb:
        st.markdown(f"""<div class="glass-card"><div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">Owl Search Algorithm (OSA)</div>
        <table class="info-table">
        <tr><td>Population Size</td><td>{config.OSA_POPULATION}</td></tr>
        <tr><td>Max Iterations</td><td>{config.OSA_MAX_ITER}</td></tr>
        <tr><td>Dimensions</td><td>{config.OSA_DIM}</td></tr>
        <tr><td>LR Range</td><td>{config.OSA_LOWER_BOUNDS[0]} — {config.OSA_UPPER_BOUNDS[0]}</td></tr>
        <tr><td>Dropout Range</td><td>{config.OSA_LOWER_BOUNDS[1]} — {config.OSA_UPPER_BOUNDS[1]}</td></tr>
        <tr><td>FC Unit Choices</td><td>{', '.join(str(x) for x in config.FC_UNIT_CHOICES)}</td></tr>
        </table></div>""", unsafe_allow_html=True)

        train_count = sum(len(f) for _,_,f in os.walk(config.TRAIN_DIR)) if os.path.isdir(config.TRAIN_DIR) else "—"
        val_count = sum(len(f) for _,_,f in os.walk(config.VAL_DIR)) if os.path.isdir(config.VAL_DIR) else "—"
        test_count = sum(len(f) for _,_,f in os.walk(config.TEST_DIR)) if os.path.isdir(config.TEST_DIR) else "—"
        st.markdown(f"""<div class="glass-card"><div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">Dataset</div>
        <table class="info-table">
        <tr><td>Classes</td><td>{' · '.join(c.capitalize() for c in config.CLASS_NAMES)}</td></tr>
        <tr><td>Train Images</td><td>{train_count}</td></tr>
        <tr><td>Validation Images</td><td>{val_count}</td></tr>
        <tr><td>Test Images</td><td>{test_count}</td></tr>
        </table></div>""", unsafe_allow_html=True)

        if model is not None:
            total_p = sum(p.numel() for p in model.parameters())
            train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.markdown(f"""<div class="glass-card"><div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">Model Parameters</div>
            <table class="info-table">
            <tr><td>Total Parameters</td><td>{total_p:,}</td></tr>
            <tr><td>Trainable</td><td>{train_p:,}</td></tr>
            <tr><td>Frozen</td><td>{total_p - train_p:,}</td></tr>
            </table></div>""", unsafe_allow_html=True)

    # Pipeline visualization
    st.markdown("""<div class="glass-card" style="text-align:center"><div style="font-size:11px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px">Processing Pipeline</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap">""" +
    "".join(f"""<div style="background:#1A1A1A;padding:10px 18px;border-radius:8px;border:1px solid rgba(255,255,255,.05);font-size:12px;color:{'#EAEAEA' if i in [2,5] else '#A3A3A3'};font-weight:{'600' if i in [2,5] else '500'}">{s}</div><div style="color:#333;font-size:16px">{'→' if i < 5 else ''}</div>"""
    for i, s in enumerate(["X-Ray Input","Preprocessing","ResNet-50","Classification Head","Softmax","Diagnosis"])) +
    "</div></div>", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""<div class="footer-bar">Bone Cancer Detection System · Deep Learning · Owl Search Optimization · Medical AI</div>""", unsafe_allow_html=True)