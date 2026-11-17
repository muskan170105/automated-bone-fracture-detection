"""Shared helpers for the Streamlit website and scripts."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DISCLAIMER = (
    "This application is an educational and research prototype. It is NOT clinically validated, "
    "is NOT a medical device and must NOT be used for diagnosis or treatment decisions. "
    "All results require review by a qualified radiologist."
)


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_json(path):
    """Return parsed JSON or None if the file is missing/corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def results_dir() -> Path:
    return ROOT / "results"


def model_path(cfg=None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg.get("model", {}).get("weights", "models/best.pt"))
    return p if p.is_absolute() else ROOT / p


def load_css() -> str:
    css = ROOT / "assets" / "styles.css"
    return css.read_text(encoding="utf-8") if css.exists() else ""


def html(block: str) -> None:
    """Render an HTML block (indentation/blank lines stripped so Markdown does not treat it as code)."""
    import streamlit as st

    st.markdown("\n".join(l.strip() for l in block.splitlines() if l.strip()), unsafe_allow_html=True)


def setup_page(title: str, icon: str = "🩻") -> None:
    """Must be the first Streamlit call on every page."""
    import streamlit as st

    st.set_page_config(page_title=f"{title} | FractureAI", page_icon=icon, layout="wide")
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)
    logo = ROOT / "assets" / "logo.png"
    if logo.exists():
        st.sidebar.image(str(logo), width=96)
    st.sidebar.markdown("**FractureAI**  \nResearch prototype")
    st.sidebar.caption("Not for clinical use.")


def page_header(title: str, subtitle: str = "") -> None:
    html(f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>')


def render_disclaimer() -> None:
    html(f'<div class="disclaimer"><b>⚠️ Medical research disclaimer.</b> {DISCLAIMER}</div>')
