"""
Software Name: Profiler – Desktop Edition
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 15/04/2026
Version: 1.2.3
Context:
Desktop version of Profiler — no login, no internet, no account required. All data stays local.
Start with: streamlit run profiler_gui_desktop.py
It is designed for archiving on Zenodo and integration into GitHub releases.

License: l’Agence pour la Protection des Programmes IDDN (InterDeposit Digital Number) : FR2 .0013 .0300044 .0005 .S6 .C7 .20258 .0009 .312301
Citation:
If Profiler or this module (a part of Profiler) is used in a publication, please cite:
Zirem, Y. (2025). Profiler: an open web platform for multi-omics analysis. Journal of Bioinformatics. [DOI or Zenodo/GitHub link available in the article].

Links:
- GitHub temporary Repository: https://github.com/yanisZirem/Profiler_v1_requests_datatests
"""


import os
import tempfile
import shutil
import uuid
import zipfile
import threading
import functools
import time
import datetime
import gc
import re
import io
import base64
import traceback
import multiprocessing
from typing import Callable, Any
from pathlib import Path
from itertools import combinations
from collections import Counter

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gseapy as gp
import kaleido
import plotly.io as pio

from neurocombat_sklearn import CombatModel
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import f_oneway, gaussian_kde
from statsmodels.stats.multitest import multipletests
from scipy import stats

# ─── Ensure project root (Profiler/) is on sys.path ──────────────────────────

import sys as _sys_path_fix
from pathlib import Path as _Path_fix
_GUI_DIR      = _Path_fix(__file__).resolve().parent        # app/gui/
_APP_DIR      = _GUI_DIR.parent                             # app/
_PROJECT_ROOT = _APP_DIR.parent                             # Profiler/
for _p in (_PROJECT_ROOT, _APP_DIR):
    if str(_p) not in _sys_path_fix.path:
        _sys_path_fix.path.insert(0, str(_p))

from app.utils.profiler_imports import *
from app.analysis.profiler_features_importance import _resolve_features

# ─── Helper: build feature matrix (excludes Class, ID, File, RT, Sum, _meta) ──
_NON_FEATURE_COLS = {'Class', 'ID', 'File', 'RT', 'Sum', 'Original_index'}

def _get_X(df, extra_drop=None):
    """Return numeric feature matrix, dropping metadata columns + _meta suffix cols."""
    drop = set(_NON_FEATURE_COLS)
    if extra_drop:
        drop.update(extra_drop)
    # also drop any _meta column
    drop.update(c for c in df.columns if str(c).endswith('_meta'))
    cols = [c for c in df.columns if c not in drop]
    return df[cols].select_dtypes(include='number')

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.stats import f_oneway
import io


# ── Performance: pandas Copy-on-Write (pandas 2+) — avoid hidden copies ──────
try:
    pd.options.mode.copy_on_write = True
except Exception:
    pass
# ── Use all available threads for numpy/scipy (BLAS) ─────────────────────────
import os as _os_perf
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    _os_perf.environ.setdefault(_env, '10')
import plotly.express as px
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy import stats

import kaleido

# ═══════════════════════════════════════════════════
# PROFILER CUSTOM SVG ICONS
# ═══════════════════════════════════════════════════
_PICONS = {
    'home':        '<svg width="16" height="16" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><polygon points="9,1 16,4.5 16,13.5 9,17 2,13.5 2,4.5" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linejoin="round"/><circle cx="9" cy="9" r="2" fill="currentColor"/></svg>',
    'datalab':     '<svg width="16" height="16" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 2v9l-3 3.5a1 1 0 00.8 1.5h10.4a1 1 0 00.8-1.5L12 11V2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><line x1="6" y1="2" x2="12" y2="2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    'dataviz':     '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><polyline points="2,13 5,8 8,11 11,5 14,9 16,6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/><circle cx="11" cy="5" r="2" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="11" cy="5" r="0.6" fill="currentColor"/></svg>',
    'comparisons': '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><circle cx="7" cy="9" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="11" cy="9" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
    'aimodeling':  '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><circle cx="3" cy="6" r="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="3" cy="12" r="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="9" cy="4" r="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="9" cy="9" r="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="9" cy="14" r="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="15" cy="9" r="1.5" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="4.5" y1="6" x2="7.5" y2="4" stroke="currentColor" stroke-width="0.8" opacity="0.7"/><line x1="4.5" y1="6" x2="7.5" y2="9" stroke="currentColor" stroke-width="0.8" opacity="0.7"/><line x1="4.5" y1="12" x2="7.5" y2="9" stroke="currentColor" stroke-width="0.8" opacity="0.7"/><line x1="4.5" y1="12" x2="7.5" y2="14" stroke="currentColor" stroke-width="0.8" opacity="0.7"/><line x1="10.5" y1="4" x2="13.5" y2="9" stroke="currentColor" stroke-width="0.8" opacity="0.7"/><line x1="10.5" y1="9" x2="13.5" y2="9" stroke="currentColor" stroke-width="0.8" opacity="0.7"/><line x1="10.5" y1="14" x2="13.5" y2="9" stroke="currentColor" stroke-width="0.8" opacity="0.7"/></svg>',
    'biomarkers':  '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="7" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="9" cy="9" r="3.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="9" cy="9" r="1" fill="currentColor"/><line x1="9" y1="1" x2="9" y2="5.5" stroke="currentColor" stroke-width="1.2" opacity="0.4"/><line x1="9" y1="12.5" x2="9" y2="17" stroke="currentColor" stroke-width="1.2" opacity="0.4"/><line x1="1" y1="9" x2="5.5" y2="9" stroke="currentColor" stroke-width="1.2" opacity="0.4"/><line x1="12.5" y1="9" x2="17" y2="9" stroke="currentColor" stroke-width="1.2" opacity="0.4"/></svg>',
    'enrichment':  '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><path d="M5 2c0 4 8 3 8 7s-8 3-8 7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"/><path d="M13 2c0 4-8 3-8 7s8 3 8 7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none" opacity="0.5"/><line x1="5.5" y1="6" x2="12.5" y2="6" stroke="currentColor" stroke-width="1" opacity="0.6"/><line x1="5.5" y1="12" x2="12.5" y2="12" stroke="currentColor" stroke-width="1" opacity="0.6"/></svg>',
    'survival':    '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><path d="M2 4h3v4h3v4h3v4h5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/><line x1="2" y1="16" x2="16" y2="16" stroke="currentColor" stroke-width="1" opacity="0.3"/><line x1="2" y1="4" x2="2" y2="16" stroke="currentColor" stroke-width="1" opacity="0.3"/></svg>',
    'realtime':    '<svg width="16" height="16" viewBox="0 0 18 18" fill="none"><polyline points="1,9 4,9 5,5 6,13 7,7 8,11 9,9 17,9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>',
    'load':        '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="14" height="14" rx="2" stroke="currentColor" stroke-width="1.3" fill="none"/><line x1="1" y1="5" x2="15" y2="5" stroke="currentColor" stroke-width="1" opacity="0.5"/><line x1="5" y1="5" x2="5" y2="15" stroke="currentColor" stroke-width="1" opacity="0.5"/><path d="M8 7.5v5M5.5 10.5L8 13l2.5-2.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    'overview':    '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><circle cx="7" cy="7" r="4.5" stroke="currentColor" stroke-width="1.3" fill="none"/><line x1="10.5" y1="10.5" x2="14" y2="14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="5" y1="7" x2="9" y2="7" stroke="currentColor" stroke-width="1" opacity="0.6"/><line x1="7" y1="5" x2="7" y2="9" stroke="currentColor" stroke-width="1" opacity="0.6"/></svg>',
    'missing':     '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="5.5" height="5.5" rx="1" stroke="currentColor" stroke-width="1.2" fill="none"/><rect x="9.5" y="1" width="5.5" height="5.5" rx="1" stroke="currentColor" stroke-width="1.2" fill="none"/><rect x="1" y="9.5" width="5.5" height="5.5" rx="1" stroke="currentColor" stroke-width="1.2" fill="none"/><rect x="9.5" y="9.5" width="5.5" height="5.5" rx="1" stroke="currentColor" stroke-width="1.2" fill="none" stroke-dasharray="2 1.5"/><line x1="11" y1="11" x2="14" y2="14" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" opacity="0.5"/><line x1="14" y1="11" x2="11" y2="14" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" opacity="0.5"/></svg>',
    'distrib':     '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M1 13 C3 13 4 4 8 4 C12 4 13 13 15 13" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" fill="none"/><line x1="8" y1="4" x2="8" y2="13" stroke="currentColor" stroke-width="1" stroke-dasharray="1.5 1.5" opacity="0.5"/><line x1="1" y1="13" x2="15" y2="13" stroke="currentColor" stroke-width="1" opacity="0.4"/></svg>',
    'rename':      '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M10 2.5l3.5 3.5-7 7H3v-3.5l7-7z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" fill="none"/><line x1="8" y1="4.5" x2="11.5" y2="8" stroke="currentColor" stroke-width="1" opacity="0.5"/></svg>',
    'edit':        '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><circle cx="4" cy="4" r="2" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="4" cy="12" r="2" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="5.5" y1="5" x2="13" y2="8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><line x1="5.5" y1="11" x2="13" y2="8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><circle cx="13" cy="8" r="1" fill="currentColor"/></svg>',
    'savedata':    '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><rect x="2" y="2" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.3" fill="none"/><path d="M5 2v4h6V2" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round" fill="none"/><rect x="9" y="2.5" width="1" height="2.5" rx="0.4" fill="currentColor"/><rect x="5" y="9" width="6" height="4" rx="1" stroke="currentColor" stroke-width="1" fill="none"/></svg>',
    'preprocess':  '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M2 3h12l-4.5 5v5l-3-1.5V8L2 3z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" fill="none"/></svg>',
    'postqc':      '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M8 2L3 4v4c0 3 2.5 5 5 6 2.5-1 5-3 5-6V4L8 2z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" fill="none"/><path d="M5.5 8l2 2 3-3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    'balance':     '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><line x1="8" y1="2" x2="8" y2="14" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><line x1="3" y1="6" x2="13" y2="6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M3 6 C3 8 6 9 6 6" stroke="currentColor" stroke-width="1.2" fill="none"/><path d="M13 6 C13 8 10 9 10 6" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="6" y1="2" x2="10" y2="2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',
    'sampleqc':    '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><rect x="6" y="1.5" width="4" height="7" rx="1" stroke="currentColor" stroke-width="1.2" fill="none"/><line x1="8" y1="8.5" x2="8" y2="11" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M5 11h6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M4 13.5h8" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="8" cy="4.5" r="1" stroke="currentColor" stroke-width="1"/></svg>',
    'sampling':    '<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M10 2l4 4-6 6H4V8l6-6z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round" fill="none"/><path d="M4 12l-2 2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="7" cy="9" r="1" fill="currentColor"/></svg>',
}

def _picon(key, label, color="#318CE7", size=15):
    """Render a custom Profiler icon + label as an HTML inline element."""
    svg = _PICONS.get(key, '')
    svg_colored = svg.replace('currentColor', color)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'font-size:0.85rem;font-weight:600;color:{color};">'
        f'{svg_colored}<span style="color:#1e3a5f;">{label}</span></span>'
    )



# ─── Data source resolver (used across all tabs) ───────────────────────────────

def _resolve_data_source(source_name: str) -> "pd.DataFrame | None":
    """
    Resolve a data source name to the corresponding DataFrame.
    Used consistently across all analysis tabs.
    """
    mapping = {
        'Raw Data': lambda: st.session_state.get('final_data') or st.session_state.get('data'),
        'Raw data': lambda: st.session_state.get('final_data') or st.session_state.get('data'),
        'Raw': lambda: st.session_state.get('final_data') or st.session_state.get('data'),
        'Edited/Renamed': lambda: st.session_state.get('final_data'),
        'Preprocessed': lambda: st.session_state.get('preprocessed_data'),
        'Oversampled': lambda: st.session_state.get('oversampled_data'),
        'Undersampled': lambda: st.session_state.get('undersampled_data'),
        'Preprocessed + Oversampled': lambda: st.session_state.get('oversampled_data'),
        'Preprocessed + Undersampled': lambda: st.session_state.get('undersampled_data'),
    }
    resolver = mapping.get(source_name)
    if resolver:
        return resolver()
    return None


def _resolve_features_df(df: "pd.DataFrame", features: list) -> list:
    """
    Resolve a list of feature names against the actual columns of a DataFrame.

    Handles the mzML case where column names are floats (e.g. 100.07) but the
    caller may hold them as strings (e.g. '100.07') or vice-versa.  Returns
    only the features that are present in the DataFrame, preserving order and
    converting each element to the exact type used by the DataFrame columns so
    the returned list can be used directly for column indexing.
    """
    if df is None or not features:
        return []

    col_set = set(df.columns)

    # Build a str->actual-col lookup for float columns (mzML m/z values)
    str_to_col = {}
    for col in df.columns:
        if isinstance(col, float):
            str_to_col[str(col)] = col
            # Also cover truncated representations like '100.07' vs '100.0700…'
            try:
                str_to_col[f"{col:g}"] = col
            except (ValueError, TypeError):
                pass

    resolved = []
    for f in features:
        if f in col_set:
            resolved.append(f)
        elif isinstance(f, str) and f in str_to_col:
            resolved.append(str_to_col[f])
        elif isinstance(f, float) and str(f) in {str(c) for c in col_set}:
            # feature is float but columns are stored as strings
            for col in df.columns:
                try:
                    if float(col) == f:
                        resolved.append(col)
                        break
                except (ValueError, TypeError):
                    pass
    return resolved


def check_stop(message="Analysis stopped by user"):
    """
    Vérifie si l'utilisateur a demandé l'arrêt.
    Si oui, affiche un message et arrête l'exécution.
    """
    if st.session_state.get('stop_analysis', False):
        st.warning(f"⚠️ {message}")
        st.session_state['stop_analysis'] = False  # Reset
        st.stop()


# ── Page icon: profiler_logo.png from same folder, SVG fallback ──────────────
def _load_page_icon():
    """Load profiler_logo.png as page icon; fall back to embedded SVG."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _assets_dir = os.path.join(_script_dir, "..", "assets")
    for _candidate in [
        os.path.join(_assets_dir, "profiler_logo.png"),
        os.path.join(_script_dir, "profiler_logo.png"),
        os.path.join(_script_dir, "logo", "Log2.png"),
        "profiler_logo.png",
    ]:
        if os.path.exists(_candidate):
            return _candidate  # Streamlit accepts a file path string directly
    # Fallback: embedded SVG as data-URI
    _svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#1a4a80"/>'
        '<path d="M10 4 C14 9 18 9 22 14 C18 19 14 19 10 24 C14 28 18 28 22 24"'
        ' stroke="#5ab4ff" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        '<path d="M22 4 C18 9 14 9 10 14 C14 19 18 19 22 24 C18 28 14 28 10 24"'
        ' stroke="#ffffff" stroke-width="2.2" fill="none" stroke-linecap="round" opacity="0.85"/>'
        '<line x1="10" y1="14" x2="22" y2="14" stroke="#5ab4ff" stroke-width="1.4" opacity="0.6"/>'
        '</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(_svg.encode()).decode()

st.set_page_config(
    page_title="Profiler Offline",
    page_icon=_load_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yanisZirem/Profiler_v1_requests_datatests',
        'About': 'Profiler Desktop v1.2 — Local offline multi-omics analysis. No login required.'
    }
)

# ── Professional theme ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Core resets ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Global typography ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ── Sidebar – fond clair professionnel ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f5fc 0%, #e8f0f9 60%, #dde8f5 100%) !important;
    border-right: 2px solid #c5d8ee !important;
    box-shadow: 3px 0 16px rgba(49,140,231,0.08) !important;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #1e3a5f !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #1e3a5f !important; font-size: 0.78rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #ffffff !important;
    border-color: #94b8d8 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #1e293b !important; font-weight: 500 !important; }
[data-testid="stSidebar"] .stExpander {
    border: 1px solid #c0d4ea !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
    background: rgba(255,255,255,0.75) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stSidebar"] .stExpander:hover {
    border-color: #318CE7 !important;
    box-shadow: 0 2px 8px rgba(49,140,231,0.12) !important;
}
[data-testid="stSidebar"] .stExpander summary {
    color: #1a4a80 !important; font-weight: 800 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] details[open] summary {
    color: #1565c0 !important;
    border-bottom: 1px solid #c0d4ea !important;
    background: rgba(49,140,231,0.05) !important;
    border-radius: 10px 10px 0 0 !important;
}
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    color: #1a4a80 !important;
    border: 2px solid #318CE7 !important;
    border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 4px rgba(49,140,231,0.1) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #318CE7 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(49,140,231,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: #ffffff !important;
    border-color: #ef4444 !important;
    color: #dc2626 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: #ef4444 !important;
    color: #fff !important;
}
/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(49,140,231,0.04) !important;
    border: 2px dashed #318CE7 !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    background: rgba(49,140,231,0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(135deg,#318CE7,#1a65c0) !important;
    color: #fff !important; border: none !important; border-radius: 6px !important;
}
/* Sidebar hr */
[data-testid="stSidebar"] hr {
    border-color: #c0d4ea !important;
    margin: 10px 0 !important;
}
/* Sidebar checkbox */
[data-testid="stSidebar"] [data-baseweb="checkbox"] span {
    border-color: #318CE7 !important;
    background: rgba(49,140,231,0.08) !important;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Tab navigation ── */
div[data-baseweb="tab-list"] {
    gap: 3px;
    background: linear-gradient(135deg, #0d1f3c 0%, #1a2f50 100%);
    padding: 6px;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.18);
    width: 100% !important;
    display: flex !important;
    flex-wrap: wrap !important;
    overflow: hidden !important;
}
button[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 6px 8px !important;
    font-weight: 700 !important;
    font-size: 0.73rem !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
    color: #1e293b !important;
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    letter-spacing: 0.01em !important;
    text-shadow: none !important;
    opacity: 1 !important;
    flex: 1 1 auto !important;
    min-width: 60px !important;
    max-width: 130px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
@media (max-width: 900px) {
    button[data-baseweb="tab"] {
        font-size: 0.65rem !important;
        padding: 5px 5px !important;
        min-width: 50px !important;
    }
}
@media (max-width: 600px) {
    div[data-baseweb="tab-list"] {
        gap: 2px !important;
        padding: 4px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 0.6rem !important;
        padding: 4px 4px !important;
        min-width: 40px !important;
        max-width: 90px !important;
    }
}
button[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.92) !important;
    color: #1e293b !important;
    border-color: rgba(255,255,255,0.8) !important;
}
/* ── Tab 1 Home – bleu ── */
button[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
    background: linear-gradient(135deg,#318CE7,#1a65c0) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(49,140,231,0.45) !important;
    border-color: rgba(99,174,255,0.4) !important;
}
/* ── Tab 2 Data Lab – violet ── */
button[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
    background: linear-gradient(135deg,#8b5cf6,#6d28d9) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(139,92,246,0.45) !important;
    border-color: rgba(167,139,250,0.4) !important;
}
/* ── Tab 3 Data Viz – cyan ── */
button[data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
    background: linear-gradient(135deg,#06b6d4,#0284c7) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(6,182,212,0.45) !important;
    border-color: rgba(103,232,249,0.4) !important;
}
/* ── Tab 4 Comparisons – indigo ── */
button[data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#4338ca) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.45) !important;
    border-color: rgba(165,180,252,0.4) !important;
}
/* ── Tab 5 AI Modeling – rose ── */
button[data-baseweb="tab"]:nth-child(5)[aria-selected="true"] {
    background: linear-gradient(135deg,#ec4899,#be185d) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(236,72,153,0.45) !important;
    border-color: rgba(249,168,212,0.4) !important;
}
/* ── Tab 6 Biomarkers – orange ── */
button[data-baseweb="tab"]:nth-child(6)[aria-selected="true"] {
    background: linear-gradient(135deg,#f97316,#ea580c) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(249,115,22,0.45) !important;
    border-color: rgba(253,186,116,0.4) !important;
}
/* ── Tab 7 Enrichment – vert émeraude ── */
button[data-baseweb="tab"]:nth-child(7)[aria-selected="true"] {
    background: linear-gradient(135deg,#10b981,#047857) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(16,185,129,0.45) !important;
    border-color: rgba(110,231,183,0.4) !important;
}
/* ── Tab 8 Survival – teal ── */
button[data-baseweb="tab"]:nth-child(8)[aria-selected="true"] {
    background: linear-gradient(135deg,#14b8a6,#0f766e) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(20,184,166,0.45) !important;
    border-color: rgba(94,234,212,0.4) !important;
}
/* ── Tab 9 Real-Time – amber ── */
button[data-baseweb="tab"]:nth-child(9)[aria-selected="true"] {
    background: linear-gradient(135deg,#f59e0b,#d97706) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(245,158,11,0.45) !important;
    border-color: rgba(252,211,77,0.4) !important;
}

/* ── Expanders ── */
details.st-expander {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
details.st-expander summary {
    font-weight: 600 !important;
    padding: 10px 14px !important;
}
details.st-expander[open] summary {
    border-bottom: 1px solid #e2e8f0 !important;
    background: #f8fafc !important;
    border-radius: 10px 10px 0 0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #318CE7 0%, #2060c0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(49,140,231,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(49,140,231,0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 6px rgba(16,185,129,0.25) !important;
}

/* ── Forms ── */
[data-testid="stForm"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    background: #fafbfd !important;
}

/* ── Login / Signup input fields styled like file uploaders ── */
[data-testid="stSidebar"] [data-testid="stForm"] [data-baseweb="input"] {
    border: 2px dashed rgba(49,140,231,0.5) !important;
    border-radius: 10px !important;
    background: rgba(49,140,231,0.04) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stForm"] [data-baseweb="input"]:hover,
[data-testid="stSidebar"] [data-testid="stForm"] [data-baseweb="input"]:focus-within {
    border-color: #318CE7 !important;
    background: rgba(49,140,231,0.09) !important;
    box-shadow: 0 0 0 3px rgba(49,140,231,0.12) !important;
}
[data-testid="stSidebar"] [data-testid="stForm"] input {
    background: transparent !important;
    color: #1e293b !important;
    font-size: 0.85rem !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
    background: #f8fafc;
}

/* ── Info/Warning/Success boxes ── */
.stAlert {
    border-radius: 8px !important;
}

/* ── DataFrames ── */
.dataframe {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #318CE7, #60a5fa) !important;
    border-radius: 4px !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #1e40af;
    border-bottom: 2px solid #318CE7; padding: 6px 0 5px 12px;
    margin: 14px 0 10px 0; background: linear-gradient(90deg, #eff6ff 0%, transparent 100%);
    border-radius: 4px 4px 0 0;
}
/* File uploader visibility */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(49,140,231,0.4) !important; border-radius: 10px !important;
    padding: 4px !important; background: rgba(49,140,231,0.03) !important; transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #318CE7 !important; background: rgba(49,140,231,0.07) !important; }
[data-testid="stFileUploader"] button { background: #318CE7 !important; color: white !important; border: none !important; border-radius: 6px !important; }
/* Sub-tab styling */
div[data-baseweb="tab-list"] { gap: 4px; background: #e8eef5; padding: 4px; border-radius: 10px; }

/* ── Sidebar clean redesign ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f5fc 0%, #e8f0f9 60%, #dde8f5 100%) !important;
}
[data-testid="stSidebar"] .stButton > button {
    font-size: 0.78rem !important;
    padding: 6px 10px !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] .stExpander {
    border: 1px solid #d0e2f3 !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.7) !important;
    margin-bottom: 6px !important;
    box-shadow: 0 1px 4px rgba(49,140,231,0.06) !important;
}
[data-testid="stSidebar"] .stExpander:hover {
    border-color: #318CE7 !important;
    box-shadow: 0 2px 8px rgba(49,140,231,0.14) !important;
}
[data-testid="stSidebar"] .stExpander > details > summary {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #1a4a80 !important;
    padding: 9px 12px !important;
}
/* ── Main content full width ── */
.main .block-container {
    max-width: 100% !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)
# ── End theme ────────────────────────────────────────────────

if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
# Liste des étapes
workflow_steps = [
    "🧭 <strong>Data Conversion (if needed)</strong><br><small>Convert raw files (e.g., .raw) to mzML using MSConvert.</small>",
    "📥 <strong>Data Import</strong><br><small>Upload tabular omics datasets (e.g., CSV, mzML, MaxQuant).</small>",
    "🧹 <strong>Preprocessing</strong><br><small>Clean, normalize,correct batches, impute, and transform your data.</small>",
    "🔍 <strong>Exploratory Analysis</strong><br><small>Check distributions, correlations, and class balance.</small>",
    "🌀 <strong>Clustering & Heterogeneity</strong><br><small>Discover natural groups or study tumor heterogeneity.</small>",
    "📊 <strong>Visualization</strong><br><small>Generate PCA, t-SNE, heatmaps, volcano plots, etc.</small>",
    "📈 <strong>Statistical Modeling</strong><br><small>Differential analysis and hypothesis testing.</small>",
    "🤖 <strong>Machine Learning</strong><br><small>Train, evaluate, and interpret predictive models.</small>",
    "🕸️ <strong>Pathway & Enrichment</strong><br><small>Biological interpretation via enrichment analysis.</small>",
    "⏳ <strong>Survival Analysis</strong><br><small>Model survival outcomes and stratify risk groups.</small>",
    "⚡ <strong>Real-Time Prediction</strong><br><small>Predict instantly from new unseen data.</small>",
]
st.markdown(
    """
    <style>
        .sidebar .block-container {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .sidebar img {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s;
        }
        .sidebar img:hover {
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)




# ─── Data source resolver (used across all tabs) ───────────────────────────────

def _resolve_data_source(source_name: str) -> "pd.DataFrame | None":
    """
    Resolve a data source name to the corresponding DataFrame.
    Used consistently across all analysis tabs.
    """
    mapping = {
        'Raw Data': lambda: st.session_state.get('final_data') or st.session_state.get('data'),
        'Raw data': lambda: st.session_state.get('final_data') or st.session_state.get('data'),
        'Raw': lambda: st.session_state.get('final_data') or st.session_state.get('data'),
        'Edited/Renamed': lambda: st.session_state.get('final_data'),
        'Preprocessed': lambda: st.session_state.get('preprocessed_data'),
        'Oversampled': lambda: st.session_state.get('oversampled_data'),
        'Undersampled': lambda: st.session_state.get('undersampled_data'),
        'Preprocessed + Oversampled': lambda: st.session_state.get('oversampled_data'),
        'Preprocessed + Undersampled': lambda: st.session_state.get('undersampled_data'),
    }
    resolver = mapping.get(source_name)
    if resolver:
        return resolver()
    return None


def _section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header consistently across all tabs."""
    sub_html = f'<p style="color:#6b7280;font-size:0.82rem;margin:2px 0 0 12px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
        <div class="section-header">
            {title}
        </div>{sub_html}
    """, unsafe_allow_html=True)



def initialize_session_state():
    session_vars = [
        'data', 'preprocessed_data', 'oversampled_data', 'undersampled_data',
        'compressed_data', 'reduced_data', 'models', 'dl_models', 'class_colors',
        'oversampling_technique', 'undersampling_technique', 'feature_distribution_plot',
        'mean_spectrum_plot', 'individual_spectra_plot', 'pc_index', 'file_groups',
        'class_renaming', 'rename_pending', 'feature_data_source_1', 'feature_data_source_2',
        'feature_data_source_3', 'show_shap', 'show_lime', 'show_boxplots', 'show_heatmap',
        'show_volcano', 'class_column', 'is_maxquant', 'selected_feature_row',
        'selected_data_source', 'latest_result', 'processed_files', 'monitoring',
        'normalization', 'numeric_features', 'label_encoder', 'svd_model', 'label_colors',
        'selected_feature', 'survival_data', 'expand_load_data', 'workflow_step', 'workflow_active'
    ]

    for var in session_vars:
        if var not in st.session_state:
            if var in ['models', 'dl_models', 'class_colors', 'label_colors', 'class_renaming', 'rename_pending', 'processed_files']:
                st.session_state[var] = {}
            elif var in ['oversampling_technique', 'undersampling_technique', 'feature_data_source_1', 'feature_data_source_2', 'feature_data_source_3', 'selected_feature_row', 'selected_data_source', 'normalization']:
                st.session_state[var] = 'None'
            elif var in ['numeric_features']:
                st.session_state[var] = []
            elif var in ['pc_index']:
                st.session_state[var] = 0
            elif var in ['is_maxquant', 'monitoring', 'show_shap', 'show_lime', 'show_boxplots', 'show_heatmap', 'show_volcano', 'expand_load_data']:
                st.session_state[var] = False  
            elif var == 'class_column':
                st.session_state[var] = 'Class'
            elif var == 'file_groups':
                st.session_state[var] = []  
            else:
                st.session_state[var] = None





# ════════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATEUR DE RAPPORT HTML
# ════════════════════════════════════════════════════════════════════════════════


# ─── helpers ──────────────────────────────────────────────────────────────────

def _capture_plotly(fig, key: str):
    """Store a Plotly figure in session_state for the HTML report."""
    if fig is not None:
        st.session_state[f"_report_{key}"] = ("plotly", fig)


def _capture_matplotlib(fig, key: str):
    """Convert a Matplotlib figure to base64 PNG and store for the HTML report."""
    if fig is not None:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
            buf.seek(0)
            st.session_state[f"_report_{key}"] = ("b64", base64.b64encode(buf.read()).decode())
        except Exception:
            pass


def _fig_to_b64(entry) -> str:
    """Convert a stored figure entry to base64 PNG string."""
    if entry is None:
        return ""
    kind, fig = entry
    if kind == "b64":
        return fig  # already base64
    if kind == "plotly":
        try:
            img = pio.to_image(fig, format='png', width=800, height=480, scale=1.5)
            return base64.b64encode(img).decode()
        except Exception:
            return ""
    return ""


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    trunc = df.head(max_rows)
    html = trunc.to_html(
        index=False, classes="profiler-table", border=0,
        float_format=lambda x: f"{x:.4g}" if isinstance(x, float) else str(x)
    )
    if len(df) > max_rows:
        html += f'<p class="table-note">Table truncated — {len(df):,} rows total (showing first {max_rows}).</p>'
    return html


def _fmt_val(v) -> str:
    if v is None or v == 'N/A':
        return '—'
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


# ─── main report generator ────────────────────────────────────────────────────

def generate_html_report() -> str:
    """Generate a complete HTML analysis report from session_state — all plots included."""
    ss = st.session_state
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root{--blue:#318CE7;--blue-dark:#1a5fa8;--blue-light:#e8f4fd;--text:#1a1a2e;
      --muted:#6b7280;--bg:#f8fafc;--white:#fff;--border:#e2e8f0;
      --green:#10b981;--orange:#f59e0b;--red:#ef4444;--radius:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;overflow-x:hidden}
.report-header{background:linear-gradient(135deg,var(--blue-dark),var(--blue) 55%,#5ba8ff);
  color:#fff;padding:52px 64px 40px;position:relative;overflow:hidden}
.report-header::before{content:'';position:absolute;top:-80px;right:-80px;
  width:360px;height:360px;background:rgba(255,255,255,.05);border-radius:50%}
.report-header::after{content:'';position:absolute;bottom:-50px;left:25%;
  width:220px;height:220px;background:rgba(255,255,255,.04);border-radius:50%}
.report-header h1{font-size:2.2rem;font-weight:700;letter-spacing:-.5px;position:relative}
.report-header .subtitle{font-size:.95rem;opacity:.85;margin-top:6px;position:relative}
.report-header .meta-bar{display:flex;gap:10px;margin-top:20px;font-size:.82rem;flex-wrap:wrap;position:relative}
.report-header .meta-bar span{background:rgba(255,255,255,.18);padding:4px 14px;border-radius:20px}
.toc{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);
  padding:24px 36px;margin:32px 64px;overflow:visible}
.toc h3{font-size:.75rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin-bottom:12px}
.toc ol{padding-left:20px;columns:2;column-gap:40px}
.toc li{margin:5px 0;break-inside:avoid}
.toc a{color:var(--blue);text-decoration:none;font-weight:500;font-size:.88rem}
.toc a:hover{text-decoration:underline}
.report-section{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);
  margin:24px 64px;overflow:visible;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.section-title{display:flex;align-items:center;gap:12px;padding:16px 28px;font-size:1rem;
  font-weight:600;background:var(--blue-light);border-bottom:1px solid var(--border);color:var(--blue-dark)}
.section-icon{font-size:1.25rem;display:inline-flex;align-items:center;vertical-align:middle;margin-right:4px}
.section-body{padding:24px 20px;overflow:visible}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px;margin-bottom:22px}
.stat-card{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px 18px;text-align:center}
.stat-card .stat-value{font-size:1.8rem;font-weight:700;color:var(--blue);font-family:'IBM Plex Mono',monospace}
.stat-card .stat-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
.fig-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(580px,100%),1fr));gap:20px;margin:16px 0;width:100%;box-sizing:border-box}
.fig-card{overflow:visible;width:100%;box-sizing:border-box}
.js-plotly-plot,.plot-container{max-width:100%!important;width:100%!important}
.fig-card{border:1px solid var(--border);border-radius:10px;overflow:visible;
  box-shadow:0 2px 8px rgba(0,0,0,.06);transition:box-shadow 0.2s;margin-bottom:4px}
.fig-card:hover{box-shadow:0 4px 16px rgba(49,140,231,.15)}
.fig-card .fig-title{font-size:.82rem;font-weight:600;padding:10px 14px;background:var(--bg);
  color:var(--blue-dark);border-bottom:1px solid var(--border);border-radius:10px 10px 0 0}
.fig-card img{width:100%;display:block;cursor:zoom-in;transition:transform 0.25s;
  border-radius:0 0 10px 10px}
.fig-card img:hover{transform:scale(1.02);box-shadow:0 8px 24px rgba(0,0,0,.12)}
.js-plotly-plot .plotly .modebar{opacity:0.7!important;transition:opacity 0.2s!important}
.js-plotly-plot:hover .plotly .modebar{opacity:1!important}
.profiler-table{width:100%;border-collapse:collapse;font-size:.83rem;margin-top:12px}
.profiler-table th{background:var(--blue);color:#fff;padding:9px 13px;text-align:left;
  font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.5px}
.profiler-table td{padding:8px 13px;border-bottom:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;font-size:.81rem}
.profiler-table tr:nth-child(even) td{background:var(--bg)}
.profiler-table tr:hover td{background:var(--blue-light)}
.table-note{font-size:.78rem;color:var(--muted);margin-top:6px;font-style:italic}
.info-box{background:var(--blue-light);border-left:4px solid var(--blue);
  border-radius:0 8px 8px 0;padding:12px 16px;font-size:.88rem;color:var(--blue-dark);margin:10px 0}
.warn-box{background:#fef3c7;border-left:4px solid var(--orange);
  border-radius:0 8px 8px 0;padding:12px 16px;font-size:.88rem;margin:10px 0}
.success-box{background:#d1fae5;border-left:4px solid var(--green);
  border-radius:0 8px 8px 0;padding:12px 16px;font-size:.88rem;color:#065f46;margin:10px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.74rem;font-weight:600;margin:2px}
.badge-blue{background:var(--blue-light);color:var(--blue-dark)}
.badge-green{background:#d1fae5;color:#065f46}
.badge-orange{background:#fef3c7;color:#92400e}
.report-footer{text-align:center;padding:36px;color:var(--muted);font-size:.8rem;
  border-top:1px solid var(--border);margin-top:40px}
.report-footer a{color:var(--blue)}
.no-data-msg{text-align:center;padding:32px;color:var(--muted);font-size:.9rem}
@media(max-width:800px){.report-header{padding:28px 20px}
  .report-header h1{font-size:1.5rem}.toc,.report-section{margin:14px}
  .section-body{padding:16px}.toc ol{columns:1}.fig-grid{grid-template-columns:1fr}}
"""

    sections_html = []
    toc_items = []

    def _sec(sid: str, title: str, icon: str, content: str):
        toc_items.append(f'<li><a href="#{sid}">{icon} {title}</a></li>')
        sections_html.append(
            f'<div id="{sid}"><section class="report-section">'
            f'<h2 class="section-title"><span class="section-icon">{icon}</span>{title}</h2>'
            f'<div class="section-body">{content}</div></section></div>'
        )



    _plotlyjs_included = [False]  # mutable flag — first plotly fig loads CDN

    def _fig_html(key: str, title: str) -> str:
        entry = ss.get(f"_report_{key}")
        if entry is None:
            return ""
        kind, fig = entry

        if kind == "b64":
            return (
                f'<div class="fig-card">'
                f'<div class="fig-title">{title}</div>'
                f'<img src="data:image/png;base64,{fig}" alt="{title}" '
                f'style="width:100%;display:block;cursor:zoom-in;" '
                f'onclick="this.style.transform=this.style.transform?'':\'scale(1.8)\'">' 
                f'</div>'
            )

        if kind == "plotly":
            try:
                # First plotly fig loads the CDN script; subsequent ones reuse it
                _include = 'cdn' if not _plotlyjs_included[0] else False
                _plotlyjs_included[0] = True
                plot_html = pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs=_include,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['downloadImage', 'zoom2d', 'pan2d',
                                                  'zoomIn2d', 'zoomOut2d', 'resetScale2d'],
                        'modeBarButtonsToRemove': ['sendDataToCloud'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': title.replace(' ', '_').lower(),
                            'height': 700, 'width': 1100, 'scale': 2,
                        },
                        'scrollZoom': True,
                        'responsive': True,
                    }
                )
                return (
                    f'<div class="fig-card" style="overflow:visible;position:relative;width:100%;box-sizing:border-box;">'
                    f'<div class="fig-title" style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span>{title}</span>'
                    f'<span style="font-size:0.68rem;color:#94a3b8;font-style:italic;">'
                    f'🔍 scroll to zoom &nbsp;·&nbsp; 📥 camera icon to save</span>'
                    f'</div>'
                    f'<div style="padding:4px 8px 8px;width:100%;overflow:visible;box-sizing:border-box;">{plot_html}</div>'
                    f'</div>'
                )
            except Exception as e:
                return f'<div class="warn-box">⚠ Figure "{title}" could not be rendered: {e}</div>'

        return ""

    def _figs_section(fig_keys: list) -> str:
        """Render a grid of figures from a list of (key, label) tuples."""
        cards = "".join(_fig_html(k, lbl) for k, lbl in fig_keys if ss.get(f"_report_{k}") is not None)
        return f'<div class="fig-grid">{cards}</div>' if cards else ""

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Imported Data
    # ══════════════════════════════════════════════════════════════════════════
    data_df = ss.get('final_data')
    if data_df is None:
        data_df = ss.get('data')

    if data_df is not None:
        ns, nc = data_df.shape
        nfeat = nc - (1 if 'Class' in data_df.columns else 0)
        nclass = data_df['Class'].nunique() if 'Class' in data_df.columns else 'N/A'
        miss_pct = round(data_df.isnull().mean().mean() * 100, 2)
        zero_pct = round((data_df.select_dtypes(include='number') == 0).mean().mean() * 100, 2)

        content = (
            f'<div class="stat-grid">'
            f'<div class="stat-card"><div class="stat-value">{ns:,}</div><div class="stat-label">Samples</div></div>'
            f'<div class="stat-card"><div class="stat-value">{nfeat:,}</div><div class="stat-label">Features</div></div>'
            f'<div class="stat-card"><div class="stat-value">{nclass}</div><div class="stat-label">Classes</div></div>'
            f'<div class="stat-card"><div class="stat-value">{miss_pct}%</div><div class="stat-label">Missing values</div></div>'
            f'<div class="stat-card"><div class="stat-value">{zero_pct}%</div><div class="stat-label">Zero values</div></div>'
            f'</div>'
        )
        if 'Class' in data_df.columns:
            rows = "".join(
                f"<tr><td>{cls}</td><td>{cnt}</td><td>{cnt/ns*100:.1f}%</td></tr>"
                for cls, cnt in data_df['Class'].value_counts().items()
            )
            content += (
                f'<h3 style="font-size:.9rem;margin:16px 0 8px;color:var(--blue-dark)">Class Distribution</h3>'
                f'<table class="profiler-table"><thead><tr><th>Class</th><th>N</th><th>%</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
            )
        content += f'<h3 style="font-size:.9rem;margin:18px 0 8px;color:var(--blue-dark)">Data Preview (first 10 rows)</h3>{_df_to_html_table(data_df, 10)}'
        _sec("sec-data", "Imported Data", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="1" y="1" width="14" height="14" rx="2" stroke="#1a5fa8" stroke-width="1.3" fill="none"/><line x1="1" y1="5" x2="15" y2="5" stroke="#1a5fa8" stroke-width="1" opacity="0.5"/><line x1="5" y1="5" x2="5" y2="15" stroke="#1a5fa8" stroke-width="1" opacity="0.5"/><path d="M8 7.5v5M5.5 10.5L8 13l2.5-2.5" stroke="#1a5fa8" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg></span>""", content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Quality Control (QC)
    # ══════════════════════════════════════════════════════════════════════════
    qc_figs = [
        ("qc_class_balance",       "Class Balance"),
        ("density_intensity_per_class",     "Density Distribution of Feature Intensities per Class"),
        ("qc_class_pie",           "Class Distribution (Pie)"),
        ("qc_features_per_sample", "Features per Sample (Boxplot)"),
        ("missing_values_per_class_pies",     "Missing Values per Class (Pie)"),
        ("missing_heatmap",               "Missing Value Pattern (Heatmap)"),
        ("missing_per_class_bar",         "Top Missing Features per Class"),
        ("feature_completeness_rank",     "Feature Completeness Ranking"),
        ("cumulative_missing_curve",      "Cumulative Missing Value Curve"),
        ("zero_inflation_violin_class",   "Zero-Inflation Distribution per Class"),
        ("cumulative_feature_detection_curve", "Cumulative Feature Detection Curve"),
        ("missing_values_per_feature_class_stacked_bar",         "Missing Values per Feature"),
        ("zero_inflation_per_sample_qc_boxplot",      "Zero-Inflation per Feature"),
        ("Features per Sample Distribution Skewness",            "Feature Skewness Distribution"),
        ("Features per Sample Normality Proportion",       "Normality Proportion"),

    ]
    qc_content = _figs_section(qc_figs)
    if qc_content:
        _sec("sec-qc", "Quality Control", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="8" cy="8" r="5.5" stroke="#1a5fa8" stroke-width="1.3" fill="none"/><line x1="12" y1="12" x2="16" y2="16" stroke="#1a5fa8" stroke-width="1.5" stroke-linecap="round"/><line x1="6" y1="8" x2="10" y2="8" stroke="#1a5fa8" stroke-width="1" opacity="0.6"/><line x1="8" y1="6" x2="8" y2="10" stroke="#1a5fa8" stroke-width="1" opacity="0.6"/></svg></span>""", qc_content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Preprocessing
    # ══════════════════════════════════════════════════════════════════════════

    prep_df = ss.get('preprocessed_data')
    prep_summary = ss.get('preprocessing_summary', {})

    if prep_df is not None:
        content = ""

        # Affichage du résumé du preprocessing comme "fig-card"
        if prep_summary:
            summary_cards = ""
            for k, v in prep_summary.items():
                summary_cards += (
                    f'<div class="fig-card">'
                    f'<div class="fig-title">{k.replace("_", " ").title()}</div>'
                    f'<div class="section-body" style="padding:12px;font-size:.88rem">{v}</div>'
                    f'</div>'
                )
            content += f'<div class="fig-grid">{summary_cards}</div>'

        # Info-box pour normalization et shape
        norm = ss.get('normalization', 'N/A')
        content += (
            f'<div class="info-box">'
            f'<strong>Normalization:</strong> {norm} &nbsp;|&nbsp;'
            f'<strong>Shape after preprocessing:</strong> {prep_df.shape[0]:,} samples × {prep_df.shape[1]:,} features'
            f'</div>'
        )

        # Figures post-QC / post-preprocessing
        prep_figs = [
            ("postqc_zero",        "Zero-Inflation after Preprocessing"),
            ("postqc_samples_box", "Sample Coverage after Preprocessing"),
        ]
        content += _figs_section(prep_figs)

        # Aperçu du tableau
        content += f'<h3 style="font-size:.9rem;margin:16px 0 8px;color:var(--blue-dark)">Preview (first 8 rows)</h3>{_df_to_html_table(prep_df, 8)}'

        # Ajouter la section au rapport
        _sec("sec-prep", "  Preprocessing", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M2 4h14l-5 6v5l-4-2v-3L2 4z" stroke="#1a5fa8" stroke-width="1.3" stroke-linejoin="round" fill="none"/></svg></span>""", content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Dimensionality Reduction & Visualization
    # ══════════════════════════════════════════════════════════════════════════
    viz_figs = [
        ("fig_initial",              "Dimensionality Reduction (initial)"),
        ("fig_feature",              "Dimensionality Reduction (compressed)"),
        ("pca_fig",                  "PCA"),
        ("tsne_fig",                 "t-SNE"),
        ("umap_fig",                 "UMAP"),
        ("feature_distribution_plot","Feature Distribution"),
        ("multi_feature_plot",       "Multi-Feature Plot"),
        ("multiple_feature_plot",       "Multi-Feature Plot"),

        ("mean_spectrum_plot",       "Mean Spectrum"),
        ("individual_spectra_plot",  "Individual Spectra"),
    ]
    viz_content = _figs_section(viz_figs)
    if viz_content:
        _sec("sec-viz", "Dimensionality Reduction & Visualization", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><polyline points="2,14 5,9 8,12 11,5 14,9 16,6" stroke="#1a5fa8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/><circle cx="11" cy="5" r="2" stroke="#1a5fa8" stroke-width="1.3" fill="none"/><circle cx="11" cy="5" r="0.7" fill="#1a5fa8"/></svg></span>""", viz_content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Correlation & Similarity
    # ══════════════════════════════════════════════════════════════════════════
    corr_figs = [
        ("corr_heatmap",   "Correlation Heatmap"),
        ("sim_heatmap",    "Similarity Heatmap"),
        ("venn_diagram_plot", "Venn Diagram"),
        ("upset_plot",        "UpSet Plot"),
    ]
    corr_content = _figs_section(corr_figs)
    if corr_content:
        _sec("sec-corr", "Correlation & Similarity", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="7" cy="9" r="5" stroke="#1a5fa8" stroke-width="1.3" fill="none"/><circle cx="11" cy="9" r="5" stroke="#1a5fa8" stroke-width="1.3" fill="none"/></svg></span>""", corr_content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Statistical Analysis
    # ══════════════════════════════════════════════════════════════════════════
    # Collect boxplot pages dynamically (pagination: boxplots_fig_p0, p1, ...)
    _boxplot_pages = [(f"boxplots_fig_p{i}", f"Significant Feature Boxplots (page {i+1})")
                      for i in range(20) if ss.get(f"_report_boxplots_fig_p{i}") is not None]
    if not _boxplot_pages:
        _boxplot_pages = [("boxplots_fig", "Significant Feature Boxplots")]
    stat_figs = [
        ("volcano_fig",   "Volcano Plot"),
        ("heatmap_fig",   "Heatmap Clustering"),
    ] + _boxplot_pages
    stat_content = _figs_section(stat_figs)

    volcano_df = ss.get('volcano_data')
    if volcano_df is not None and isinstance(volcano_df, pd.DataFrame) and not volcano_df.empty:
        n_up   = (volcano_df.get('Regulation Type', pd.Series()) == 'Upregulated').sum()
        n_down = (volcano_df.get('Regulation Type', pd.Series()) == 'Downregulated').sum()
        stat_content = (
            f'<div class="stat-grid">'
            f'<div class="stat-card"><div class="stat-value">{len(volcano_df):,}</div><div class="stat-label">Significant features</div></div>'
            f'<div class="stat-card"><div class="stat-value">{n_up}</div><div class="stat-label">Upregulated</div></div>'
            f'<div class="stat-card"><div class="stat-value">{n_down}</div><div class="stat-label">Downregulated</div></div>'
            f'</div>'
        ) + stat_content
        stat_content += f'<h3 style="font-size:.9rem;margin:16px 0 8px;color:var(--blue-dark)">Significant Features Table</h3>{_df_to_html_table(volcano_df, 100)}'

    latest = ss.get('latest_result')
    if latest is not None and isinstance(latest, pd.DataFrame) and not latest.empty:
        stat_content += f'<h3 style="font-size:.9rem;margin:16px 0 8px;color:var(--blue-dark)">Statistical Results</h3>{_df_to_html_table(latest, 100)}'

    if stat_content:
        _sec("sec-stats", "Statistical Analysis", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="2" y="10" width="3" height="6" rx="1" stroke="#1a5fa8" stroke-width="1.3" fill="none"/><rect x="7" y="6" width="3" height="10" rx="1" stroke="#1a5fa8" stroke-width="1.3" fill="none"/><rect x="12" y="2" width="3" height="14" rx="1" stroke="#1a5fa8" stroke-width="1.3" fill="none"/></svg></span>""", stat_content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Machine Learning Models
    # ══════════════════════════════════════════════════════════════════════════
    models = ss.get('models', {})
    if models:
        rows = ""
        best_acc, best_name = -1.0, ""
        for mname, info in models.items():
            if isinstance(info, dict):
                acc = info.get('accuracy', info.get('test_accuracy'))
                f1  = info.get('f1', info.get('f1_score'))
                auc = info.get('auc', info.get('roc_auc'))
                try:
                    acc_f = float(acc) if acc is not None else -1.0
                except (TypeError, ValueError):
                    acc_f = -1.0
                if acc_f > best_acc:
                    best_acc, best_name = acc_f, mname
                rows += f"<tr><td>{mname}</td><td>{_fmt_val(acc)}</td><td>{_fmt_val(f1)}</td><td>{_fmt_val(auc)}</td></tr>"
            else:
                rows += f"<tr><td>{mname}</td><td colspan='3'>Trained</td></tr>"

        ml_figs = [
            ("ml_confusion",       "Confusion Matrix"),
            ("ml_confusion_norm",  "Normalized Confusion Matrix (%)"),
            ("ml_roc",             "ROC Curves"),
            ("ml_learning_curve",  "Learning Curve"),
            ("model_comparison_fig","Model Comparison"),
            ("model_comparison_rmse_fig", "Model Comparison — RMSE"),
            ("ml_comparison",      "Model Comparison (Classification)"),
            ("dl_comparison",      "Deep Learning Comparison"),
            ("lime_feature_contrib",           "LIME Feature Importance"),
            ("pred_vs_true_fig",   "Predicted vs True"),
            ("residual_plot_fig",  "Residual Plot"),
            ("shap_beeswarm", "SHAP Beeswarm Plot"),
            ("shap_bar",      "SHAP Bar Plot")
        ]
        content = (
            f'<table class="profiler-table"><thead>'
            f'<tr><th>Model</th><th>Accuracy</th><th>F1</th><th>AUC</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )
        if best_name and best_acc > 0:
            content += f'<div class="success-box" style="margin-top:14px">🏆 Best model: <strong>{best_name}</strong> — Accuracy {best_acc:.4f}</div>'
        content += _figs_section(ml_figs)
        _sec("sec-ml", "Machine Learning Models", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="3" cy="6" r="1.5" stroke="#1a5fa8" stroke-width="1.2" fill="none"/><circle cx="3" cy="12" r="1.5" stroke="#1a5fa8" stroke-width="1.2" fill="none"/><circle cx="9" cy="4" r="1.5" stroke="#1a5fa8" stroke-width="1.2" fill="none"/><circle cx="9" cy="9" r="1.5" stroke="#1a5fa8" stroke-width="1.2" fill="none"/><circle cx="9" cy="14" r="1.5" stroke="#1a5fa8" stroke-width="1.2" fill="none"/><circle cx="15" cy="9" r="1.5" stroke="#1a5fa8" stroke-width="1.2" fill="none"/><line x1="4.5" y1="6" x2="7.5" y2="4" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/><line x1="4.5" y1="6" x2="7.5" y2="9" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/><line x1="4.5" y1="12" x2="7.5" y2="9" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/><line x1="4.5" y1="12" x2="7.5" y2="14" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/><line x1="10.5" y1="4" x2="13.5" y2="9" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/><line x1="10.5" y1="9" x2="13.5" y2="9" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/><line x1="10.5" y1="14" x2="13.5" y2="9" stroke="#1a5fa8" stroke-width="0.9" opacity="0.7"/></svg></span>""", content)



        # ══════════════════════════════════════════════════════════════════════════
        # SECTION 8 — Survival Analysis
        # ══════════════════════════════════════════════════════════════════════════
        surv_figs = [
            ("km_fig",  "Kaplan-Meier Survival Curves"),
            ("cox_fig", "Cox Model Forest Plot"),
        ]
        surv_content = _figs_section(surv_figs)
        if surv_content:
            _sec("sec-surv", "Survival Analysis", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M2 4h3v4h3v4h3v4h5" stroke="#1a5fa8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/><line x1="2" y1="16" x2="16" y2="16" stroke="#1a5fa8" stroke-width="1" opacity="0.3"/><line x1="2" y1="4" x2="2" y2="16" stroke="#1a5fa8" stroke-width="1" opacity="0.3"/></svg></span>""", surv_content)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Pathway & Enrichment
    # ══════════════════════════════════════════════════════════════════════════
    enrich_figs = [
        ("enrichment_bar",              "Enrichment Bar Chart"),
        ("enrichment_dot",              "Dot Plot (clusterProfiler style)"),
        ("enrichment_heatmap",          "Pathway Enrichment Heatmap"),
        ("enrichment_gene_count",       "Gene Count per Pathway"),
        ("enrichment_gene_pathway",     "Gene × Pathway Heatmap"),
        ("enrichment_network",          "Gene Co-Pathway Network"),
    ]
    enrich_content = ""
    # Gene table (DataFrame stored separately)
    _gene_table_df = ss.get("enrichment_gene_table")
    if _gene_table_df is not None:
        enrich_content += (
            "<h3 style='font-size:.9rem;margin:0 0 10px;color:var(--blue-dark)'>Gene Involvement per Pathway & Class</h3>"
            + _df_to_html_table(_gene_table_df, max_rows=500)
            + "<br>"
        )
    enrich_content += _figs_section(enrich_figs)
    if enrich_content.strip():
        _sec("sec-enrich", "Pathway & Enrichment Analysis", """<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M5 2c0 4 8 3 8 7s-8 3-8 7" stroke="#1a5fa8" stroke-width="1.5" stroke-linecap="round" fill="none"/><path d="M13 2c0 4-8 3-8 7s8 3 8 7" stroke="#1a5fa8" stroke-width="1.5" stroke-linecap="round" fill="none" opacity="0.5"/><line x1="5.5" y1="6" x2="12.5" y2="6" stroke="#1a5fa8" stroke-width="1" opacity="0.6"/><line x1="5.5" y1="12" x2="12.5" y2="12" stroke="#1a5fa8" stroke-width="1" opacity="0.6"/></svg></span>""", enrich_content)

    # ── fallback if nothing was generated ─────────────────────────────────────
    if not sections_html:
        sections_html.append(
            '<section class="report-section"><h2 class="section-title">'
            '<span class="section-icon" style="display:inline-flex;align-items:center;vertical-align:middle;"><svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 2L1 16h16L9 2z" stroke="#1a5fa8" stroke-width="1.3" stroke-linejoin="round" fill="none"/><line x1="9" y1="8" x2="9" y2="12" stroke="#1a5fa8" stroke-width="1.5" stroke-linecap="round"/><circle cx="9" cy="14.5" r="0.8" fill="#1a5fa8"/></svg></span>No analysis available</h2>'
            '<div class="section-body"><div class="warn-box">'
            'No data or results found in the current session. '
            'Please import data and run analyses before generating the report.'
            '</div></div></section>'
        )

    toc_html = (
        '<nav class="toc"><h3>Table of Contents</h3><ol>'
        + (''.join(toc_items) if toc_items else '<li>No sections available</li>')
        + '</ol></nav>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Profiler — Analysis Report — {now}</title>
<style>{css}</style>
</head>
<body>
<header class="report-header">
  <h1>Profiler Analysis Report</h1>
  <p class="subtitle">Open multi-omics analysis platform — PRISM U1192 / INSERM</p>
  <div class="meta-bar">
    <span>📅 Generated: {now}</span>
    <span>🔢 Profiler v1.2</span>

  </div>
</header>
{toc_html}
{''.join(sections_html)}
<footer class="report-footer">
  <p>Report automatically generated by <strong>Profiler v1.2</strong> — PRISM U1192 · INSERM</p>
  <p style="margin-top:6px"><em>Bioinformatics</em>, 2025 —
  <a href="https://doi.org/10.1093/bioinformatics/btaf644">DOI: 10.1093/bioinformatics/btaf644</a></p>
</footer>
</body>
</html>"""


def render_run_all_button():
    """Sidebar button to generate and download the HTML report."""
    with st.sidebar.expander("📊 Export HTML Report", expanded=False):
        st.markdown(
            '<p style="font-size:12px;color:#555">Generate a full HTML report of all analyses in the current session.</p>',
            unsafe_allow_html=True
        )
        if st.button("📄 Generate HTML Report", key="run_all_report_btn", use_container_width=True):
            with st.spinner("Building report…"):
                html_content = generate_html_report()
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_content.encode("utf-8"),
                file_name=f"profiler_report_{now_str}.html",
                mime="text/html",
                key="download_report_btn",
                use_container_width=True
            )
            st.success("✅ Report ready!")



def main():
    initialize_session_state()
    # ── Desktop mode: no cleanup thread, no auth ─────────────────────────────

    # working directory to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # assets/ lives at app/assets/, two levels up from app/gui/
    _assets_dir = os.path.normpath(os.path.join(script_dir, "..", "assets"))
    os.chdir(script_dir)

    # ── Profiler Icons — load custom font from file at runtime ────────────────
    st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    white-space: normal !important;
    word-break: break-word !important;
    max-width: 180px !important;
    font-size: 0.78rem !important;
    padding: 6px 8px !important;
    text-align: center !important;
    height: auto !important;
    min-height: 36px !important;
}
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    gap: 4px !important;
}
</style>
""", unsafe_allow_html=True)
    try:
        _fp = os.path.join(_assets_dir, "profiler_icons.ttf") if os.path.exists(os.path.join(_assets_dir, "profiler_icons.ttf")) else os.path.join(script_dir, "profiler_icons.ttf")
        if os.path.exists(_fp):
            with open(_fp, "rb") as _ff:
                _fb = base64.b64encode(_ff.read()).decode()
            st.markdown(f'''<style>
@font-face {{
    font-family: "PI";
    src: url("data:font/truetype;base64,{_fb}") format("truetype");
    font-display: swap;
    unicode-range: U+E000-E0FF;
}}
button[data-baseweb="tab"],
button[data-baseweb="tab"] span,
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] * {{
    font-family: "PI", "Segoe UI", system-ui, sans-serif !important;
    color: #1e293b !important;
    font-size: 0.90rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] span,
button[data-baseweb="tab"][aria-selected="true"] * {{
    color: #ffffff !important;
}}
[data-testid="stSidebar"] .stExpander summary,
[data-testid="stSidebar"] .stExpander summary * {{
    font-family: "PI", "Segoe UI", system-ui, sans-serif !important;
}}
</style>''', unsafe_allow_html=True)
    except Exception:
        pass

    # ── Sidebar logo: profiler_logo.png (same folder), then logo/Log2.png, then text ──
    _logo_loaded = False
    for _logo_path in [
        os.path.join(_assets_dir, "profiler_logo.png"),
        os.path.join(script_dir, "profiler_logo.png"),
        os.path.join(script_dir, "logo", "Log2.png"),
        os.path.join(script_dir, "logo", "profiler_logo.png"),
    ]:
        if not os.path.exists(_logo_path):
            continue
        try:
            with open(_logo_path, "rb") as _lf:
                _lb = base64.b64encode(_lf.read()).decode()
            _ext = os.path.splitext(_logo_path)[1].lower().lstrip(".")
            _mime = "image/png" if _ext == "png" else "image/jpeg"
            st.sidebar.markdown(
                f'''<div style="background:#fff;border-radius:12px;padding:10px 6px 8px;
                    margin-bottom:10px;text-align:center;border:1px solid #c5d5e8;
                    box-shadow:0 2px 8px rgba(49,140,231,0.1);">
                    <img src="data:{_mime};base64,{_lb}"
                         style="width:100%;max-width:220px;height:auto;
                                object-fit:contain;display:block;margin:0 auto;"/>
                </div>''',
                unsafe_allow_html=True
            )
            _logo_loaded = True
            break
        except Exception:
            continue
    if not _logo_loaded:
        st.sidebar.markdown(
            '''<div style="text-align:center;padding:12px;background:#fff;border-radius:12px;
                margin-bottom:10px;border:1px solid #c5d5e8;">
                <span style="font-size:1.8rem;">🔬</span>
                <div style="color:#1e3a5f;font-weight:800;font-size:1rem;margin-top:4px;">Profiler</div>
            </div>''',
            unsafe_allow_html=True
        )

    # ── Desktop greeting banner (replaces login/session block) ───────────────
    hour = datetime.datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.sidebar.markdown(
        f"""<div style='background:linear-gradient(135deg,rgba(49,140,231,0.14),rgba(49,140,231,0.04));
            border-left:3px solid #318CE7;border-radius:10px;padding:10px 14px;
            margin-bottom:8px;'>
            <div style='color:#1a4a80;font-weight:800;font-size:0.88rem;'>
                {greeting} . Desktop Mode
            </div>
            <div style='color:#4a7ab5;font-size:0.72rem;margin-top:3px;'>
                <span style='background:#dbeafe;color:#1e40af;font-size:0.65rem;font-weight:700;
                padding:2px 7px;border-radius:12px;'>● Local · Offline · No login</span>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    # ── Desktop session controls (Reset / Stop — no Logout) ──────────────────
    with st.sidebar.expander("  Session & Controls", expanded=True):
        st.markdown(
            "<div style='font-size:0.78rem;color:#64748b;margin-bottom:10px;'>"
            "Manage your working session below.</div>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Reset", key="desktop_reset_btn", help="Clear all analysis data",
                         use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                gc.collect()
                st.rerun()
        with c2:
            if st.button("⏹ Stop", key="global_stop_button", help="Stop current analysis",
                         use_container_width=True):
                st.session_state['stop_analysis'] = True
                st.warning("⚠️ Analysis stopped.")

        # ═══════════════════════════════════════════════════════════
        #  SIDEBAR — Data Import & Pipeline
        # ═══════════════════════════════════════════════════════════

        # ── 0. Quick Help & PDF guide ───────────────────────────────
        with st.sidebar.expander("❓ Help & Import Guide", expanded=False):
            st.markdown(
                "<div style='font-size:0.80rem;line-height:1.7;color:#334155;'>"
                "<b>Desktop mode — no size limits.</b><br>"
                "&nbsp;&nbsp;• Any file size &nbsp;·&nbsp; Any number of features<br><br>"
                "<b>Privacy:</b> All data stays on your machine — <b>never uploaded</b>."
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")
            # ── PDF Data Import Guide ──────────────────────────────
            try:
                _docs_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "docs"))
                _pdf_path = next(
                    (_p for _p in [
                        os.path.join(_docs_dir, "profiler_data_import_guide.pdf"),
                        os.path.join(_docs_dir, "Profiler_DataImportGuide.pdf"),
                        os.path.join(script_dir, "profiler_data_import_guide.pdf"),
                        os.path.join(script_dir, "Profiler_DataImportGuide.pdf"),
                        "profiler_data_import_guide.pdf",
                    ] if os.path.exists(_p)), None
                )
                if _pdf_path:
                    with open(_pdf_path, "rb") as _pf:
                        st.download_button(
                            label="📄 Download Full Import Guide (PDF)",
                            data=_pf.read(),
                            file_name="Profiler_DataImportGuide.pdf",
                            mime="application/pdf",
                            key="dl_import_guide",
                            use_container_width=True,
                            help="Detailed guide: column names, supported formats, examples"
                        )
            except Exception:
                pass
            # ── Example CSV ────────────────────────────────────────
            try:
                _csv_path = next(
                    (_p for _p in [
                        _os.path.join(_os.path.dirname(__file__), "example_proteomics_3class.csv"),
                        "example_proteomics_3class.csv",
                    ] if _os.path.exists(_p)), None
                )
                if _csv_path:
                    with open(_csv_path, "rb") as _cf:
                        st.download_button(
                            label="📊 Download Example Dataset",
                            data=_cf.read(),
                            file_name="example_proteomics_3class.csv",
                            mime="text/csv",
                            key="dl_example_csv",
                            use_container_width=True,
                        )
                    st.caption("45 samples · Cancer/Benign/Healthy · treatment_meta · age_meta · 202 proteins")
            except Exception:
                pass
            st.markdown("---")
            st.markdown("""
<div style='font-size:0.75rem;color:#475569;line-height:1.65;'>
<b>Column naming (auto-detected):</b><br>
&nbsp;• <b>Target:</b> <code>Class</code> / <code>Target</code> / <code>Condition</code> / <code>Label</code> / <code>Group</code><br>
&nbsp;• <b>ID:</b> <code>ID</code> / <code>SampleID</code> / <code>Name</code> / <code>Patient</code><br>
&nbsp;• <b>Metadata:</b> any column ending in <code>_meta</code><br><br>
<b>Supported specialised formats:</b><br>
&nbsp;Proteomics: MaxQuant · DIA-NN · Spectronaut · FragPipe · PD · Progenesis · PEAKS · Perseus<br>
&nbsp;Transcriptomics: DESeq2 · Salmon · kallisto · featureCounts · STAR · HTSeq<br>
&nbsp;Metabolomics: MetaboAnalyst · XCMS · MZmine<br>
&nbsp;Spectra: mzML · mzXML (MS1 tab below)
</div>""", unsafe_allow_html=True)

        st.sidebar.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════
        #  SECTION 1 — RAW CONVERSION
        # ═══════════════════════════════════════════════════════════
        with st.sidebar.expander("⚙️ Step 1 — RAW File Conversion", expanded=False):
            st.caption("Convert Waters/Thermo/Bruker RAW files to mzML/mzXML before loading.")
            uploaded_raw = st.file_uploader(
                "Upload RAW files or ZIP archive",
                type=["raw", "RAW", "zip", "ZIP", "d"],
                accept_multiple_files=True,
                key="uploaded_raw_files",
                help="Waters .raw, Thermo .raw, Bruker .d, or a ZIP containing them."
            )
            col_ft, col_of = st.columns(2)
            with col_ft:
                file_type = st.selectbox("Instrument", ["waters", "thermo", "bruker"], key="file_type")
            with col_of:
                output_format = st.selectbox("Output", ["mzML", "mzXML", "mz5", "mzDB"], key="output_format")

            col_pp, col_lm = st.columns(2)
            with col_pp:
                peak_picking = st.checkbox("Peak Picking", key="peak_picking")
            with col_lm:
                lock_mass = st.text_input("Lock Mass", placeholder="e.g. 554", key="lock_mass") if file_type == "waters" else None

            mass_range = st.text_input("Mass range", placeholder="e.g. [600,1000]", key="mass_range")

            if st.button("🚀 Convert", key="convert_raw", use_container_width=True):
                with st.spinner("⏳ Converting RAW files…"):
                    project_root = os.path.dirname(os.path.abspath(__file__))
                    raw_dir    = os.path.join(project_root, "temp_files", "raw_input")
                    output_dir = os.path.join(project_root, "temp_files", "converted")
                    os.makedirs(raw_dir, exist_ok=True)
                    os.makedirs(output_dir, exist_ok=True)
                    for folder in [raw_dir, output_dir]:
                        for f in os.listdir(folder):
                            fp = os.path.join(folder, f)
                            (shutil.rmtree if os.path.isdir(fp) else os.remove)(fp)
                    try:
                        for uploaded_file in (uploaded_raw or []):
                            fn = uploaded_file.name
                            sp_ = os.path.join(raw_dir, fn)
                            with open(sp_, "wb") as f:
                                f.write(uploaded_file.read())
                            if fn.lower().endswith(".zip"):
                                with zipfile.ZipFile(sp_, "r") as z:
                                    z.extractall(raw_dir)
                                os.remove(sp_)
                        convert_raw_to_mzml(
                            raw_dir, output_dir, file_type,
                            eval(mass_range) if mass_range else None,
                            peak_picking,
                            float(lock_mass) if lock_mass else None,
                            output_format
                        )
                        st.success(f"✅ {len(os.listdir(output_dir))} file(s) converted.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                    finally:
                        gc.collect()

            if os.path.exists(output_dir if 'output_dir' in dir() else '') and os.listdir(output_dir):
                st.markdown("**📂 Converted files:**")
                for filename in os.listdir(output_dir):
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "rb") as f:
                        st.download_button(f"📥 {filename}", f, filename,
                            mime="application/octet-stream", key=f"dl_conv_{filename}")
                zip_path = os.path.join(script_dir, "temp_files", "converted_files.zip")
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for fn in os.listdir(output_dir):
                        zipf.write(os.path.join(output_dir, fn), arcname=fn)
                with open(zip_path, "rb") as f:
                    st.download_button("📦 Download all as ZIP", f, "converted_files.zip",
                        mime="application/zip", key="download_all_zip", use_container_width=True)
                for fn in os.listdir(output_dir):
                    try: os.remove(os.path.join(output_dir, fn))
                    except Exception: pass
                try: os.remove(zip_path)
                except Exception: pass
                gc.collect()



        with st.sidebar.expander("📡 Step 2 — Load MS1 Spectra (mzML / mzXML)", expanded=False):
            st.caption("Upload mzML/mzXML files grouped by class. Each group = one biological class.")
            for i, group in enumerate(st.session_state["file_groups"]):
                with st.container():
                    c_name, c_del = st.columns([4, 1])
                    with c_name:
                        st.session_state["file_groups"][i]["class_name"] = st.text_input(
                            f"Class {i+1} name", value=group["class_name"],
                            key=f"class_name_{i}", placeholder="e.g. Tumor, Control…",
                            label_visibility="collapsed"
                        )
                    with c_del:
                        if st.button("✕", key=f"del_group_{i}", help="Remove this class"):
                            st.session_state["file_groups"].pop(i)
                            st.rerun()
                    uploaded = st.file_uploader(
                        f"Files for {st.session_state['file_groups'][i]['class_name'] or f'Class {i+1}'}",
                        accept_multiple_files=True, type=["mzML", "mzXML"],
                        key=f"files_{i}", label_visibility="collapsed"
                    )
                    st.session_state["file_groups"][i]["files"] = uploaded
                    del uploaded
                if i < len(st.session_state["file_groups"]) - 1:
                    st.markdown("<hr style='margin:4px 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)

            if st.button("➕ Add class", key="add_group_btn", use_container_width=True):
                add_group()

            st.markdown("<hr style='margin:6px 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)
            st.markdown("**⚙️ Peak detection parameters**")

            col_noise, col_width = st.columns(2)
            with col_noise:
                min_apex_intensity_pct = st.number_input(
                    "Noise floor (%)", min_value=0.0, max_value=20.0, value=1.0,
                    step=0.5, key="min_apex_intensity_pct",
                    help=(
                        "Minimum apex intensity as % of global TIC max. "
                        "Acts as a noise floor: only peaks taller than this are kept. "
                        "1% = keep all real peaks; raise to ignore minor peaks."
                    )
                )
            with col_width:
                min_peak_width = st.number_input(
                    "Min width (scans)", min_value=1, max_value=20, value=1,
                    key="min_peak_width",
                    help="Minimum consecutive scans to form a valid peak."
                )

            col_dist, col_prom = st.columns(2)
            with col_dist:
                _dist_auto = st.session_state.get("mzml_auto_distance", 0)
                _dist_label = f"Min dist (auto={_dist_auto})" if _dist_auto else "Min dist (scans)"
                min_peak_distance = st.number_input(
                    _dist_label, min_value=0, max_value=200, value=0,
                    key="min_peak_distance",
                    help=(
                        "Minimum scans between two peak apices. "
                        "0 = auto-detected from valley spacing of the TIC — recommended. "
                        "Raise only if peaks are being merged incorrectly."
                    )
                )
            with col_prom:
                min_prominence_pct = st.number_input(
                    "Min prominence (%)", min_value=0.0, max_value=50.0, value=5.0,
                    step=1.0, key="min_prominence_pct",
                    help="Peak prominence as % of global TIC max. Filters noise and shoulders."
                )

            st.markdown("**m/z binning**")

            # ── Binning mode toggle ──────────────────────────────────────────────────
            _grid_mode = st.toggle(
                "Fixed grid (uniform bins across the full m/z range)",
                value=True, key="mz_fixed_grid_mode",
                help=(
                    "Fixed-grid mode: same chromatographic peak detection pipeline, "
                    "but each peak is projected onto a uniform m/z grid. "
                    "Number of features = (mz_max - mz_min) / bin_Da, computed automatically. "
                    "Example: 600–1000 Da with 0.1 Da → exactly 4,000 features per peak.\n\n"
                    "✅ Recommended for machine learning: guarantees identical feature vectors "
                    "across all samples and files — required for most ML algorithms "
                    "(PCA, Random Forest, SVM, neural networks, etc.).\n\n"
                    "⚠️ Adaptive mode (toggle OFF): features are aligned by consensus m/z across "
                    "files, which preserves natural peak positions. Better suited for peak "
                    "annotation, metabolite identification, and targeted analysis, but may "
                    "produce sparse or misaligned matrices across heterogeneous datasets."
                )
            )

            col_ppm, col_da = st.columns(2)
            with col_ppm:
                mz_bin_ppm = st.number_input(
                    "Bin (ppm)", min_value=0.0, max_value=500.0,
                    value=0.0 if _grid_mode else 0.0,
                    step=10.0, key="mz_bin_ppm",
                    help="ppm tolerance. Mettre 0 en mode grille fixe.",
                    disabled=_grid_mode
                )
            with col_da:
                mz_bin_da = st.number_input(
                    "Bin (Da)", min_value=0.001, max_value=2.0, value=0.1,
                    step=0.01, format="%.3f", key="mz_bin_da",
                    help="Pas de bin en Da. En mode grille fixe, définit la résolution de la grille."
                )

            if _grid_mode:
                col_mzmin, col_mzmax = st.columns(2)
                with col_mzmin:
                    mz_grid_min = st.number_input(
                        "m/z min", min_value=0.0, max_value=10000.0, value=0.0,
                        step=50.0, key="mz_grid_min",
                        help="Borne basse de la grille. 0 = détecte automatiquement depuis les données."
                    )
                with col_mzmax:
                    mz_grid_max = st.number_input(
                        "m/z max", min_value=0.0, max_value=10000.0, value=0.0,
                        step=50.0, key="mz_grid_max",
                        help="Borne haute de la grille. 0 = détecte automatiquement depuis les données."
                    )
                _mz_min_arg = float(mz_grid_min) if mz_grid_min > 0 else None
                _mz_max_arg = float(mz_grid_max) if mz_grid_max > 0 else None
                if _mz_min_arg and _mz_max_arg:
                    _n_bins_preview = round((_mz_max_arg - _mz_min_arg) / mz_bin_da)
                    st.caption(
                        f"Grille fixe : [{_mz_min_arg:.0f}, {_mz_max_arg:.0f}] Da "
                        f"/ {mz_bin_da:.3f} Da = **{_n_bins_preview:,} features** par peak chromatographique"
                    )
                else:
                    st.caption(f"Grille fixe : plage auto / {mz_bin_da:.3f} Da")
            else:
                _mz_min_arg = None
                _mz_max_arg = None
                if mz_bin_ppm <= 20 and mz_bin_da == 0:
                    st.caption("ℹ️ Fine-grained · Orbitrap / QTOF")
                elif mz_bin_da > 0:
                    st.caption(f"ℹ️ Da mode · {mz_bin_da:.3f} Da")
                elif mz_bin_ppm <= 100:
                    st.caption("ℹ️ Balanced · TOF")
                else:
                    st.caption("ℹ️ Aggressive binning")

            if st.button(" Load Spectra", key="load_spectra_btn",
                        use_container_width=True, type="primary"):
                if any(group.get("files") for group in st.session_state["file_groups"]):
                    with st.spinner("🔄 Loading and processing MS files…"):
                        progress_bar = st.progress(0)
                        st.session_state["data"] = load_uploaded_files(
                            st.session_state["file_groups"],
                            progress_bar,
                            min_apex_intensity_pct = float(min_apex_intensity_pct),
                            tol_ppm                = float(mz_bin_ppm),
                            tol_da                 = float(mz_bin_da),
                            min_peak_width         = int(min_peak_width),
                            min_distance           = int(min_peak_distance),
                            min_prominence_pct     = float(min_prominence_pct),
                            use_fixed_grid         = bool(_grid_mode),
                            mz_min                 = _mz_min_arg,
                            mz_max                 = _mz_max_arg,
                        )
                        # Expose auto-detected distance for sidebar hint
                        _chrom = st.session_state.get("mzml_chrom_data", [])
                        _dists = [c.get("auto_distance", 0) for c in _chrom if c.get("auto_distance", 0) > 0]
                        if _dists:
                            import numpy as _np
                            st.session_state["mzml_auto_distance"] = int(_np.median(_dists))
                        # Desktop: data kept in session_state (no disk persistence)
                        del progress_bar
                        for group in st.session_state["file_groups"]:
                            group.pop("files", None)
                        gc.collect()
                        # Auto-load into Data Lab
                        _raw_ml = st.session_state.get("data")
                        if _raw_ml is not None:
                            st.session_state["datalab_df"]      = _raw_ml.copy()
                            st.session_state["datalab_source"]  = "Raw / Edited"
                            st.session_state["overview_df"]     = _raw_ml.copy()
                            st.session_state["overview_source"] = "Raw / Edited"
                else:
                    st.error("Please select at least one mzML/mzXML file.")


        # ═══════════════════════════════════════════════════════════
        #  SECTION 3 — TABULAR / OMICS DATA
        # ═══════════════════════════════════════════════════════════
        if "expand_load_data" not in st.session_state:
            st.session_state.expand_load_data = False

        with st.sidebar.expander(
            "📂 Step 3 — Load Tabular / Omics Data",
            expanded=st.session_state.expand_load_data
        ):
            # Format summary card
            st.markdown("""
<div style='font-size:0.76rem;background:#f8fafc;border:1px solid #e2e8f0;
border-radius:8px;padding:10px 12px;margin-bottom:10px;line-height:1.8;'>
<div style='color:#0891b2;font-weight:700;margin-bottom:4px;'>Auto-detect identifies your format automatically</div>
<table style='width:100%;border-collapse:collapse;font-size:0.74rem;'>
<tr>
  <td style='color:#475569;padding-right:8px;vertical-align:top;white-space:nowrap;'><b> Generic CSV/TSV</b></td>
  <td style='color:#64748b;'>One row = one sample · <code>Class</code>/<code>Target</code>/<code>Condition</code> column required · <code>ID</code> column optional · columns ending in <code>_meta</code> = clinical metadata</td>
</tr>
<tr>
  <td style='color:#475569;padding-right:8px;vertical-align:top;white-space:nowrap;'><b> Proteomics</b></td>
  <td style='color:#64748b;'>MaxQuant · DIA-NN · Spectronaut · FragPipe · Proteome Discoverer · Progenesis · PEAKS · Perseus (protein & peptide level)</td>
</tr>
<tr>
  <td style='color:#475569;padding-right:8px;vertical-align:top;white-space:nowrap;'><b> RNA-seq</b></td>
  <td style='color:#64748b;'>DESeq2/edgeR counts · Salmon · kallisto · featureCounts · STAR · HTSeq</td>
</tr>
<tr>
  <td style='color:#475569;padding-right:8px;vertical-align:top;white-space:nowrap;'><b> Metabolomics</b></td>
  <td style='color:#64748b;'>MetaboAnalyst · XCMS · MZmine feature tables</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload dataset (CSV, TSV, TXT, XLSX)",
                type=["csv", "xlsx", "txt", "tsv"],
                key="uploaded_file",
                help=(
                    "Auto-detected delimiter (,  ;  tab  |) and encoding. "
                    "Class/Target/Condition → target column. "
                    "Columns ending with _meta → clinical metadata."
                )
            )
            if uploaded_file is not None:
                st.session_state.expand_load_data = True
                _file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("_last_tabular_file_id") != _file_id:
                    _load_result = load_structured_data(uploaded_file)
                    df = _load_result[0] if isinstance(_load_result, tuple) else _load_result
                    if df is not None:
                        has_class  = "Class" in df.columns
                        _meta_cols = [c for c in df.columns if str(c).endswith("_meta")]
                        if has_class or _meta_cols:
                            finalize_data_load(df, "Generic")
                            st.session_state["_last_tabular_file_id"] = _file_id
                            # Auto-load into Data Lab
                            st.session_state["datalab_df"]      = df.copy()
                            st.session_state["datalab_source"]  = "Raw / Edited"
                            st.session_state["overview_df"]     = df.copy()
                            st.session_state["overview_source"] = "Raw / Edited"
                            if "ID" in df.columns:
                                st.info(f"🪪 ID column detected ({df['ID'].nunique()} unique values)")
                            if _meta_cols:
                                st.info(f"🏷️ Metadata columns: {', '.join(_meta_cols)}")
                        else:
                            st.warning("⚠️ No class/target column detected. Choose format or pick a column.")
                            _auto_fmt = detect_omics_format(uploaded_file)
                            if _auto_fmt:
                                st.info(f"🔍 Detected format: **{_auto_fmt}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                _fmt_options = [
                                    "Auto-detect",
                                    "── Proteomics (protein) ──",
                                    "MaxQuant proteins", "DIA-NN proteins",
                                    "Spectronaut proteins", "FragPipe / MSFragger",
                                    "Proteome Discoverer", "Progenesis QI",
                                    "PEAKS Studio", "Perseus",
                                    "── Proteomics (peptide) ──",
                                    "MaxQuant peptides", "DIA-NN peptides",
                                    "Spectronaut peptides",
                                    "── Transcriptomics ──",
                                    "RNA-seq count matrix", "Salmon / kallisto",
                                    "featureCounts", "STAR counts", "HTSeq counts",
                                    "── Metabolomics ──",
                                    "MetaboAnalyst", "XCMS / MZmine",
                                    "── Generic ──", "Pick column manually",
                                ]
                                file_type = st.selectbox(
                                    "File format",
                                    _fmt_options,
                                    index=_fmt_options.index(_auto_fmt) if _auto_fmt in _fmt_options else 0,
                                    key="ft_sel"
                                )
                            with col2:
                                class_col_choice = st.selectbox(
                                    "Or pick target column",
                                    ["— none —"] + list(df.columns),
                                    key="class_col_pick"
                                )
                            if class_col_choice != "— none —":
                                df = df.rename(columns={class_col_choice: "Class"})
                                finalize_data_load(df, "Generic (manual class)")
                                st.session_state["_last_tabular_file_id"] = _file_id
                            elif file_type == "Auto-detect" and _auto_fmt:
                                with st.spinner(f"Auto-loading as {_auto_fmt}…"):
                                    try:
                                        df2, lbl = load_omics_auto(uploaded_file)
                                        finalize_data_load(df2, lbl)
                                        st.session_state["_last_tabular_file_id"] = _file_id
                                    except Exception as _e:
                                        st.error(f"Auto-load failed: {_e}")
                            elif not file_type.startswith("──"):
                                _parser_map = {
                                    "MaxQuant proteins":    lambda f: maxquant_data(f),
                                    "DIA-NN proteins":      lambda f: diann_data(f),
                                    "Spectronaut proteins": lambda f: spectronaut_protein_data(f),
                                    "FragPipe / MSFragger": lambda f: fragpipe_data(f),
                                    "Proteome Discoverer":  lambda f: proteome_discoverer_data(f),
                                    "Progenesis QI":        lambda f: progenesis_data(f),
                                    "PEAKS Studio":         lambda f: peaks_data(f),
                                    "MaxQuant peptides":    lambda f: maxquant_peptide_data(f),
                                    "DIA-NN peptides":      lambda f: diann_peptide_data(f),
                                    "Spectronaut peptides": lambda f: spectronaut_peptide_data(f),
                                    "RNA-seq count matrix": lambda f: rnaseq_counts_data(f),
                                    "Salmon / kallisto":    lambda f: salmon_kallisto_data(f),
                                    "featureCounts":        lambda f: featurecounts_data(f),
                                    "STAR counts":          lambda f: star_counts_data(f),
                                    "HTSeq counts":         lambda f: htseq_counts_data(f),
                                    "MetaboAnalyst":        lambda f: metaboanalyst_data(f),
                                    "XCMS / MZmine":        lambda f: xcms_mzmine_data(f),
                                }
                                if file_type == "Perseus":
                                    feat_opts = ["Choose an option", "T: Gene names", "T: Protein names"]
                                    sel = st.selectbox("Feature names row:", feat_opts,
                                        index=feat_opts.index(st.session_state.get("selected_feature_row", "Choose an option")))
                                    st.session_state["selected_feature_row"] = sel
                                    if sel != "Choose an option":
                                        with st.spinner("Processing Perseus…"):
                                            df2 = perseus_data(uploaded_file, feature_row_index=-2 if sel == "T: Gene names" else -1)
                                            finalize_data_load(df2, "Perseus")
                                            st.session_state["_last_tabular_file_id"] = _file_id
                                elif file_type in _parser_map:
                                    with st.spinner(f"Processing {file_type}…"):
                                        try:
                                            df2 = _parser_map[file_type](uploaded_file)
                                            finalize_data_load(df2, file_type)
                                            st.session_state["_last_tabular_file_id"] = _file_id
                                        except Exception as _pe:
                                            st.error(f"Error: {_pe}")

            # ── Inline quick-reference (checkbox, no nested expander) ────
            if st.checkbox("📋 Show format reference", key="show_fmt_ref"):
                st.markdown("""
| Format | Key columns |
|--------|-------------|
| Generic CSV | `Class` / `Target`, features |
| MaxQuant | `LFQ intensity *` cols |
| DIA-NN | `Protein.Group`, `Genes` + sample paths |
| Spectronaut | `PG.Genes` + sample cols |
| FragPipe | `Gene` + `* MaxLFQ Intensity` |
| DESeq2/edgeR | `gene_name`/`gene_id` + sample cols |
| Salmon | `Name` + `TPM` cols |
| featureCounts | `Geneid` + `Chr` + sample cols |
| MetaboAnalyst | `Sample`, `Class`, metabolite cols |
| XCMS/MZmine | `row m/z`, `row retention time` |
""")

        # ═══════════════════════════════════════════════════════════
        #  SECTION 4 — SURVIVAL DATA
        # ═══════════════════════════════════════════════════════════
        with st.sidebar.expander("⏳ Step 4 — Load Survival Data", expanded=False):
            st.caption("Kaplan-Meier & Cox regression. Requires survival time + event status columns.")
            import pandas as pd
            uploaded_file_surv = st.file_uploader(
                "Upload survival dataset (CSV, XLSX, TXT)",
                type=["csv", "xlsx", "txt"],
                key="uploaded_file_survival_side",
                help="Needs: Overall survival (numeric), State (0/1 = censored/event), Class or covariates."
            )
            if uploaded_file_surv is not None:
                try:
                    if uploaded_file_surv.name.endswith((".csv", ".txt")):
                        file_content = uploaded_file_surv.getvalue().decode("utf-8")
                        buffer = io.StringIO(file_content)
                        delimiter = detect_delimiter(file_content)
                        df = pd.read_csv(buffer, delimiter=delimiter)
                    elif uploaded_file_surv.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file_surv)
                    else:
                        st.error("Unsupported format.")
                        df = None
                    if df is not None:
                        st.session_state["survival_data"] = df
                        st.success(f"✅ Survival data loaded: {df.shape[0]} samples, {df.shape[1]} columns.")
                        if st.checkbox("Show preview", key="surv_preview"):
                            st.dataframe(df.head(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading survival data: {e}")

        # ═══════════════════════════════════════════════════════════
        #  SECTION 5 — LONGITUDINAL DATA
        # ═══════════════════════════════════════════════════════════
        with st.sidebar.expander("📈 Step 5 — Longitudinal / Time-Series Data", expanded=False):
            # st.markdown(
            #     "<div style='background:#fffbeb;border-left:3px solid #d97706;"
            #     "border-radius:6px;padding:8px 12px;font-size:0.76rem;color:#92400e;margin-bottom:8px;'>"
            #     "<b>⚠️</b> — Profiler is primarily designed for cross-sectional omics data. "
            #     "Longitudinal support is partial: repeated-measures data can be loaded and visualised, "
            #     "but dedicated mixed-effects models (LME, GEE) is not available in AI modeling tab. "
            #     "Standard ML models will treat each row independently."
            #     "</div>",
            #     unsafe_allow_html=True
            # )
            st.caption(
                "Expected format: one row per sample-timepoint. "
                "Requires a **Subject_ID** column (patient/animal ID), a **Time** column "
                "(visit/timepoint), and a **Class** column (group/condition)."
            )
            if st.checkbox("📋 Show expected format", key="show_long_fmt"):
                st.markdown("""
| Subject_ID | Time | Class | Protein_A | Protein_B | age_meta |
|---|---|---|---|---|---|
| P01 | T0 | Ctrl | 1257 | 0.45 | 58 |
| P01 | T1 | Ctrl | 1389 | 0.50 | 58 |
| P02 | T0 | Case | 752 | 1.30 | 62 |
| P02 | T1 | Case | 810 | 1.20 | 62 |

- **Subject_ID** / `patient_id` / `subject` → identifies the individual across timepoints
- **Time** / `visit` / `timepoint` → numeric or categorical (T0, T1, Week4…)
- **Class** → group/condition label (treated as cross-sectional by current ML modules)
""")
            uploaded_long = st.file_uploader(
                "Upload longitudinal dataset (CSV, XLSX, TSV)",
                type=["csv", "xlsx", "txt", "tsv"],
                key="uploaded_file_longitudinal",
                help="One row per sample-timepoint. Subject_ID + Time columns required."
            )
            if uploaded_long is not None:
                try:
                    # ── Load raw bytes — bypass load_structured_data to avoid
                    #    coercion of Time/Subject_ID columns ──────────────────
                    import io as _io
                    _raw_bytes = uploaded_long.read()
                    _fname = uploaded_long.name.lower()
                    if _fname.endswith((".csv", ".txt", ".tsv")):
                        _sep = "\t" if _fname.endswith(".tsv") else None
                        df_long = pd.read_csv(_io.BytesIO(_raw_bytes), sep=_sep,
                                              engine="python")
                    elif _fname.endswith(".xlsx"):
                        df_long = pd.read_excel(_io.BytesIO(_raw_bytes))
                    else:
                        df_long = pd.read_csv(_io.BytesIO(_raw_bytes), sep=None,
                                              engine="python")

                    if df_long is not None and not df_long.empty:
                        # ── Auto-detect Subject_ID and Time columns ──────────
                        _subj_aliases = {"subject_id", "subjectid", "patient_id",
                                         "patientid", "subject", "patient",
                                         "individual", "animal_id"}
                        _time_aliases = {"time", "timepoint", "visit", "week",
                                         "day", "month", "year"}
                        _auto_subj = next(
                            (c for c in df_long.columns
                             if c.lower().replace(" ", "_") in _subj_aliases), None)
                        _auto_time = next(
                            (c for c in df_long.columns
                             if c.lower().replace(" ", "_") in _time_aliases), None)

                        if _auto_subj:
                            st.success(f"✅ Subject ID: **{_auto_subj}** "
                                       f"({df_long[_auto_subj].nunique()} subjects)")
                        else:
                            st.warning("⚠️ No Subject_ID column detected.")
                        if _auto_time:
                            _tp = sorted(df_long[_auto_time].dropna().unique().tolist())
                            st.success(f"✅ Time: **{_auto_time}** → "
                                       f"{len(_tp)} timepoints: {_tp[:6]}")
                        else:
                            st.warning("⚠️ No Time column detected.")

                        # ── Column override selectors ────────────────────────
                        _col_opts = ["— auto —"] + list(df_long.columns)
                        col_s, col_t = st.columns(2)
                        with col_s:
                            _sel_subj = st.selectbox(
                                "Subject ID column", _col_opts,
                                index=0 if _auto_subj is None
                                      else _col_opts.index(_auto_subj),
                                key="long_subj_col")
                        with col_t:
                            _sel_time = st.selectbox(
                                "Time column", _col_opts,
                                index=0 if _auto_time is None
                                      else _col_opts.index(_auto_time),
                                key="long_time_col")

                        if st.button("✅ Load Longitudinal Data", key="load_long_btn",
                                     use_container_width=True):
                            # ── Resolve final column names ───────────────────
                            _final_subj = (_sel_subj if _sel_subj != "— auto —"
                                           else _auto_subj)
                            _final_time = (_sel_time if _sel_time != "— auto —"
                                           else _auto_time)

                            if _final_subj and _final_subj != "Subject_ID":
                                df_long = df_long.rename(
                                    columns={_final_subj: "Subject_ID"})
                            if _final_time and _final_time != "Time":
                                df_long = df_long.rename(
                                    columns={_final_time: "Time"})

                            # ── Coerce only numeric feature columns ──────────
                            _long_protect = {"Subject_ID", "Time", "Class", "ID",
                                             "File", "RT", "Sum", "Original_index"}
                            _long_protect.update(
                                c for c in df_long.columns
                                if str(c).endswith("_meta"))
                            _long_feats = [c for c in df_long.columns
                                           if c not in _long_protect]
                            if _long_feats:
                                df_long[_long_feats] = df_long[_long_feats].apply(
                                    pd.to_numeric, errors="coerce")

                            # Add compatibility columns
                            for _rc in ["File", "RT", "Sum"]:
                                if _rc not in df_long.columns:
                                    df_long[_rc] = "Unknown" if _rc == "File" else 0

                            # ── Store in session state ───────────────────────
                            st.session_state["data"]            = df_long
                            st.session_state["longitudinal_df"] = df_long.copy()
                            st.session_state["is_longitudinal"] = True
                            if "Class" in df_long.columns:
                                import plotly.express as _px_l
                                _cls_list = df_long["Class"].dropna().unique().tolist()
                                _plotly_pal = _px_l.colors.qualitative.Plotly
                                st.session_state["class_renaming"] = {
                                    c: c for c in _cls_list}
                                st.session_state["class_colors"] = {
                                    c: _plotly_pal[i % len(_plotly_pal)]
                                    for i, c in enumerate(_cls_list)}
                            _tp_final = df_long["Time"].dropna().unique().tolist()
                            st.success(
                                f"✅ Longitudinal data loaded — "
                                f"{df_long['Subject_ID'].nunique()} subjects · "
                                f"{len(_tp_final)} timepoints: "
                                f"{sorted(_tp_final)}"
                            )
                            st.rerun()
                except Exception as _le:
                    import traceback as _tb
                    st.error(f"Error loading longitudinal data: {_le}")
                    st.code(_tb.format_exc(), language="python")

        # ═══════════════════════════════════════════════════════════
        #  SECTION 6 — ADDITIONAL TOOLS (MSI2Profiler)
        # ═══════════════════════════════════════════════════════════
        with st.sidebar.expander("🔧 Additional Tools — MSI2Profiler", expanded=False):
            st.markdown("""
<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:10px 12px;font-size:0.78rem;color:#0c4a6e;'>
<b>MSI2Profiler</b> is a desktop tool for <b>Mass Spectrometry Imaging (MSI)</b> data.<br>
It converts <code>.imzML</code> files → CSV for direct import into Profiler.<br><br>
<b>Key features:</b> load ROIs from imzML · bin spectra · normalize · export CSV/Excel · visualize average spectra · concatenate multiple ROI files.
</div>""", unsafe_allow_html=True)
            msi_zip_path = os.path.join(script_dir, "MSI2Profiler_Windows.zip")
            if os.path.exists(msi_zip_path):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with open(msi_zip_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download MSI2Profiler (Windows)",
                            f, "MSI2Profiler_Windows.zip",
                            mime="application/zip",
                            key="msi2profiler_download_sidebar",
                            use_container_width=True
                        )
            else:
                st.info("MSI2Profiler is available on the desktop version.")

        # ═══════════════════════════════════════════════════════════
        #  PIPELINE STATUS
        # ═══════════════════════════════════════════════════════════
        # ── Data status indicator in sidebar ──
        st.sidebar.markdown("---")
        has_raw      = st.session_state.get("data") is not None or st.session_state.get("final_data") is not None
        has_pp       = st.session_state.get("preprocessed_data") is not None
        has_os       = st.session_state.get("oversampled_data") is not None
        has_us       = st.session_state.get("undersampled_data") is not None
        has_ml       = bool(st.session_state.get("models"))
        has_dl       = bool(st.session_state.get("dl_models"))
        has_shap     = st.session_state.get("shap_values") is not None
        has_survival = st.session_state.get("survival_data") is not None
        has_enrich   = st.session_state.get("_report_enrichment_bar") is not None
        has_realtime = has_ml or has_dl
        has_volcano  = st.session_state.get("volcano_data") is not None
        has_heatmap  = st.session_state.get("heatmap_fig") is not None
        has_pca      = st.session_state.get("pca_fig") is not None or st.session_state.get("fig_initial") is not None

        has_batch    = st.session_state.get("preprocessed_data") is not None and st.session_state.get("preprocessing_summary", {}).get("batch_correction") is not None
        has_qc       = st.session_state.get("qc_analysis_done", False)
        has_sampling = has_os or has_us
        has_biomark  = st.session_state.get("volcano_data") is not None or st.session_state.get("show_boxplots") is True

        def _status(flag, label, icon=""):
            if flag:
                dot    = "<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:linear-gradient(135deg,#16a34a,#22c55e);margin-right:7px;vertical-align:middle;box-shadow:0 0 4px rgba(22,163,74,0.5);'></span>"
                color, fw = "#15803d", "700"
                prefix = "✓ "
            else:
                dot    = "<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:rgba(100,116,139,0.25);margin-right:7px;vertical-align:middle;border:1px solid rgba(100,116,139,0.35);'></span>"
                color, fw = "#94a3b8", "400"
                prefix = ""
            return f"<div style='font-size:0.71rem;color:{color};font-weight:{fw};padding:2px 0 2px 2px;'>{dot}{prefix}{label}</div>"

        steps = [
            (has_raw,       "Data loaded"),
            (has_qc,        "QC / Overview"),
            (has_pp,        "Preprocessing"),
            (has_sampling,  "Class balancing"),
            (has_pca,       "Dim. reduction (PCA/UMAP)"),
            (has_heatmap,   "Heatmap clustering"),
            (has_volcano,   "Volcano / Biomarkers"),
            (has_ml,        "ML models trained"),
            (has_dl,        "DL models trained"),
            (has_shap,      "SHAP explained"),
            (has_enrich,    "Enrichment analysis"),
            (has_survival,  "Survival analysis"),
            (has_realtime,  "Real-time ready"),
        ]
        done_count = sum(1 for f, _ in steps if f)
        pct        = int(done_count / len(steps) * 100)

        # Adaptive colour: red-orange < 30 %, yellow 30–65 %, green > 65 %
        if pct < 30:
            _bar_col  = "linear-gradient(90deg,#ef4444,#f97316)"
            _bdg_col  = "#fecaca"; _bdg_text = "#991b1b"
        elif pct < 65:
            _bar_col  = "linear-gradient(90deg,#f59e0b,#eab308)"
            _bdg_col  = "#fef08a"; _bdg_text = "#78350f"
        else:
            _bar_col  = "linear-gradient(90deg,#16a34a,#22c55e)"
            _bdg_col  = "#bbf7d0"; _bdg_text = "#14532d"

        bar_html = (
            "<div style='margin-bottom:4px;'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>"
            "<span style='color:#1e3a5f;font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:0.10em;'>⬡ Pipeline Status</span>"
            f"<span style='background:{_bdg_col};color:{_bdg_text};font-size:0.62rem;font-weight:700;"
            f"padding:1px 6px;border-radius:999px;'>{done_count}/{len(steps)}</span>"
            "</div>"
            "<div style='background:#c5d5e8;border-radius:4px;height:7px;overflow:hidden;"
            "box-shadow:inset 0 1px 2px rgba(0,0,0,0.10);'>"
            f"<div style='background:{_bar_col};height:100%;width:{pct}%;border-radius:4px;"
            "transition:width 0.4s ease;'></div>"
            "</div>"
            f"<div style='text-align:right;font-size:0.62rem;color:#64748b;margin-top:2px;'>{pct}% complete</div>"
            "</div>"
        )
        steps_html = "".join([_status(f, l) for f, l in steps])

        st.sidebar.markdown(
            "<div style='padding:10px 12px;background:rgba(255,255,255,0.85);border-radius:10px;"
            "border:1px solid #c0d4ea;margin-bottom:8px;"
            "box-shadow:0 1px 4px rgba(49,140,231,0.08);'>"
            + bar_html
            + f"<div style='margin-top:8px;'>{steps_html}</div>"
            + "</div>",
            unsafe_allow_html=True
        )
        st.sidebar.markdown("---")

        render_run_all_button()

        # ── mzML TIC chromatogram sidebar (only after mzML import) ──
        if st.session_state.get("mzml_chrom_data"):
            render_mzml_chromatogram_sidebar()

    # (decoration bar hidden via global theme)

    # ── Desktop: all tabs always visible ────────────────────
    _tabs_list = [
            "  Home", "  Data Lab", "  Data Viz",
            "  Comparisons", "  AI Modeling", "  Biomarkers",
            "  Enrichment", "  Survival", "  Real-Time"
        ]
    tabs = st.tabs(_tabs_list)



    with tabs[0]:
        _n_raw = st.session_state.get('data') is not None or st.session_state.get('final_data') is not None
        _n_pp  = st.session_state.get('preprocessed_data') is not None
        _n_ml  = bool(st.session_state.get('models'))

        # ── Hero banner ──
        _s_raw = 'rgba(16,185,129,0.25)' if _n_raw else 'rgba(100,116,139,0.12)'
        _c_raw = '#15803d' if _n_raw else '#94a3b8'
        _t_raw = '✅ Data loaded' if _n_raw else '○ No data'
        _s_pp  = 'rgba(16,185,129,0.25)' if _n_pp  else 'rgba(100,116,139,0.12)'
        _c_pp  = '#15803d' if _n_pp  else '#94a3b8'
        _t_pp  = '✅ Preprocessed' if _n_pp  else '○ Not preprocessed'
        _s_ml  = 'rgba(16,185,129,0.25)' if _n_ml  else 'rgba(100,116,139,0.12)'
        _c_ml  = '#15803d' if _n_ml  else '#94a3b8'
        _t_ml  = '✅ Models trained' if _n_ml  else '○ No model'
        _omics_pills = " ".join([
            f"<span style='background:rgba(255,255,255,0.07);color:#cbd5e1;font-size:0.7rem;"
            f"padding:3px 10px;border-radius:20px;border:1px solid rgba(255,255,255,0.1);'>{p}</span>"
            for p in [' Proteomics',' Metabolomics',' Lipidomics',
                      ' Genomics',' Transcriptomics',' MS Imaging']
        ])
        st.markdown(
            f'''<div style="background:linear-gradient(135deg,#0b1f45 0%,#0d2d5a 100%);
                border-radius:16px;padding:28px 32px 22px;margin-bottom:20px;
                border:1px solid rgba(49,140,231,0.25);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <h1 style="margin:0;font-size:1.8rem;font-weight:900;color:#fff;">Profiler</h1>
                <span style="background:rgba(49,140,231,0.3);color:#93c5fd;font-size:0.7rem;
                font-weight:700;padding:2px 9px;border-radius:20px;border:1px solid rgba(49,140,231,0.5);">v1.2</span>
            </div>
            <p style="color:#7ea8cc;font-size:0.78rem;margin:0 0 6px;">PRISM U1192 · INSERM · Université de Lille</p>
            <p style="color:#a8c8e8;font-size:0.88rem;margin:0 0 12px;max-width:580px;line-height:1.6;">
                Open-source omics analysis platform : from raw data to biological insights.
            </p>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">{_omics_pills}</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <span style="background:{_s_raw};color:{_c_raw};font-size:0.71rem;padding:2px 10px;border-radius:20px;">{_t_raw}</span>
                <span style="background:{_s_pp};color:{_c_pp};font-size:0.71rem;padding:2px 10px;border-radius:20px;">{_t_pp}</span>
                <span style="background:{_s_ml};color:{_c_ml};font-size:0.71rem;padding:2px 10px;border-radius:20px;">{_t_ml}</span>
            </div>
            </div>''',
            unsafe_allow_html=True
        )

        # ── Capability cards – Row 1: Core pipeline ──
        st.markdown("<div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px;'><div style='flex:1;height:1px;background:linear-gradient(90deg,#e2e8f0,transparent);'></div><p style='color:#94a3b8;font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:0.14em;margin:0;white-space:nowrap;'>◈ Core Pipeline</p><div style='flex:1;height:1px;background:linear-gradient(90deg,transparent,#e2e8f0);'></div></div>", unsafe_allow_html=True)
        _r1 = st.columns(4)
        for _col, _ic, _ti, _de, _bc in zip(_r1,
            ["","","",""],
            ["Data Import"," Preprocessing","QC & Filtering","Class Balancing"],
            ["CSV · mzML · MaxQuant · DIA-NN · Perseus · MS1 spectra",
             "Normalize · Impute · Batch-correct · Transform",
             "Missing values · Outliers · Zero-inflation · Normality",
             "SMOTE · ADASYN · RandomUnderSampler · NearMiss"],
            ["#0ea5e9","#8b5cf6","#f59e0b","#10b981"]):
            _col.markdown(f'''<div style="border:1px solid rgba(255,255,255,0.1);border-left:3px solid {_bc};
                border-radius:12px;padding:14px 16px;
                background:linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.06));
                backdrop-filter:blur(8px);height:100%;margin-bottom:10px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08),inset 0 1px 0 rgba(255,255,255,0.08);">
                <div style="font-size:1.3rem;margin-bottom:6px;">{_ic}</div>
                <div style="font-weight:800;color:#1e293b;font-size:0.82rem;margin-bottom:5px;letter-spacing:0.01em;">{_ti}</div>
                <div style="width:24px;height:2px;background:{_bc};border-radius:2px;margin-bottom:6px;opacity:0.7;"></div>
                <p style="color:#64748b;font-size:0.71rem;margin:0;line-height:1.5;">{_de}</p>
            </div>''', unsafe_allow_html=True)

        # ── Row 2: Visualization & Statistics ──
        st.markdown("<div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px;'><div style='flex:1;height:1px;background:linear-gradient(90deg,#e2e8f0,transparent);'></div><p style='color:#94a3b8;font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:0.14em;margin:0;white-space:nowrap;'>◈ Visualization & Statistics</p><div style='flex:1;height:1px;background:linear-gradient(90deg,transparent,#e2e8f0);'></div></div>", unsafe_allow_html=True)
        _r2 = st.columns(4)
        for _col, _ic, _ti, _de, _bc in zip(_r2,
            ["","","",""],
            ["Data Visualization","Class Comparisons","Differential Analysis","Heatmap Clustering"],
            ["PCA · UMAP · t-SNE · Radar · Feature profiles",
             "Pearson · Spearman · Cosine · Cohen's κ · Venn · UpSet",
             "Volcano plot · p-value · Fold-change · FDR · Bonferroni",
             "Hierarchical clustering · Feature & sample heatmaps"],
            ["#06b6d4","#6366f1","#ef4444","#f97316"]):
            _col.markdown(f'''<div style="border:1px solid rgba(255,255,255,0.1);border-left:3px solid {_bc};
                border-radius:12px;padding:14px 16px;
                background:linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.06));
                backdrop-filter:blur(8px);height:100%;margin-bottom:10px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08),inset 0 1px 0 rgba(255,255,255,0.08);">
                <div style="font-size:1.3rem;margin-bottom:6px;">{_ic}</div>
                <div style="font-weight:800;color:#1e293b;font-size:0.82rem;margin-bottom:5px;letter-spacing:0.01em;">{_ti}</div>
                <div style="width:24px;height:2px;background:{_bc};border-radius:2px;margin-bottom:6px;opacity:0.7;"></div>
                <p style="color:#64748b;font-size:0.71rem;margin:0;line-height:1.5;">{_de}</p>
            </div>''', unsafe_allow_html=True)

        # ── Row 3: AI & Interpretation ──
        st.markdown("<div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px;'><div style='flex:1;height:1px;background:linear-gradient(90deg,#e2e8f0,transparent);'></div><p style='color:#94a3b8;font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:0.14em;margin:0;white-space:nowrap;'>◈ AI & Biological Interpretation</p><div style='flex:1;height:1px;background:linear-gradient(90deg,transparent,#e2e8f0);'></div></div>", unsafe_allow_html=True)
        _r3 = st.columns(4)
        for _col, _ic, _ti, _de, _bc in zip(_r3,
            ["","","",""],
            ["Machine Learning","Deep Learning","Explainability","Pathway Enrichment"],
            ["20+ classifiers · Cross-validation · ROC · Confusion matrix",
             "MLP · Custom DNN · Learning curves · Batch training",
             "SHAP beeswarm/bar · LIME feature importance",
             "GSEA · Enrichr · Gene Ontology · KEGG · Reactome"],
            ["#8b5cf6","#ec4899","#f59e0b","#10b981"]):
            _col.markdown(f'''<div style="border:1px solid rgba(255,255,255,0.1);border-left:3px solid {_bc};
                border-radius:12px;padding:14px 16px;
                background:linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.06));
                backdrop-filter:blur(8px);height:100%;margin-bottom:10px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08),inset 0 1px 0 rgba(255,255,255,0.08);">
                <div style="font-size:1.3rem;margin-bottom:6px;">{_ic}</div>
                <div style="font-weight:800;color:#1e293b;font-size:0.82rem;margin-bottom:5px;letter-spacing:0.01em;">{_ti}</div>
                <div style="width:24px;height:2px;background:{_bc};border-radius:2px;margin-bottom:6px;opacity:0.7;"></div>
                <p style="color:#64748b;font-size:0.71rem;margin:0;line-height:1.5;">{_de}</p>
            </div>''', unsafe_allow_html=True)

        # ── Row 4: Clinical & Real-Time ──
        st.markdown("<div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px;'><div style='flex:1;height:1px;background:linear-gradient(90deg,#e2e8f0,transparent);'></div><p style='color:#94a3b8;font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:0.14em;margin:0;white-space:nowrap;'>◈ Clinical & Real-Time</p><div style='flex:1;height:1px;background:linear-gradient(90deg,transparent,#e2e8f0);'></div></div>", unsafe_allow_html=True)
        _r4 = st.columns(4)
        for _col, _ic, _ti, _de, _bc in zip(_r4,
            ["","","",""],
            ["Biomarker Discovery","Survival Analysis","Real-Time Prediction","Model Export"],
            ["Statistical ranking · Effect size · Feature selection",
             "Kaplan-Meier · Log-rank test · Cox regression · Risk groups",
             "SpiderMass MS · Post-hoc tabular · Live classification",
             "Save & load models · .pkl format · Label encoder export"],
            ["#ef4444","#14b8a6","#7c3aed","#64748b"]):
            _col.markdown(f'''<div style="border:1px solid rgba(255,255,255,0.1);border-left:3px solid {_bc};
                border-radius:12px;padding:14px 16px;
                background:linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.06));
                backdrop-filter:blur(8px);height:100%;margin-bottom:10px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08),inset 0 1px 0 rgba(255,255,255,0.08);">
                <div style="font-size:1.3rem;margin-bottom:6px;">{_ic}</div>
                <div style="font-weight:800;color:#1e293b;font-size:0.82rem;margin-bottom:5px;letter-spacing:0.01em;">{_ti}</div>
                <div style="width:24px;height:2px;background:{_bc};border-radius:2px;margin-bottom:6px;opacity:0.7;"></div>
                <p style="color:#64748b;font-size:0.71rem;margin:0;line-height:1.5;">{_de}</p>
            </div>''', unsafe_allow_html=True)

        # ── About + Downloads ──
        with st.expander("About Profiler & Citation", expanded=False):
            _ac1, _ac2 = st.columns([3, 2])
            with _ac1:
                st.markdown("""
                **Profiler** is a peer-reviewed open-source omics analysis platform developed at
                [PRISM U1192](https://www.laboratoire-prism.fr/) (INSERM, Université de Lille),
                protected by INSERM Transfer (IDDN).

                Supports proteomics, metabolomics, lipidomics, genomics, transcriptomics and MS imaging.
                """)
            with _ac2:
                st.markdown("""
                **Cite:** *Zirem, Y., Ledoux, L., Fournier, I., Salzet, M.*
                "Profiler: an open web platform for multi-omics analysis"
                ***Bioinformatics***, 2025.
                [DOI: 10.1093/bioinformatics/btaf644](https://doi.org/10.1093/bioinformatics/btaf644)
                """)

        _lnk = st.columns(4)
        with _lnk[0]:
            try:
                _doc_path = next((_p for _p in [
                    os.path.normpath(os.path.join(script_dir, "..", "..", "docs", "documentation.pdf")),
                    os.path.join(script_dir, "documentation.pdf"),
                    "documentation.pdf",
                ] if os.path.exists(_p)), None)
                if _doc_path is None: raise FileNotFoundError
                with open(_doc_path,"rb") as _f: st.download_button("📘 Documentation",data=_f,file_name="documentation.pdf",mime="application/pdf",use_container_width=True)
            except FileNotFoundError: st.caption("📘 documentation.pdf not found")
        with _lnk[1]:
            try:
                _cert_path = next((_p for _p in [
                    os.path.normpath(os.path.join(script_dir, "..", "..", "docs", "IDDN_Certificate.pdf")),
                    os.path.join(script_dir, "IDDN_Certificate.pdf"),
                    "IDDN_Certificate.pdf",
                ] if os.path.exists(_p)), None)
                if _cert_path is None: raise FileNotFoundError
                with open(_cert_path,"rb") as _f: st.download_button("⚖️ INSERM Certificate",data=_f,file_name="IDDN_Certificate.pdf",mime="application/pdf",use_container_width=True)
            except FileNotFoundError: st.caption("⚖️ Certificate not found")
        with _lnk[2]: st.link_button("📦 Example Datasets","https://github.com/yanisZirem/Profiler_v1_requests_datatests",use_container_width=True)
        with _lnk[3]: st.link_button("💻 GitHub","https://github.com/yanisZirem/prism-profiler",use_container_width=True)
        st.markdown('''<hr style="margin:18px 0 8px;border:none;border-top:1px solid #e2e8f0;">
        <p style="text-align:center;font-size:0.71rem;color:#94a3b8;">
          © 2025 Profiler — PRISM U1192 INSERM · Université de Lille · All rights reserved
        </p>''', unsafe_allow_html=True)

    # ── Tab content (shown after login) ──
    # ── Protected tabs (login required) ──
    with tabs[1]:
            # ═══════════════════════════════════════════════════════════════
            # DATA LAB ─ source selector + 4 thematic groups (2-level tabs)
            # ═══════════════════════════════════════════════════════════════

            # ── session state init ───────────────────────────────────────────────
            if 'show_info' not in st.session_state:
                st.session_state.show_info = {"dataset_info": False, "missing_values": False, "shapiro_wilk_test": False, "feature_distributions": False}
            st.session_state.setdefault('datalab_df', None)
            st.session_state.setdefault('datalab_source', None)
            st.session_state.setdefault('overview_df', None)
            st.session_state.setdefault('overview_source', None)

            # ── AUTO DATASET LOADER ──────────────────────────────────────────────
            # Priority: Raw/Edited first in DataLab (user can switch to Preprocessed manually)
            _dl_raw = st.session_state.get('final_data', st.session_state.get('data'))
            _dl_pp  = st.session_state.get('preprocessed_data')
            _dl_os  = st.session_state.get('oversampled_data')
            _dl_us  = st.session_state.get('undersampled_data')

            if _dl_raw is not None:
                _dl_best, _dl_best_name = _dl_raw, "Raw / Edited"
            elif _dl_pp is not None:
                _dl_best, _dl_best_name = _dl_pp, "Preprocessed"
            elif _dl_os is not None:
                _dl_best, _dl_best_name = _dl_os, "Oversampled"
            elif _dl_us is not None:
                _dl_best, _dl_best_name = _dl_us, "Undersampled"
            else:
                _dl_best, _dl_best_name = None, None

            if _dl_best is not None and st.session_state.get('datalab_source') is None:
                st.session_state['datalab_df']      = _dl_best.copy()
                st.session_state['datalab_source']  = _dl_best_name
                st.session_state['overview_df']     = _dl_best.copy()
                st.session_state['overview_source'] = _dl_best_name
                st.session_state.setdefault('class_colors', {})
                if 'Class' in _dl_best.columns:
                    for _cls in _dl_best['Class'].unique():
                        st.session_state['class_colors'].setdefault(_cls, "#318CE7")

            if st.session_state.get('datalab_df') is not None:
                _cdf   = st.session_state['datalab_df']
                _ns    = _cdf.shape[0]
                _nf    = _cdf.shape[1] - (1 if 'Class' in _cdf.columns else 0)
                _nc    = _cdf['Class'].nunique() if 'Class' in _cdf.columns else 0
                _src   = st.session_state.get('datalab_source', '')
                _badge = "#10b981" if _src == "Preprocessed" else "#318CE7"
                st.markdown(
                    f"<div style='padding:6px 10px;background:#f8fafc;border:1px solid #e2e8f0;"
                    f"border-radius:8px;font-size:0.82rem;color:#64748b;'>"
                    f"<span style='background:{_badge};color:white;border-radius:4px;"
                    f"padding:2px 8px;font-size:0.75rem;font-weight:700;margin-right:8px;'>{_src}</span>"
                    f"<b style='color:#1e3a5f;'>{_ns:,}</b> samples &nbsp;·&nbsp;"
                    f"<b style='color:#10b981;'>{_nf:,}</b> features &nbsp;·&nbsp;"
                    f"<b style='color:#f59e0b;'>{_nc}</b> classes</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("📂 Load a dataset from the sidebar to get started.", icon="ℹ️")

            # Manual override selector (only shown when multiple sources are available)
            _dl_avail = {k: v for k, v in {
                "Raw / Edited": st.session_state.get('final_data', st.session_state.get('data')),
                "Preprocessed": st.session_state.get('preprocessed_data'),
                "Oversampled":  st.session_state.get('oversampled_data'),
                "Undersampled": st.session_state.get('undersampled_data'),
            }.items() if v is not None}
            if len(_dl_avail) > 1:
                _dl_src_keys = list(_dl_avail.keys())
                # Always derive current source from session_state — never from widget key
                _dl_src_cur = st.session_state.get('datalab_source', 'Raw / Edited')
                if _dl_src_cur not in _dl_src_keys:
                    _dl_src_cur = _dl_src_keys[0]
                _dl_src_idx = _dl_src_keys.index(_dl_src_cur)
                # Sync widget key to match datalab_source so the selectbox reflects truth
                st.session_state['datalab_manual_source'] = _dl_src_cur
                _dl_col_src, _ = st.columns([3, 5])
                with _dl_col_src:
                    _dl_src_sel = st.selectbox(
                        "🗂️ Data source",
                        _dl_src_keys,
                        index=_dl_src_idx,
                        key="datalab_manual_source",
                        help="Switch between available dataset versions.",
                    )
                if _dl_src_sel != st.session_state.get('datalab_source'):
                    _sel = _dl_avail[_dl_src_sel]
                    st.session_state['datalab_df']      = _sel.copy()
                    st.session_state['datalab_source']  = _dl_src_sel
                    st.session_state['overview_df']     = _sel.copy()
                    st.session_state['overview_source'] = _dl_src_sel
                    st.rerun()

            _t1_df = st.session_state.get('datalab_df')
            df     = _t1_df
            _t1_no_data = _t1_df is None



            st.markdown("<hr style='margin:14px 0;border:none;border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

            # ── LEVEL 1 : 4 thematic group tabs — always visible after login ──────────
            _grp_clean, _grp_balance, _grp_explore, _grp_process = st.tabs([
                "  Clean & Edit",
                "  Class & Sample QC",
                "  Data Overview",
                "  Preprocess",
            ])

            # # Show message if no dataset loaded — group tabs visible, content locked
            # if _t1_no_data:
            #     for _g in [_grp_clean, _grp_balance, _grp_explore, _grp_process]:
            #         with _g:
            #             st.info("Select and load a dataset above to access the analyses.", icon="ℹ️")

            # ════════════════════════════════════════════════════════════════════════
            # GROUP 1 — EXPLORE   (Overview · Missing & Zeros · Distribution)
            # ════════════════════════════════════════════════════════════════════════
            with _grp_explore:
                st.markdown(_picon("overview","Inspect dataset structure, missing values and feature distributions.", "#318CE7"), unsafe_allow_html=True)
                _t1_missing, _t1_distrib = st.tabs([
                    "  Missing & Zeros", "  Distribution"
                ])

                with _t1_missing:

                    with st.form("missing_values_form"):
                        toggle = st.form_submit_button("Show Missing / Zero Info")

                    if toggle:
                        st.session_state.show_info["missing_values"] = not st.session_state.show_info["missing_values"]

                    if st.session_state.show_info["missing_values"]:

                        # import plotly.express as px
                        # import plotly.graph_objects as go
                        # from plotly.subplots import make_subplots
                

                        relevant_cols, missing_df = calculate_missing_values(df)
                        # Safety: keep only numeric columns to avoid str > int errors
                        relevant_cols = [c for c in relevant_cols if pd.api.types.is_numeric_dtype(df[c])]

                        # ============================================================
                        # MISSING VALUES
                        # ============================================================
                        st.markdown("**Missing Values**")

                        if missing_df.empty:
                            st.success("✅ No missing values detected.")
                            total_missing_pct = 0
                            zero_inflated_features = pd.Series(dtype=float)
                        else:
                            total_missing_pct = (
                                df[relevant_cols].isnull().sum().sum()
                                / (df.shape[0] * len(relevant_cols))
                            ) * 100

                            st.success(f"Overall missingness: **{total_missing_pct:.2f}%**")

                            # ── 1. Pie charts par classe ──────────────────────────────
                            if "Class" in df.columns:
                                classes = df["Class"].unique()


                                import plotly.express as px
                                palette = px.colors.qualitative.Plotly
                                color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(sorted(classes))}

                                # Une subplot de pies : 1 pie par classe
                                from plotly.subplots import make_subplots
                                n_cols = min(3, len(classes))
                                n_rows = -(-len(classes) // n_cols)  # ceiling division

                                fig_pies = make_subplots(
                                    rows=n_rows,
                                    cols=n_cols,
                                    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
                                    subplot_titles=[str(c) for c in sorted(classes)]
                                )

                                for i, cls in enumerate(sorted(classes)):
                                    sub = df[df["Class"] == cls][relevant_cols].select_dtypes(include="number")
                                    n_missing = sub.isnull().sum().sum()
                                    n_total = sub.shape[0] * len(relevant_cols)
                                    n_present = n_total - n_missing

                                    row = i // n_cols + 1
                                    col = i % n_cols + 1

                                    fig_pies.add_trace(
                                        go.Pie(
                                            labels=["Present", "Missing"],
                                            values=[n_present, n_missing],
                                            marker_colors=[color_map[cls], "#d3d3d3"],
                                            hole=0.4,
                                            textinfo="percent",
                                            showlegend=(i == 0),
                                            name=str(cls)
                                        ),
                                        row=row, col=col
                                    )

                                fig_pies.update_layout(
                                    title_text="<b>Missing Values (%) per Class</b>",
                                    height=300 * n_rows,
                                    font=dict(size=15, color="black", family="Arial"),
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                                                font=dict(size=14, color="black")),
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    title_font=dict(size=18, color="black", family="Arial"),
                                )
                                st.plotly_chart(fig_pies, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                _capture_plotly(fig_pies, "missing_values_per_class_pies")


                            # ── Cumulative Missing Value Curve ─────────────────────────
                            # Mirrors exactly the zero-detection cumulative curve logic:
                            # per-class lines + class_colors palette + thresholds at 20 & 30 %
                            thresholds = range(0, 101, 5)
                            cum_records = []
                            _cls_cmap = st.session_state.get('class_colors', {})

                            if "Class" in df.columns:
                                for cls in sorted(df["Class"].unique()):
                                    sub = df[df["Class"] == cls][relevant_cols].select_dtypes(include="number")
                                    missing_frac = sub.isna().sum() / max(len(sub), 1)
                                    for thr in thresholds:
                                        n_feat_kept = (missing_frac * 100 <= thr).sum()
                                        cum_records.append({
                                            "Missing threshold (% samples)": thr,
                                            "Features retained (%)": n_feat_kept / max(len(relevant_cols), 1) * 100,
                                            "Class": str(cls)
                                        })
                                cum_missing_df = pd.DataFrame(cum_records)
                                fig_missing = px.line(
                                    cum_missing_df,
                                    x="Missing threshold (% samples)",
                                    y="Features retained (%)",
                                    color="Class",
                                    color_discrete_map={str(k): v for k, v in _cls_cmap.items()},
                                    markers=True,
                                    title="Cumulative Missing Value Curve per Class"
                                )
                            else:
                                sub = df[relevant_cols].select_dtypes(include="number")
                                missing_frac = sub.isna().sum() / max(len(sub), 1)
                                for thr in thresholds:
                                    n_feat_kept = (missing_frac * 100 <= thr).sum()
                                    cum_records.append({
                                        "Missing threshold (% samples)": thr,
                                        "Features retained (%)": n_feat_kept / max(len(relevant_cols), 1) * 100,
                                    })
                                cum_missing_df = pd.DataFrame(cum_records)
                                fig_missing = px.line(
                                    cum_missing_df,
                                    x="Missing threshold (% samples)",
                                    y="Features retained (%)",
                                    markers=True,
                                    title="Cumulative Missing Value Curve"
                                )

                            # Threshold reference lines (QC: 20 % and 30 %)
                            for xref, lbl in [(20, "20% recommended"), (30, "30% lenient")]:
                                fig_missing.add_vline(
                                    x=xref,
                                    line_dash="dash",
                                    line_color="#f59e0b" if xref == 20 else "#ef4444",
                                    annotation_text=lbl,
                                    annotation_position="top right",
                                    annotation_font=dict(size=12, color="#555"),
                                )

                            fig_missing.update_layout(
                                plot_bgcolor="white", paper_bgcolor="white",
                                height=460,
                                font=dict(size=15, color="black", family="Arial"),
                                title=dict(font=dict(size=18, color="black", family="Arial")),
                                legend=dict(font=dict(size=14, color="black", family="Arial"),
                                            bordercolor="#ccc", borderwidth=1),
                                xaxis=dict(
                                    title="Missing value threshold (% of samples in class)",
                                    gridcolor="#eeeeee",
                                    title_font=dict(size=15, color="black"),
                                    tickfont=dict(size=14, color="black"),
                                    linecolor="#333", linewidth=1.5, mirror=True,
                                ),
                                yaxis=dict(
                                    title="Features retained (%)",
                                    gridcolor="#eeeeee", range=[0, 105],
                                    title_font=dict(size=15, color="black"),
                                    tickfont=dict(size=14, color="black"),
                                    linecolor="#333", linewidth=1.5, mirror=True,
                                ),
                            )

                            st.plotly_chart(fig_missing, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                            _capture_plotly(fig_missing, "cumulative_missing_curve")

                            # ── NEW: Missing heatmap + per-class bar + completeness ──
                            try:
                                fig_miss_hm = plot_missing_heatmap(df)
                                st.plotly_chart(fig_miss_hm, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                _capture_plotly(fig_miss_hm, "missing_heatmap")
                            except Exception:
                                pass

                            try:
                                fig_miss_cls = plot_missing_per_class(df)
                                if fig_miss_cls:
                                    st.plotly_chart(fig_miss_cls, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                    _capture_plotly(fig_miss_cls, "missing_per_class_bar")
                            except Exception:
                                pass

                            try:
                                fig_compl = plot_feature_completeness_rank(df)
                                if fig_compl:
                                    st.plotly_chart(fig_compl, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                    _capture_plotly(fig_compl, "feature_completeness_rank")
                            except Exception:
                                pass

                            st.markdown("**📋 Per-feature missing summary**")
                            st.dataframe(missing_df, use_container_width=True)

                        # # ============================================================
                        # # ZERO-INFLATION
                        # # ============================================================
                        # st.markdown("---")
                        # st.markdown("**Zero-Inflation Analysis**")
                        # st.caption("NaN values are excluded from zero-inflation calculations.")

                        non_nan_df = df[relevant_cols].select_dtypes(include='number')

                        zero_pct_features = (
                            (non_nan_df == 0).sum()
                            / non_nan_df.notna().sum()
                            * 100
                        )

                        zero_pct_samples = (
                            (non_nan_df == 0).sum(axis=1)
                            / non_nan_df.notna().sum(axis=1)
                            * 100
                        )

                        zero_feat_df = (
                            zero_pct_features
                            .reset_index()
                            .rename(columns={"index": "Feature", 0: "Zero (%)"})
                            .sort_values("Zero (%)", ascending=False)
                        )

                        # # st.markdown("**QC-style: Zero-inflation per Sample (colored by Class)**")

                        zero_sample_df = zero_pct_samples.reset_index()
                        zero_sample_df.columns = ["SampleIndex", "Zero (%)"]


                        # st.caption(
                        #     "Each point = % of features detected (non-zero, non-NaN) in at least X% of samples. "
                        #     "Helps determine filtering thresholds for zero-inflated features."
                        # )

                        thresholds = range(0, 101, 5)
                        cum_records = []

                        if "Class" in df.columns:
                            for cls in sorted(df["Class"].unique()):
                                sub = df[df["Class"] == cls][relevant_cols].select_dtypes(include='number')
                                for thr in thresholds:

                                    detected_frac = ((sub > 0) & sub.notna()).sum() / len(sub)
                                    n_feat_detected = (detected_frac * 100 >= thr).sum()
                          
                                    cum_records.append({
                                        "Detection threshold (% samples)": thr,
                                        "Features retained (%)": n_feat_detected / len(relevant_cols) * 100,
                                        "Class": str(cls)
                                    })
                            cum_df = pd.DataFrame(cum_records)
                            fig_cum = px.line(
                                cum_df,
                                x="Detection threshold (% samples)",
                                y="Features retained (%)",
                                color="Class",
                                color_discrete_map={str(k): v for k, v in color_map.items()},
                                markers=True,
                                title="Cumulative Feature Detection Curve per Class"
                            )
                        else:
                            for thr in thresholds:
                                detected_frac = (non_nan_df != 0).sum() / non_nan_df.notna().sum()
                                n_feat_detected = (detected_frac * 100 >= thr).sum()
                                cum_records.append({
                                    "Detection threshold (% samples)": thr,
                                    "Features retained (%)": n_feat_detected / len(relevant_cols) * 100,
                                })
                            cum_df = pd.DataFrame(cum_records)
                            fig_cum = px.line(
                                cum_df,
                                x="Detection threshold (% samples)",
                                y="Features retained (%)",
                                markers=True,
                                title="Cumulative Feature Detection Curve"
                            )

                        # Ligne de référence 50% et 70%
                        for xref in [50, 70]:
                            fig_cum.add_vline(
                                x=xref,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text=f"{xref}% threshold",
                                annotation_position="top right"
                            )

                        fig_cum.update_layout(
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=460,
                            font=dict(size=15, color="black", family="Arial"),
                            title=dict(font=dict(size=18, color="black", family="Arial")),
                            legend=dict(font=dict(size=14, color="black", family="Arial"),
                                        bordercolor="#ccc", borderwidth=1),
                            xaxis=dict(
                                gridcolor="#eeeeee",
                                title_font=dict(size=16, color="black"),
                                tickfont=dict(size=14, color="black"),
                                linecolor="#333", linewidth=1.5, mirror=True,
                            ),
                            yaxis=dict(
                                gridcolor="#eeeeee", range=[0, 105],
                                title_font=dict(size=16, color="black"),
                                tickfont=dict(size=14, color="black"),
                                linecolor="#333", linewidth=1.5, mirror=True,
                            ),
                        )
                        st.plotly_chart(fig_cum, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_cum, "cumulative_feature_detection_curve")

                        # ── NEW: Zero-inflation violin per class ──────────────────
                        try:
                            fig_zi_viol = plot_zero_inflation_per_class(df)
                            if fig_zi_viol:
                                st.plotly_chart(fig_zi_viol, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                _capture_plotly(fig_zi_viol, "zero_inflation_violin_class")
                        except Exception:
                            pass

                        # ── Détection features zero-inflated ─────────────────────────
                        zero_threshold = 50
                        zero_inflated_features = zero_pct_features[zero_pct_features > zero_threshold]

                        if len(zero_inflated_features) > 0:
                            st.warning(
                                f"⚠️ {len(zero_inflated_features)} features show >{zero_threshold}% zeros "
                                "(zero-inflated, NaN excluded)."
                            )
                            st.dataframe(
                                zero_inflated_features
                                .reset_index()
                                .rename(columns={"index": "Feature", 0: "Zero (%)"}),
                                use_container_width=True
                            )
                        else:
                            st.success("✅ No strongly zero-inflated features detected.")

                        st.markdown("**📋 Zero % per feature (NaN excluded)**")
                        st.dataframe(zero_feat_df, use_container_width=True)

                        # ============================================================
                        # IMPUTATION RECOMMENDATIONS
                        # ============================================================
                        st.markdown("---")
                        st.markdown("**💡 Imputation & Filtering Recommendations**")

                        if total_missing_pct < 5 and len(zero_inflated_features) == 0:
                            st.success(
                                "✅ **Low missingness & low sparsity detected**\n\n"
                                "- Data are globally well-covered across samples\n"
                                "- **Recommended imputation:** Mean / Median or fillna(0) (not detected)\n"
                                "- Feature filtering not strictly required"
                            )
                        elif total_missing_pct < 20:
                            st.info(
                                "🔬 **LC-MS/MS after identification (LFQ-like data)**\n\n"
                                "- Missing values mainly reflect **low-abundance signals**\n"
                                "- Zeros should be interpreted as *missing below detection limit*\n\n"
                                "**Recommended strategy:**\n"
                                "- 🔹 **Shifted Gaussian imputation** (default & biologically realistic)\n"
                                "- 🔹 **KNN imputation** if samples are homogeneous and well clustered\n"
                                "- 🔹 Median preferred over Mean if data are skewed\n\n"
                                "**Filtering guidance:**\n"
                                "- Keep features detected in **≥50–60% of samples per class**\n"
                                "- Remove features missing in entire classes"
                            )
                        else:
                            st.warning(
                                "⚠️ **High sparsity / strong zero-inflation detected**\n\n"
                                "- Many features are absent in a large fraction of samples\n"
                                "- Typical of **count-like data** or **direct MS spectral matrices**\n\n"
                                "**Recommended strategy:**\n"
                                "- 🔹 Apply **feature detection filtering first** → keep features detected in **≥70% of samples**\n"
                                "- 🔹 Then apply **KNN** or **Shifted Gaussian** imputation\n"
                                "- ❌ Avoid Mean/Mode in highly sparse matrices\n\n"
                                "**Practical rule:** if a feature is zero or missing in >70% of samples → discard it"
                            )


            # -------------------- Normality, Distribution & Normalization Guidance --------------------

                with _t1_distrib:

                    with st.form("normality_form"):
                        toggle = st.form_submit_button("Show Distribution Info")
                    if toggle:
                        st.session_state.show_info["shapiro_wilk_test"] = not st.session_state.show_info["shapiro_wilk_test"]

                    if st.session_state.show_info["shapiro_wilk_test"] and df is not None:
                        relevant_cols, _ = calculate_missing_values(df)

                        # diagnostics
                        skewness = df[relevant_cols].skew().dropna()
                        skew_mean = skewness.abs().mean()


                        import plotly.express as px

                        # st.markdown("**📈 Feature Distribution Skewness**")

                        fig_density = px.histogram(
                            skewness,
                            nbins=40,
                            marginal="violin",
                            opacity=0.85,
                            title="Distribution of Feature Skewness",
                            labels={"value": "Skewness"},
                            color_discrete_sequence=["#FF9800"],
                        )

                        fig_density.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Symmetric"
                        )

                        fig_density.add_vline(
                            x=1,
                            line_dash="dot",
                            line_color="red",
                            annotation_text="High skew"
                        )

                        fig_density.update_layout(
                            bargap=0.05,
                            height=400
                        )

                        st.plotly_chart(fig_density, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_density, "Features per Sample Distribution Skewness")


                        # Normal vs non-normal features
                        normal_features = (skewness.abs() < 0.5).sum()
                        non_normal_features = (skewness.abs() >= 0.5).sum()


                        fig_norm = px.pie(
                            names=["Normal (|skew| < 0.5)", "Non-normal"],
                            values=[normal_features, non_normal_features],
                            color_discrete_sequence=["#4CAF50", "#F44336"],
                            title="Proportion of Features Following Approximate Normality"
                        )

                        fig_norm.update_traces(
                            textposition="inside",
                            textinfo="percent+label",
                            pull=[0.05, 0]
                        )

                        fig_norm.update_layout(height=400)

                        st.plotly_chart(fig_norm, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_norm, "Features per Sample Normality Proportion")

                        cv = (df[relevant_cols].std() / df[relevant_cols].mean()).replace([np.inf, -np.inf], np.nan)
                        cv_mean = cv.mean()

                        zero_ratio = (df[relevant_cols] == 0).sum().sum() / df[relevant_cols].size

                        st.markdown("**💡Multi-omics diagnostics**")
                        st.info(
                            f"Mean |skew|: {skew_mean:.2f}\n"
                            f"Mean CV: {cv_mean:.2f}\n"
                            f"Zero ratio: {zero_ratio*100:.1f}%"
                        )

                        st.markdown("**💡Normalization guidance (non-prescriptive)**")
                        if zero_ratio > 0.3:
                            st.warning("High sparsity → Median of Ratios / TMM / VST")
                        elif cv_mean > 1:
                            st.warning("Strong mean–variance dependency → VST or log-based normalization")
                        elif skew_mean > 1:
                            st.info("Right-skewed distributions → Log2 / Log10 / VST")
                        elif skew_mean < 0.5:
                            st.success("Near-normal distributions → Total intensity / RMS / BasePeak suitable")
                        else:
                            st.info("Mixed signals → Median / Mean normalization recommended")

                        st.caption(
                            "ℹ️ Recommendations are data-driven and intended to guide, not enforce, normalization choices in multi-omics settings."
                        )



                        # statistical test suggestions (identique à ton code)
                        n_classes = df['Class'].nunique() if 'Class' in df.columns else 0
                        if skew_mean < 0.5:
                            normality_status = "mostly normal"
                            prefer_parametric = True
                        elif skew_mean > 1:
                            normality_status = "highly skewed"
                            prefer_parametric = False
                        else:
                            normality_status = "moderately skewed"
                            prefer_parametric = False

                        if n_classes == 2:
                            test_suggestion = "t-test (parametric)" if prefer_parametric else "Mann-Whitney U (non-parametric)"
                        elif n_classes > 2:
                            test_suggestion = "ANOVA (parametric)" if prefer_parametric else "Kruskal-Wallis (non-parametric)"
                        else:
                            test_suggestion = "N/A (no classes detected)"

                        st.markdown("**💡Recommended Statistical Test:**")
                        st.info(
                            f"Data appear **{normality_status}** (mean |skew| = {skew_mean:.2f}).\n\n"
                            f"Suggested test: **{test_suggestion}**"
                        )

                        st.info(
                            "⚠️ Note: Central Limit Theorem: For n ≥ 30, sampling distribution of the mean approximates normality."
                        )


                        # -------------------- Survival Data --------------------
                        if 'Overall survival' in df.columns and 'State' in df.columns:
                            st.subheader("Survival Data Preview")
                            st.dataframe(df[['Overall survival', 'State']])
                            gc.collect()

                    elif st.session_state.get('survival_data') is not None:
                        surv_df = st.session_state['survival_data']
                        if 'Overall survival' in surv_df.columns and 'State' in surv_df.columns:
                            st.markdown("*Survival Data*")
                            st.dataframe(surv_df)
                            del surv_df
                            gc.collect()




            # ════════════════════════════════════════════════════════════════════════
            # GROUP 2 — CLEAN & EDIT   (Dataset Inspector · Rename · Edit)
            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═            # ═
            with _grp_clean:
                st.markdown(_picon("edit","View data, rename classes, edit columns/rows, then export your cleaned dataset.", "#10b981"), unsafe_allow_html=True)
                _t1_save, _t1_rename, _t1_edit = st.tabs([
                    "  📊 Dataset Inspector", "  ✏️ Rename Classes", "  🛠️ Edit Dataset"
                ])

                # ── TAB 1: DATASET INSPECTOR ────────────────────────────────────────
                with _t1_save:
                    st.markdown(
                        '<p style="color: gray; font-size: 14px">Preview your dataset, inspect column metadata, then download.</p>',
                        unsafe_allow_html=True
                    )
                    import csv as _csv_mod
                    _save_options = {
                        "Raw / Edited": st.session_state.get("final_data", st.session_state.get("data")),
                        "Preprocessed": st.session_state.get("preprocessed_data"),
                        "Oversampled": st.session_state.get("oversampled_data"),
                        "Undersampled": st.session_state.get("undersampled_data"),
                    }
                    _save_options = {k: v for k, v in _save_options.items() if v is not None}
                    if not _save_options:
                        st.info("⚠️ No dataset available yet. Load or process data first.")
                    else:
                        _save_choice = st.selectbox("📂 Select dataset to inspect:", list(_save_options.keys()), key="save_tab_source")
                        _save_df = _save_options[_save_choice]
                        _n_rows, _n_cols = _save_df.shape
                        _n_classes = _save_df["Class"].nunique() if "Class" in _save_df.columns else "N/A"
                        _n_missing = int(_save_df.isnull().sum().sum())
                        _bann1, _bann2, _bann3, _bann4 = st.columns(4)
                        _bann1.metric("Samples", f"{_n_rows:,}")
                        _bann2.metric("Features", f"{_n_cols:,}")
                        _bann3.metric("Classes", _n_classes)
                        st.markdown("**Data Table**")
                        if _save_df.shape[1] > 100:
                            st.dataframe(pd.concat([_save_df.iloc[:, :50], _save_df.iloc[:, -50:]], axis=1), use_container_width=True)
                        else:
                            st.dataframe(_save_df, use_container_width=True)
                        st.markdown("**Column Metadata**")
                        _meta_rows = []
                        for _col in _save_df.columns:
                            _s = _save_df[_col]
                            _miss = int(_s.isnull().sum())
                            _miss_pct = round(_miss / _n_rows * 100, 2) if _n_rows > 0 else 0.0
                            _nuniq = int(_s.nunique())
                            if pd.api.types.is_numeric_dtype(_s):
                                _mn = round(float(_s.mean(skipna=True)), 4) if _miss < _n_rows else "N/A"
                                _std = round(float(_s.std(skipna=True)), 4) if _miss < _n_rows else "N/A"
                                _min_ = round(float(_s.min(skipna=True)), 4) if _miss < _n_rows else "N/A"
                                _max_ = round(float(_s.max(skipna=True)), 4) if _miss < _n_rows else "N/A"
                            else:
                                _mn = _std = _min_ = _max_ = "—"
                            _meta_rows.append({"Column": _col, "Type": str(_s.dtype),
                                "Missing": _miss, "Missing (%)": _miss_pct, "Unique values": _nuniq,
                                "Mean": _mn, "Std": _std, "Min": _min_, "Max": _max_})
                        st.dataframe(pd.DataFrame(_meta_rows), use_container_width=True, hide_index=True)
                        st.markdown("**Export**")
                        _save_fname = st.text_input("📄 Filename:", value=f"{_save_choice.replace('/', '_').replace(' ', '_')}.csv", key="save_tab_fname")
                        if not _save_fname.lower().endswith(".csv"):
                            _save_fname += ".csv"
                        _save_csv = _save_df.to_csv(index=False, sep=';', quoting=_csv_mod.QUOTE_NONNUMERIC, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Download Dataset (CSV)",
                            data=_save_csv.encode("utf-8-sig"),
                            file_name=_save_fname, mime="text/csv", key="save_tab_dl_btn"
                        )

                # ── TAB 2: RENAME CLASSES ────────────────────────────────────────
                with _t1_rename:
                    st.markdown(
                        '<p style="color: gray; font-size: 14px">Standardize class labels: unify replicates, merge groups, or relabel unknowns.</p>',
                        unsafe_allow_html=True
                    )
                    if "data" not in st.session_state or st.session_state["data"] is None:
                        st.warning("⚠️ No data available. Please upload or load a dataset first.")
                    else:
                        if "final_data" not in st.session_state:
                            st.session_state["final_data"] = st.session_state["data"].copy()
                
                        class_names = list(st.session_state["final_data"]["Class"].unique())
                        st.session_state.setdefault("class_renaming", {cls: cls for cls in class_names})
                        st.session_state.setdefault("rename_pending", {cls: cls for cls in class_names})
                
                        mode = st.radio(
                            "🔧 Select renaming mode:",
                            [
                                "🔁 Apply same name to all classes",
                                "✏️ Rename each class individually",
                                "🧩 Group classes under a common name"
                            ],
                            key="rename_mode",
                            help="Choose whether to apply a global name, rename individually, or group multiple classes."
                        )
                
                        # MODE 1 & 2: Avec formulaire (pas de problème de rafraîchissement)
                        if not mode.startswith("🧩"):
                            with st.form(key="class_renaming_form"):
                                temp_mapping = st.session_state["rename_pending"].copy()
                        
                                if mode.startswith("🔁"):
                                    global_name = st.text_input("🆕 New name for all classes:", "UnifiedClass", key="global_class_name")
                                    if global_name:
                                        temp_mapping = {cls: global_name for cls in class_names}
                        
                                elif mode.startswith("✏️"):
                                    for cls in class_names:
                                        new_name = st.text_input(
                                            f"Rename '**{cls}**' to:",
                                            st.session_state["rename_pending"].get(cls, cls),
                                            key=f"rename_{cls}"
                                        )
                                        temp_mapping[cls] = new_name
                        
                                col1, col2 = st.columns(2)
                                with col1:
                                    apply_changes = st.form_submit_button("Apply Renaming")
                                with col2:
                                    reset_changes = st.form_submit_button("🔄 Reset Changes")
                    
                            if apply_changes:
                                with st.spinner("Applying class name changes..."):
                                    st.session_state["rename_pending"] = temp_mapping.copy()
                                    st.session_state["class_renaming"] = st.session_state["rename_pending"].copy()
                                    st.session_state["final_data"]["Class"] = st.session_state["final_data"]["Class"].replace(
                                        st.session_state["class_renaming"]
                                    )
                                    st.session_state["data"] = st.session_state["final_data"]
                                    # Propagate to all derived datasets
                                    _rmap = st.session_state["class_renaming"]
                                    for _dk in ("preprocessed_data", "oversampled_data", "undersampled_data"):
                                        _ddf = st.session_state.get(_dk)
                                        if _ddf is not None and "Class" in _ddf.columns:
                                            st.session_state[_dk] = _ddf.copy()
                                            st.session_state[_dk]["Class"] = st.session_state[_dk]["Class"].replace(_rmap)
                                    # Re-sync DataLab cache
                                    _cur_src = st.session_state.get('datalab_source', 'Raw / Edited')
                                    _sync_map = {
                                        'Raw / Edited': st.session_state.get('final_data', st.session_state.get('data')),
                                        'Preprocessed': st.session_state.get('preprocessed_data'),
                                        'Oversampled':  st.session_state.get('oversampled_data'),
                                        'Undersampled': st.session_state.get('undersampled_data'),
                                    }
                                    _sync_df = _sync_map.get(_cur_src)
                                    if _sync_df is not None:
                                        st.session_state['datalab_df']  = _sync_df.copy()
                                        st.session_state['overview_df'] = _sync_df.copy()
                                    st.success("Class names updated successfully!")
                                    st.rerun()
                    
                            if reset_changes:
                                st.session_state["rename_pending"] = {cls: cls for cls in class_names}
                                st.session_state["class_renaming"] = {cls: cls for cls in class_names}
                                st.session_state["final_data"]["Class"] = st.session_state["final_data"]["Class"].map(
                                    lambda x: x if x in class_names else x
                                )
                                st.session_state["data"] = st.session_state["final_data"]
                                # Re-sync DataLab cache
                                st.session_state['datalab_df']  = st.session_state["final_data"].copy()
                                st.session_state['overview_df'] = st.session_state["final_data"].copy()
                                st.success("🗑️ All renaming changes have been reset.")
                                gc.collect()
                                st.rerun()
                



                        # MODE 3: Sans formulaire pour permettre le rafraîchissement dynamique
                        else:
                            # Initialiser le mapping temporaire pour les groupes
                            if "group_temp_mapping" not in st.session_state:
                                st.session_state["group_temp_mapping"] = {}

                            n_groups = st.number_input("Number of class groups:", 1, 100, 1, key="num_groups")

                            st.markdown("---")

                            # Afficher les groupes
                            for i in range(n_groups):
                                st.markdown(f"**📦 Group** {i+1}")

                                # Champ pour le nom du groupe
                                gname = st.text_input(
                                    f"Group name:",
                                    value=st.session_state.get(f"group_name_{i}", ""),
                                    key=f"group_name_{i}",
                                    placeholder=f"Enter name for group {i+1}"
                                )

                                # Identifier les classes déjà sélectionnées pour CE groupe spécifique
                                # On cherche par l'index du groupe (stocké sous group_idx_{i}) plutôt que par gname
                                # pour éviter les décalages quand gname change
                                _group_key = f"_group_idx_{i}"
                                current_group_classes = [
                                    cls for cls, grp_idx in st.session_state.get("group_temp_mapping_idx", {}).items()
                                    if grp_idx == i
                                ]

                                # Recalculer les classes assignées MAINTENANT (après les inputs précédents)
                                assigned_classes = set(st.session_state["group_temp_mapping"].keys())

                                # Classes disponibles = toutes les classes NON assignées + celles de ce groupe
                                available_classes = [cls for cls in class_names if cls not in assigned_classes or cls in current_group_classes]
                                available_classes.sort()

                                # Sélection des classes (en dessous du champ "Group name")
                                selected = st.multiselect(
                                    f"Select classes:",
                                    available_classes,
                                    default=current_group_classes,
                                    key=f"selected_classes_{i}",
                                    help=f"Available: {len(available_classes)} classes"
                                )

                                # Mettre à jour le mapping temporaire (indexé par position du groupe)
                                if "group_temp_mapping_idx" not in st.session_state:
                                    st.session_state["group_temp_mapping_idx"] = {}

                                # Retirer les anciennes assignations de ce groupe
                                for cls in current_group_classes:
                                    if cls not in selected:
                                        st.session_state["group_temp_mapping"].pop(cls, None)
                                        st.session_state["group_temp_mapping_idx"].pop(cls, None)

                                # Mettre à jour avec la sélection courante
                                if gname and selected:
                                    for cls in selected:
                                        st.session_state["group_temp_mapping"][cls] = gname
                                        st.session_state["group_temp_mapping_idx"][cls] = i
                                elif not gname and selected:
                                    # Nom de groupe pas encore renseigné: stocker l'index en attendant
                                    for cls in selected:
                                        st.session_state["group_temp_mapping_idx"][cls] = i
                                        # On ne met pas dans group_temp_mapping sans nom
                                elif not selected:
                                    # Si plus de sélection, retirer ce groupe du mapping
                                    for cls in list(st.session_state["group_temp_mapping_idx"].keys()):
                                        if st.session_state["group_temp_mapping_idx"].get(cls) == i:
                                            st.session_state["group_temp_mapping"].pop(cls, None)
                                            st.session_state["group_temp_mapping_idx"].pop(cls, None)

                                st.markdown("---")

                            # Résumé
                            assigned_count = len(st.session_state["group_temp_mapping"])
                            remaining_classes = set(class_names) - set(st.session_state["group_temp_mapping"].keys())

                            if assigned_count > 0:
                                if remaining_classes:
                                    st.info(f"📊 **Summary:** {assigned_count} classes assigned to groups, {len(remaining_classes)} remaining: {', '.join(sorted(remaining_classes))}")
                                else:
                                    st.success(f"✅ **Summary:** All {len(class_names)} classes have been assigned to groups!")

                            # Afficher le message persistant de succès/reset s'il y en a un
                            if st.session_state.get("_group_apply_success"):
                                st.success(st.session_state.pop("_group_apply_success"))
                            if st.session_state.get("_group_reset_success"):
                                st.success(st.session_state.pop("_group_reset_success"))

                            # Boutons d'action
                            col1, col2 = st.columns(2)
                    
                            with col1:
                                if st.button("Apply Grouping", type="primary", use_container_width=True):
                                    if st.session_state["group_temp_mapping"]:
                                        with st.spinner("Applying class grouping..."):
                                            # Appliquer le mapping
                                            st.session_state["rename_pending"] = st.session_state["group_temp_mapping"].copy()
                                            st.session_state["class_renaming"] = st.session_state["rename_pending"].copy()
                                            st.session_state["final_data"]["Class"] = st.session_state["final_data"]["Class"].replace(
                                                st.session_state["class_renaming"]
                                            )
                                            st.session_state["data"] = st.session_state["final_data"]
                                            # Propagate to all derived datasets
                                            _rmap = st.session_state["class_renaming"]
                                            for _dk in ("preprocessed_data", "oversampled_data", "undersampled_data"):
                                                _ddf = st.session_state.get(_dk)
                                                if _ddf is not None and "Class" in _ddf.columns:
                                                    st.session_state[_dk] = _ddf.copy()
                                                    st.session_state[_dk]["Class"] = st.session_state[_dk]["Class"].replace(_rmap)

                                            # Nettoyer les états temporaires
                                            st.session_state["group_temp_mapping"] = {}
                                            st.session_state["group_temp_mapping_idx"] = {}
                                            for i in range(n_groups):
                                                st.session_state.pop(f"selected_classes_{i}", None)
                                                st.session_state.pop(f"group_name_{i}", None)
                                            # Re-sync DataLab cache
                                            _cur_src_g = st.session_state.get('datalab_source', 'Raw / Edited')
                                            _sync_g = {
                                                'Raw / Edited': st.session_state.get('final_data', st.session_state.get('data')),
                                                'Preprocessed': st.session_state.get('preprocessed_data'),
                                                'Oversampled':  st.session_state.get('oversampled_data'),
                                                'Undersampled': st.session_state.get('undersampled_data'),
                                            }.get(_cur_src_g)
                                            if _sync_g is not None:
                                                st.session_state['datalab_df']  = _sync_g.copy()
                                                st.session_state['overview_df'] = _sync_g.copy()
                                            # Message persistant (survit au rerun)
                                            st.session_state["_group_apply_success"] = "✅ Class grouping applied successfully!"
                                            st.rerun()
                                    else:
                                        st.warning("⚠️ No classes have been assigned to groups yet.")
                    
                            with col2:
                                if st.button("🔄 Reset Changes", use_container_width=True):
                                    # Reset complet vers les classes originales
                                    original_classes = list(st.session_state["data"]["Class"].unique())
                                    st.session_state["rename_pending"] = {cls: cls for cls in original_classes}
                                    st.session_state["class_renaming"] = {cls: cls for cls in original_classes}
                                    st.session_state["final_data"] = st.session_state["data"].copy()
                                    st.session_state["group_temp_mapping"] = {}
                                    st.session_state["group_temp_mapping_idx"] = {}
                            
                                    # Nettoyer tous les états de groupes
                                    for i in range(n_groups):
                                        st.session_state.pop(f"selected_classes_{i}", None)
                                        st.session_state.pop(f"group_name_{i}", None)
                            
                                    # Re-sync DataLab cache after group reset
                                    st.session_state['datalab_df']  = st.session_state["final_data"].copy()
                                    st.session_state['overview_df'] = st.session_state["final_data"].copy()
                                    # Message persistant (survit au rerun)
                                    st.session_state["_group_reset_success"] = "🗑️ All changes have been reset to original classes."
                                    gc.collect()
                                    st.rerun()



                with _t1_edit:
                    st.markdown(
                        '<p style="color: gray; font-size: 14px">Rename columns, remove or keep specific rows/columns/classes from the dataset.</p>',
                        unsafe_allow_html=True
                    )

                    if "final_data" in st.session_state:
                        df = st.session_state["final_data"].copy()

                        # ── Rename Columns ─────────────────────────────────────────────────
                        st.markdown("##### 📝 Rename Columns")
                        _renamable_cols = [c for c in df.columns if c != "Class"]
                        with st.expander("Rename one or more columns", expanded=False):
                            _col_to_rename = st.selectbox("Select column to rename:", _renamable_cols, key="col_rename_select")
                            _col_new_name = st.text_input("New column name:", value=_col_to_rename, key="col_rename_new_name")
                            if st.button("✅ Apply Column Rename", key="apply_col_rename_btn"):
                                if _col_new_name and _col_new_name != _col_to_rename:
                                    if _col_new_name in df.columns:
                                        st.error(f"Column '{_col_new_name}' already exists.")
                                    else:
                                        _renamed_df = st.session_state["final_data"].rename(columns={_col_to_rename: _col_new_name})
                                        st.session_state["final_data"] = _renamed_df
                                        st.session_state["data"] = _renamed_df.copy()
                                        for _dk in ("preprocessed_data", "oversampled_data", "undersampled_data"):
                                            _ddf = st.session_state.get(_dk)
                                            if _ddf is not None and _col_to_rename in _ddf.columns:
                                                st.session_state[_dk] = _ddf.rename(columns={_col_to_rename: _col_new_name})
                                        # Re-sync DataLab cache
                                        _cur_src_r = st.session_state.get('datalab_source', 'Raw / Edited')
                                        _sync_rn = {
                                            'Raw / Edited': _renamed_df,
                                            'Preprocessed': st.session_state.get('preprocessed_data'),
                                            'Oversampled':  st.session_state.get('oversampled_data'),
                                            'Undersampled': st.session_state.get('undersampled_data'),
                                        }.get(_cur_src_r, _renamed_df)
                                        if _sync_rn is not None:
                                            st.session_state['datalab_df']  = _sync_rn.copy()
                                            st.session_state['overview_df'] = _sync_rn.copy()
                                        st.success(f"Column '{_col_to_rename}' renamed to '{_col_new_name}'.")
                                        st.rerun()
                                else:
                                    st.warning("Please enter a different name.")

                        st.markdown("---")
                        st.markdown("##### ✂️ Remove / Keep Rows & Columns")

                        # ✅ Créer un index de position (0..N-1) pour l'affichage — indépendant de l'index pandas
                        _df_for_display = df.reset_index(drop=True)
                        display_df = _df_for_display.copy()
                        display_df.insert(0, 'Display_Index', range(len(display_df)))

                        # Pour l'affichage dans les multiselect
                        label_df = display_df[["Display_Index", "Class"]]
                        row_options = list(label_df.apply(lambda row: f"Index {row['Display_Index']} → {row['Class']}", axis=1))

                        with st.form(key="edit_dataset_form"):
                            # --- Sélection des rows à supprimer ---
                            selected_rows = st.multiselect("Rows to remove:", row_options, key="selected_rows_to_remove")
                            selected_indexes = [int(row.split()[1]) for row in selected_rows] if selected_rows else []
                    
                            # --- Colonnes disponibles à supprimer (protéger Class uniquement) ---
                            columns_available_to_remove = [col for col in df.columns if col != "Class"]
                            selected_columns = st.multiselect(
                                "Columns to remove:", columns_available_to_remove, key="selected_columns_to_remove"
                            )
                    
                            # --- Classes à supprimer ---
                            unique_classes = sorted(df["Class"].dropna().unique().tolist())
                            selected_classes_to_remove = st.multiselect(
                                "Remove all samples belonging to the following Class(es):", unique_classes, key="classes_to_remove"
                            )
                    
                            # --- Rows à garder ---
                            rows_to_keep = st.multiselect("Rows to keep:", row_options, key="rows_to_keep")
                            rows_keep_indexes = [int(row.split()[1]) for row in rows_to_keep] if rows_to_keep else []
                    
                            # --- Colonnes à garder ---
                            columns_to_keep = st.multiselect(
                                "Columns to keep:", list(df.columns), key="columns_to_keep"
                            )
                    
                            col1, col2 = st.columns(2)
                            with col1:
                                apply_changes = st.form_submit_button("Apply Changes")
                            with col2:
                                reset_changes = st.form_submit_button("🔄 Reset All Changes")
                
                        # ------------------------- APPLY CHANGES -------------------------
                        if apply_changes:
                            with st.spinner("Applying modifications..."):
                                # Toujours travailler sur une copie à index positionnel contigu
                                modified_df = st.session_state["final_data"].reset_index(drop=True).copy()
                        
                                # Keep rows first (sélection positionnelle)
                                if rows_keep_indexes:
                                    valid_keep = [i for i in rows_keep_indexes if i < len(modified_df)]
                                    modified_df = modified_df.iloc[valid_keep].reset_index(drop=True)
                        
                                # Keep columns (force Class retention)
                                if columns_to_keep:
                                    if "Class" not in columns_to_keep:
                                        columns_to_keep.append("Class")
                                    modified_df = modified_df[[c for c in columns_to_keep if c in modified_df.columns]]
                        
                                # Remove selected rows by position
                                if selected_indexes:
                                    valid_drop = [i for i in selected_indexes if i < len(modified_df)]
                                    modified_df = modified_df.drop(index=valid_drop, errors='ignore').reset_index(drop=True)
                        
                                # Remove selected columns (Class protected)
                                if selected_columns:
                                    columns_to_drop = [col for col in selected_columns if col != "Class"]
                                    modified_df = modified_df.drop(columns=columns_to_drop, errors='ignore')
                        
                                # Remove selected class samples
                                if selected_classes_to_remove:
                                    modified_df = modified_df[~modified_df["Class"].isin(selected_classes_to_remove)]
                        
                                # ✅ Reset index proprement
                                modified_df.reset_index(drop=True, inplace=True)
                        
                                st.session_state["final_data"] = modified_df
                                st.session_state["data"] = modified_df.copy()

                                # ── Propager les changements vers les datasets dérivés ──────────────
                                _remaining_cols = list(modified_df.columns)
                                _remaining_classes = set(modified_df["Class"].dropna().unique()) if "Class" in modified_df.columns else None

                                for _dk in ("preprocessed_data", "oversampled_data", "undersampled_data"):
                                    _ddf = st.session_state.get(_dk)
                                    if _ddf is None:
                                        continue
                                    _ddf2 = _ddf.copy()

                                    # Supprimer les classes retirées
                                    if selected_classes_to_remove and "Class" in _ddf2.columns:
                                        _ddf2 = _ddf2[~_ddf2["Class"].isin(selected_classes_to_remove)]

                                    # Supprimer les colonnes retirées (sauf Class)
                                    if selected_columns:
                                        _cols_drop = [c for c in selected_columns if c != "Class" and c in _ddf2.columns]
                                        if _cols_drop:
                                            _ddf2 = _ddf2.drop(columns=_cols_drop, errors='ignore')

                                    # Garder seulement les colonnes présentes dans modified_df (intersection)
                                    if columns_to_keep:
                                        _keep_derived = [c for c in _remaining_cols if c in _ddf2.columns]
                                        if _keep_derived:
                                            _ddf2 = _ddf2[_keep_derived]

                                    _ddf2.reset_index(drop=True, inplace=True)
                                    st.session_state[_dk] = _ddf2

                                # ── Re-sync DataLab cache ──────────────────────────────────────
                                _cur_src_e = st.session_state.get('datalab_source', 'Raw / Edited')
                                _sync_e = {
                                    'Raw / Edited': modified_df,
                                    'Preprocessed': st.session_state.get('preprocessed_data'),
                                    'Oversampled':  st.session_state.get('oversampled_data'),
                                    'Undersampled': st.session_state.get('undersampled_data'),
                                }.get(_cur_src_e, modified_df)
                                if _sync_e is not None:
                                    st.session_state['datalab_df']  = _sync_e.copy()
                                    st.session_state['overview_df'] = _sync_e.copy()

                                st.success("✅ Modifications applied successfully.")
                                st.rerun()
                
                        # ------------------------- RESET -------------------------
                        if reset_changes:
                            original = st.session_state["data"].copy()
                            st.session_state["final_data"] = original
                            st.session_state['datalab_df']  = original.copy()
                            st.session_state['overview_df'] = original.copy()
                            st.success("🔁 Dataset has been restored to its original state.")
                            st.rerun()
                


            # GROUP 3 — PROCESS   (Preprocessing · Post-QC)
            # ════════════════════════════════════════════════════════════════════════
            with _grp_process:
                st.markdown(_picon("preprocess","Filter, impute, normalize and batch-correct — then verify readiness.", "#f59e0b"), unsafe_allow_html=True)
                _t1_preprocess, _t1_postqc = st.tabs([
                    "  Preprocessing", "  Post-Preprocessing QC"
                ])
                with _t1_preprocess:
                    st.markdown(
                        '<p style="color: gray; font-size: 14px">Filtering, Imputation, Binning, Normalization,Batch Effect Correction and sparse matrix handling</p>',
                        unsafe_allow_html=True
                    )

                    df_overview = st.session_state.get('overview_df')
                    submitted = False
                    if df_overview is None or df_overview.empty:
                        st.warning("⚠️ You must load a dataset in **Data Overview** before running preprocessing steps.")
                    else:
                        # base copy
                        data_to_preprocess = df_overview.copy()

                        apply_binning_option = st.checkbox(
                            "Shrink mass range or apply Binning?",
                            key="apply_binning_option",
                            help="Only relevant for ion spectra (MS1 data)."
                        )

                        # helper utilities local to this block
                        # cols_exclude built dynamically — preserves _meta + ALL non-numeric columns
                        _BASE_EXCLUDE = {'Class', 'File', 'RT', 'Sum', 'Original_Index', 'ID', 'Original_index'}

                        def get_cols_exclude(df):
                            """Return all columns that are NOT numeric features (metadata + categoricals)."""
                            exclude = set(_BASE_EXCLUDE)
                            exclude.update(c for c in df.columns if str(c).endswith('_meta'))
                            exclude.update(c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]))
                            return [c for c in df.columns if c in exclude]

                        cols_exclude = get_cols_exclude(data_to_preprocess)

                        def get_numeric_features(df):
                            _exc = set(get_cols_exclude(df))
                            return [c for c in df.columns if c not in _exc and pd.api.types.is_numeric_dtype(df[c])]

                        def shifted_gaussian_fill(series, shift=1.8, width=0.3, rng=None):
                            vals = series.dropna()
                            if vals.empty:
                                return series
                            mu, sigma = vals.mean(), vals.std(ddof=0)
                            n_missing = series.isna().sum()
                            if n_missing == 0:
                                return series
                            if rng is None:
                                rng = np.random.default_rng()
                            filled = rng.normal(loc=mu - shift * sigma, scale=width * sigma, size=n_missing)
                            out = series.copy()
                            out.loc[out.isna()] = filled
                            return out

                        with st.form("preprocessing_form"):

                            # ------------------ Filters ------------------
                            min_detection_threshold = st.number_input(
                                "Minimum features detection rate per class (%)",
                                min_value=0,
                                max_value=100,
                                value=0,
                                step=1,
                                help="Keep features detected in at least X% of samples per class. "
                                    "If zero-inflated filter is applied, zeros are considered as missing values."
                            )

                            apply_zero_inflated_filter = st.checkbox(
                                "Apply zero-inflated feature filter (sparse matrix)?",
                                value=False,
                                help="Treat zeros as missing values for filtering features that are mostly zeros and choose Detele Missing Values method "
                                    "(useful for RNA-seq / transcriptomics and MS1 data)."
                            )

                            # ------------------ Imputation ------------------
                            imputation_method = st.selectbox(
                                "Select Missing Value Imputation Method",
                                [
                                    'None', 'Mean Imputation', 'Median Imputation', 'Mode Imputation',
                                    'Delete Missing Values', 'KNN Imputation', 'Fillna with 0', 'Shifted Gaussian'
                                ],
                                key="imputation_method"
                            )

                            impute_by_class = st.checkbox(
                                "🔹 Impute missing values per class?",
                                value=False,
                                help="Perform imputation separately within each Class."
                            )

                            remove_exclusive_missing = st.checkbox(
                                "🔹Remove features that remain entirely missing in any class after class-wise imputation?",
                                value=False,
                                help="Removes features that are still all NaN in at least one class after imputation per Class."
                            )

                            # ------------------ Binning ------------------
                            bin_width = mass_range_min = mass_range_max = None
                            if apply_binning_option:
                                mz_cols = [col for col in data_to_preprocess.columns if col not in cols_exclude]
                                mz_values = []
                                for col in mz_cols:
                                    try:
                                        mz_values.append(float(str(col).replace("mz_", "").strip()))
                                    except:
                                        continue
                                if mz_values:
                                    min_detected, max_detected = min(mz_values), max(mz_values)
                                    st.info(f"Detected m/z range: {min_detected:.4f} - {max_detected:.4f} Da")
                                    bin_width = st.number_input("Bin Width (Da)", 0.0001, 100.0, 0.1, 0.1)
                                    mass_range_min = st.number_input("Min Mass Range", value=min_detected, format="%.4f")
                                    mass_range_max = st.number_input("Max Mass Range", value=max_detected, format="%.4f")
                                else:
                                    st.warning("No valid m/z columns found for binning.")
                                    bin_width, mass_range_min, mass_range_max = 0.1, 0.0, 0.0

                            # ------------------ Normalization ------------------
                            normalization_type = st.selectbox(
                                "Normalization Type",
                                ['None', 'Log2', 'RMS', 'BasePeak', 'QNorm', 'Log1p', 'Log10', 'Median of Ratios (Deseq2-like)', 'TMM', 'CPM', 'logCPM', 'VST', 'Total Intensity', 'Median', 'Mean'],
                                key="normalization_type"
                            )

                            # debug / performance options
                            debug_mode = st.checkbox("Show debug logs?", value=False)
                            knn_k = st.number_input("K for KNN (if selected)", min_value=1, max_value=50, value=5, step=1,help="K defines how many nearest neighbours are used to generate each new imputed value. By default, K=5 means the missing value is estimated from the 5 closest samples. Because this method relies on neighbour structure, results may slightly vary and will never be strictly identical. Increase K for a more smoother imputation, or decrease it for more local sensitivity.")

                            submitted = st.form_submit_button("Preprocess Data")

                        # ── Combat batch correction — OUTSIDE form for live interactivity ──
                        _combat_col_options = ["Class"] + [
                            c for c in data_to_preprocess.columns
                            if str(c).endswith("_meta")
                        ]
                        apply_combat = st.checkbox(
                            "Apply batch effect correction (Combat)?",
                            key="apply_combat_cb",
                            help=(
                                "Apply Combat batch effect correction. "
                                "Choose which column defines your batches below.\n"
                                "You can use 'Class' or any metadata column ending in '_meta' "
                                "(e.g., batch_meta, site_meta, operator_meta).\n"
                                "The selected column must have at least 2 distinct values."
                            )
                        )
                        if apply_combat:
                            combat_batch_col = st.selectbox(
                                "Batch column for Combat",
                                _combat_col_options,
                                index=0,
                                key="combat_batch_col",
                                help=(
                                    "• **Class** — use the main class labels as batches (default)\n"
                                    "• **_meta columns** — use a clinical/technical variable as batches "
                                    "(e.g., acquisition_batch_meta, site_meta, scanner_meta)"
                                ),
                            )
                            if len(_combat_col_options) > 1:
                                st.caption(
                                    f"📋 {len(_combat_col_options) - 1} metadata column(s) available: "
                                    + ", ".join(f"`{c}`" for c in _combat_col_options[1:])
                                )

                        if submitted:
                            try:
                                data = data_to_preprocess.copy()
                                progress = st.progress(0)

                                # Stats for summary
                                removed_by_detection = 0
                                removed_by_exclusive_missing = 0

                                # ------------------ Detection Filter ------------------
                                try:
                                    numeric_cols = get_numeric_features(data)
                                    if min_detection_threshold > 0 and 'Class' in data.columns and numeric_cols:
                                        keep_features = []
                                        for col in numeric_cols:
                                            keep_col = True
                                            for _, group in data.groupby('Class'):
                                                values = group[col]
                                                # If zero-inflated => zero counts as missing
                                                if apply_zero_inflated_filter:
                                                    present = ((values.notna()) & (values != 0)).sum()
                                                else:
                                                    present = values.notna().sum()
                                                present_pct = present / len(values) * 100
                                                if present_pct < min_detection_threshold:
                                                    keep_col = False
                                                    # if debug_mode:
                                                    #     st.write(f"Drop {col}: {present_pct:.1f}% < {min_detection_threshold}% in a class")
                                                    # break
                                            if keep_col:
                                                keep_features.append(col)

                                        removed_by_detection = len(numeric_cols) - len(keep_features)
                                        if removed_by_detection > 0:
                                            st.warning(f"{removed_by_detection} features removed due to threshold per class.")
                                        # keep at least cols_exclude; if no feature remains, keep only metadata
                                        cols_exclude = get_cols_exclude(data)  # recompute: captures _meta + categoricals
                                        kept_cols = cols_exclude + keep_features
                                        kept_cols = [c for c in kept_cols if c in data.columns]
                                        if keep_features:
                                            data = data[kept_cols].copy()
                                        else:
                                            # keep only metadata to avoid empty dataframes
                                            data = data[[c for c in cols_exclude if c in data.columns]].copy()
                                except Exception as e:
                                    st.error(f"Error in detection filter: {e}")
                                    st.stop()
                                else:
                                    progress.progress(20)

                                # ------------------ Imputation ------------------
                                try:
                                    numeric_cols = get_numeric_features(data)  # recalc after filtering
                                    if imputation_method != 'None' and numeric_cols:
                                        st.info(f"Applying {imputation_method}...")
                                        rng = np.random.default_rng()

                                        if imputation_method in ['Mean Imputation', 'Median Imputation']:
                                            _agg = 'mean' if imputation_method == 'Mean Imputation' else 'median'
                                            if impute_by_class and 'Class' in data.columns:
                                                # Vectorized: groupby.transform fills per class in one pass
                                                data[numeric_cols] = data[numeric_cols].fillna(
                                                    data.groupby('Class')[numeric_cols].transform(_agg)
                                                )
                                            else:
                                                data[numeric_cols] = data[numeric_cols].fillna(
                                                    data[numeric_cols].agg(_agg)
                                                )

                                        elif imputation_method == 'Mode Imputation':
                                            if impute_by_class and 'Class' in data.columns:
                                                # Vectorized per-class mode fill
                                                data[numeric_cols] = data[numeric_cols].fillna(
                                                    data.groupby('Class')[numeric_cols].transform(
                                                        lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s
                                                    )
                                                )
                                            else:
                                                _modes = {c: data[c].mode().iloc[0] for c in numeric_cols if not data[c].mode().empty}
                                                data[numeric_cols] = data[numeric_cols].fillna(_modes)

                                        elif imputation_method == 'Delete Missing Values':
                                            # delete columns that contain ANY NaN
                                            data = data.dropna(axis=1)
                                        elif imputation_method == 'KNN Imputation':
                                            from sklearn.impute import KNNImputer
                                            numeric_cols = get_numeric_features(data)
                                            if not numeric_cols:
                                                # nothing to impute
                                                pass
                                            else:
                                                if impute_by_class and 'Class' in data.columns:
                                                    frames = []
                                                    for cls, grp in data.groupby('Class'):
                                                        grp_numeric = grp[numeric_cols].copy()
                                                        if len(grp_numeric) < max(1, int(knn_k)):
                                                            st.error(f"Class '{cls}' has fewer than {knn_k} samples for KNN. Aborting.")
                                                            st.stop()
                                                        # keep columns that have at least one non-NaN otherwise KNNImputer can't work on a constant-empty column
                                                        valid_cols = grp_numeric.columns[grp_numeric.notna().any()]
                                                        if len(valid_cols) == 0:
                                                            # nothing to impute in this class, keep as-is
                                                            frames.append(grp)
                                                            continue
                                                        imputer = KNNImputer(n_neighbors=min(int(knn_k), len(grp_numeric) - 1), weights="uniform")
                                                        transformed = imputer.fit_transform(grp_numeric[valid_cols])
                                                        grp.loc[:, valid_cols] = pd.DataFrame(transformed, columns=valid_cols, index=grp.index)
                                                        frames.append(grp)
                                                    data = pd.concat(frames)
                                                else:
                                                    if len(data) < max(1, int(knn_k)):
                                                        st.error(f"Dataset has fewer than {knn_k} samples for KNN. Aborting.")
                                                        st.stop()
                                                    grp_numeric = data[numeric_cols].copy()
                                                    valid_cols = grp_numeric.columns[grp_numeric.notna().any()]
                                                    if len(valid_cols) > 0:
                                                        imputer = KNNImputer(n_neighbors=min(int(knn_k), len(data) - 1), weights="uniform")
                                                        data.loc[:, valid_cols] = pd.DataFrame(imputer.fit_transform(grp_numeric[valid_cols]), columns=valid_cols, index=data.index)

                                        elif imputation_method == 'Fillna with 0':
                                            data[numeric_cols] = data[numeric_cols].fillna(0)

                                        elif imputation_method == 'Shifted Gaussian':
                                            numeric_cols = get_numeric_features(data)
                                            if impute_by_class and 'Class' in data.columns:
                                                parts = []
                                                for _, grp in data.groupby('Class'):
                                                    grp2 = grp.copy()
                                                    for c in numeric_cols:
                                                        grp2[c] = shifted_gaussian_fill(grp2[c], rng=rng)
                                                    parts.append(grp2)
                                                data = pd.concat(parts)
                                            else:
                                                for c in numeric_cols:
                                                    data[c] = shifted_gaussian_fill(data[c], rng=rng)

                                        else:
                                            st.error(f"Unknown imputation method: {imputation_method}")
                                    # if imputation method is None or no numeric cols, nothing to do
                                except Exception as e:
                                    st.error(f"Error during imputation: {e}")
                                    st.stop()
                                else:
                                    progress.progress(40)
                                    st.success("✅ Imputation completed")

                                # ------------------ Remove features still entirely missing in some class ------------------
                                try:
                                    # recalc numeric columns after imputation / drops
                                    numeric_cols = get_numeric_features(data)
                                    if remove_exclusive_missing and numeric_cols:
                                        to_drop = []
                                        if impute_by_class and 'Class' in data.columns:
                                            for col in numeric_cols:
                                                # if any class has all NaN for this column -> drop it
                                                if any(group[col].isna().all() for _, group in data.groupby('Class')):
                                                    to_drop.append(col)
                                        else:
                                            to_drop = [col for col in numeric_cols if data[col].isna().all()]

                                        if to_drop:
                                            data.drop(columns=to_drop, inplace=True)
                                            removed_by_exclusive_missing = len(to_drop)
                                            if debug_mode:
                                                st.write(f"Removed exclusive-missing features: {to_drop}")
                                except Exception as e:
                                    st.error(f"Error removing exclusive missing features: {e}")
                                    st.stop()
                                else:
                                    progress.progress(60)

                                # Final strict validation: no NaN allowed in numeric features
                                try:
                                    numeric_cols_final = get_numeric_features(data)
                                    total_remaining_nans = int(data[numeric_cols_final].isna().sum().sum()) if numeric_cols_final else 0
                                    if total_remaining_nans > 0:
                                        st.error(f"❌ {total_remaining_nans} missing values remain after preprocessing. Please adjust imputation settings or enable removal of exclusive missing features.")
                                        st.stop()
                                except Exception as e:
                                    st.error(f"Validation error after imputation: {e}")
                                    st.stop()

                                # ------------------ Binning ------------------
                                try:
                                    if apply_binning_option and bin_width is not None and mass_range_min is not None and mass_range_max is not None:
                                        # call domain-specific binning function (must exist in your codebase)
                                        data = apply_binning_to_mass_range(data, bin_width, (mass_range_min, mass_range_max))
                                except Exception as e:
                                    st.error(f"Binning error: {e}")
                                else:
                                    progress.progress(70)
                                    if apply_binning_option:
                                        st.success(f"Binning applied: {mass_range_min:.2f}-{mass_range_max:.2f} Da, bin width {bin_width:.2f}")

                                # ------------------ Normalization ------------------
                                try:
                                    if normalization_type != 'None':
                                        data = preprocess_data(data, normalization_type, progress)
                                except Exception as e:
                                    st.error(f"Normalization error: {e}")
                                else:
                                    progress.progress(85)
                                    if normalization_type != 'None':
                                        st.success(f"{normalization_type} normalization applied")
                                    else:
                                        st.info("No normalization applied")

                                # ------------------ Combat Batch Correction ------------------
                                try:
                                    if apply_combat:
                                        _cb_col = st.session_state.get("combat_batch_col", "Class")
                                        # Fallback to Class if selected col not in data
                                        if _cb_col not in data.columns:
                                            _cb_col = "Class"
                                        if _cb_col not in data.columns or data[_cb_col].nunique() < 2:
                                            st.warning(f"Combat skipped: column '{_cb_col}' not found or has fewer than 2 unique values.")
                                        else:
                                            from neurocombat_sklearn import CombatModel
                                            from sklearn.preprocessing import LabelEncoder
                                            le = LabelEncoder()
                                            batch_labels = le.fit_transform(data[_cb_col]).reshape(-1, 1)
                                            # Exclude metadata + structural cols from features, but keep _meta cols in output
                                            feat_cols = [
                                                c for c in data.columns
                                                if c not in cols_exclude
                                                and not str(c).endswith("_meta")
                                            ]
                                            # ensure numeric and drop any col that still contains NaN (shouldn't happen)
                                            features = data[feat_cols].select_dtypes(include=[np.number]).copy()
                                            features = features.dropna(axis=1)
                                            if features.shape[1] == 0:
                                                st.warning("No numeric features available for Combat after dropping NaNs.")
                                            else:
                                                combat = CombatModel()
                                                corrected = combat.fit_transform(features, batch_labels)
                                                # Rebuild: structural cols + _meta cols + corrected features
                                                _keep_meta = [c for c in data.columns if str(c).endswith("_meta")]
                                                meta = data[
                                                    [c for c in cols_exclude if c in data.columns] + _keep_meta
                                                ].reset_index(drop=True)
                                                data = pd.concat(
                                                    [meta, pd.DataFrame(corrected, columns=features.columns)],
                                                    axis=1
                                                )
                                                st.success(f"✅ Combat correction applied using **{_cb_col}** as batch variable")
                                except Exception as e:
                                    st.error(f"❌ Combat correction failed: {e}")

                                progress.progress(100)

                                # ------------------ Save & Summary ------------------
                                st.session_state['preprocessed_data'] = data
                                # Persist to disk for session survival
                                # Desktop: session_state only
                                # Do NOT auto-switch DataLab source — user stays on Raw/Edited
                                # They can manually switch to Preprocessed via the data source selector

                                st.markdown("**Preprocessing Summary**")
                                st.write(f"- Detection filter: {min_detection_threshold}% per class → {removed_by_detection} features removed")
                                st.write(f"- Imputation: {imputation_method} {'(per class)' if impute_by_class else ''} → {removed_by_exclusive_missing} exclusive missing features removed")
                                st.write(f"- Binning: {'Yes' if apply_binning_option else 'No'}")
                                st.write(f"- Normalization: {normalization_type}")
                                st.write(f"- Batch correction: {'Yes (column: ' + st.session_state.get('combat_batch_col', 'Class') + ')' if apply_combat else 'No'}")
                                # count features excluding metadata
                                feature_count = len([c for c in data.columns if c not in cols_exclude])
                                st.write(f"- Total features after preprocessing: {feature_count}")
                                # ------------------ Save & Summary ------------------
                                st.session_state['preprocessed_data'] = data
                                # Persist to disk for session survival
                                # Desktop: session_state only

                                # Stocker le résumé pour le rapport
                                st.session_state['preprocessing_summary'] = {
                                    "detection_filter": f"{min_detection_threshold}% per class → {removed_by_detection} features removed",
                                    "imputation": f"{imputation_method} {'(per class)' if impute_by_class else ''} → {removed_by_exclusive_missing} exclusive missing features removed",
                                    "binning": "Yes" if apply_binning_option else "No",
                                    "normalization": normalization_type,
                                    "batch_correction": f"Yes (batch column: {st.session_state.get('combat_batch_col', 'Class')})" if apply_combat else "No",
                                    "total_features": len([c for c in data.columns if c not in cols_exclude])
                                }

                            except Exception as e:
                                st.error(f"Preprocessing failed: {e}")



            # -------------------- Persistent Preview + Download Preprocessed Data --------------------
            _pp_dl = st.session_state.get("preprocessed_data")
            if _pp_dl is not None and not _pp_dl.empty:
                import csv as _csv_mod
                st.markdown("---")

                # ── Persistent preview (survives refresh & download button clicks) ──
                st.markdown("**Preprocessed Data Preview**")
                _pp_total_cols = _pp_dl.shape[1]
                if _pp_total_cols > 100:
                    st.info(f"Too many features ({_pp_total_cols}). Showing first 50 & last 50 columns. You can download the full preprocessed dataset below.")
                    _pp_preview_df = pd.concat([_pp_dl.iloc[:, :50], _pp_dl.iloc[:, -50:]], axis=1)
                    st.dataframe(_pp_preview_df, use_container_width=True)
                else:
                    st.dataframe(_pp_dl, use_container_width=True)

                _col_fname, _col_btn = st.columns([3, 2])
                with _col_fname:
                    # key= ensures the value survives reruns without clearing the result
                    _fname = st.text_input(
                        "CSV filename (without extension)",
                        value=st.session_state.get("pp_csv_filename", "preprocessed_data"),
                        key="pp_csv_filename",
                        help="Change the name of the downloaded file. Typing here will not clear the preprocessing result.",
                    )
                with _col_btn:
                    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                    _csv_buf = _pp_dl.to_csv(
                        index=False, sep=";",
                        quoting=_csv_mod.QUOTE_NONNUMERIC
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="📥 Download preprocessed data (CSV)",
                        data=_csv_buf,
                        file_name=f"{st.session_state.get('pp_csv_filename', 'preprocessed_data')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


            # -------------------- Post-Preprocessing QC --------------------

                with _t1_postqc:
                    st.markdown(
                        "<p style='color: gray'>Final quality checks after preprocessing to ensure data is model-ready.</p>",
                        unsafe_allow_html=True
                    )

                    data_pp = st.session_state.get("preprocessed_data")

                    if data_pp is None or data_pp.empty:
                        st.warning("⚠️ No preprocessed dataset available. Run preprocessing first.")
                    else:
                        with st.form("post_qc_form"):
                            run_post_qc = st.form_submit_button("Run Post-Preprocessing QC")

                        if run_post_qc:
                            with st.spinner("Running post-preprocessing QC..."):
                                df_qc = data_pp.copy()

                                # ------------------ Column split ------------------
                                meta_cols = ['Class', 'File', 'RT', 'Sum']
                                feature_cols = [c for c in df_qc.columns if c not in meta_cols]

                                # ------------------ 1. Missing values check ------------------
                                total_nan = df_qc[feature_cols].isna().sum().sum()

                                if total_nan == 0:
                                    st.success("✅ No missing values detected after preprocessing.")
                                else:
                                    st.error(f"❌ {total_nan} missing values remain after preprocessing.")
                                    st.info("Consider adjusting imputation or filtering settings.")

                                # ------------------ 2. Zero-inflation check ------------------
                                zero_pct_features = (df_qc[feature_cols] == 0).sum() / df_qc.shape[0] * 100
                                high_zero_features = zero_pct_features[zero_pct_features > 70]

                                st.markdown("**Zero-inflation summary**")
                                st.info(
                                    f"- Mean zero % per feature: {zero_pct_features.mean():.1f}%\n"
                                    f"- Features with >70% zeros: {len(high_zero_features)}"
                                )

                                if len(high_zero_features) > 0:
                                    st.warning(
                                        "⚠️ Some features remain highly sparse (>70% zeros).\n"
                                        "Consider additional filtering before modeling."
                                    )

                                import plotly.express as px


                                # ------------------ 3. Sample-level QC (boxplot) ------------------
                                df_qc["Features per sample"] = (df_qc[feature_cols] != 0).sum(axis=1)
                                df_qc["Total signal"] = df_qc[feature_cols].sum(axis=1)

                                st.markdown("**Sample-level QC after preprocessing**")

                                # Boxplot (same as before)
                                color_map = None
                                if "Class" in df_qc.columns and 'class_colors' in st.session_state:
                                    unique_classes = df_qc["Class"].unique()
                                    color_map = {
                                        cls: st.session_state['class_colors'].get(
                                            cls, f"#{hash(cls) % 0xFFFFFF:06x}"
                                        )
                                        for cls in unique_classes
                                    }

                                fig_box = px.box(
                                    df_qc,
                                    y="Features per sample",
                                    points="outliers",
                                    color="Class" if "Class" in df_qc.columns else None,
                                    color_discrete_map=color_map,
                                    title="Features per Sample (Post-Preprocessing)"
                                )
                                st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                _capture_plotly(fig_box, "Features per Sample Distribution Boxplot")

                                # ------------------ 4. Normalization effectiveness ------------------
                                cv = df_qc["Total signal"].std() / max(df_qc["Total signal"].mean(), 1e-6)

                                st.markdown("**Normalization consistency check**")
                                st.info(f"Total signal CV after preprocessing: **{cv:.2f}**")

                                if cv < 0.3:
                                    st.success("✅ Signal variability well controlled.")
                                elif cv < 0.5:
                                    st.warning("⚠️ Moderate variability remains across samples.")
                                else:
                                    st.error("❌ High variability remains — normalization may be insufficient.")

                                # ------------------ 5. Final verdict ------------------
                                st.markdown("**🚦 Model Readiness Verdict**")

                                if total_nan == 0 and cv < 0.5 and len(high_zero_features) < 0.2 * len(feature_cols):
                                    st.success(
                                        "🎯 **Dataset is model-ready**\n\n"
                                        "- No missing values\n"
                                        "- Controlled sparsity\n"
                                        "- Acceptable sample variability"
                                    )
                                else:
                                    st.warning(
                                        "⚠️ **Dataset may require further tuning**\n\n"
                                        "- Review imputation, filtering or normalization choices\n"
                                        "- Inspect sparse features or heterogeneous samples"
                                    )

                                # Cleanup
                                del df_qc
                                gc.collect()


            # ------------------ Oversampling ------------------


            # ════════════════════════════════════════════════════════════════════════
            # GROUP 4 — BALANCE & QC   (Dataset Balance · Sample QC · Sampling)
            # ════════════════════════════════════════════════════════════════════════
            with _grp_balance:
                st.markdown(_picon("balance","Check class balance, sample-level quality and apply resampling strategies.", "#8b5cf6"), unsafe_allow_html=True)
                _t1_qc, _t1_sampleqc, _t1_sampling = st.tabs([
                    "  Dataset Balance", "  Sample QC", "  Sampling"
                ])
                with _t1_qc:

                    with st.form("dataset_info_form"):
                        toggle = st.form_submit_button("Show Info")
                    if toggle:
                        st.session_state.show_info["dataset_info"] = not st.session_state.show_info["dataset_info"]

                    if st.session_state.show_info["dataset_info"] and df is not None:
                        st.markdown(f"""
                        <div style='background-color:#f0f8ff;padding:10px;border-radius:10px;font-size:15px;'>
                            <strong>Dataset source:</strong> {st.session_state.get('overview_source')}<br>
                            <strong>Dimensions:</strong> {df.shape[0]:,} samples × {df.shape[1]:,} features<br>
                            <strong>Classes:</strong> {df['Class'].nunique() if 'Class' in df.columns else 'N/A'}
                        </div>
                        """, unsafe_allow_html=True)

                        display_class_info(df)
                        st.info("Balanced datasets reduce model bias and variance inflation.")



                        fig_density = go.Figure()

                        if "Class" in df.columns and 'class_colors' in st.session_state:
                            # récupérer la map de couleurs
                            color_map = st.session_state['class_colors']

                            for cls in sorted(df["Class"].unique()):
                                # Colonnes numériques
                                relevant_cols = [
                                    col for col in df.columns
                                    if col != "Class" and pd.api.types.is_numeric_dtype(df[col])
                                ]
                                sub = df[df["Class"] == cls][relevant_cols]

                                values = sub.values.flatten()
                                values = values[~np.isnan(values)]
                                values = values[values > 0]
                                log_values = np.log2(values)

                                if len(log_values) > 10:
                                    kde = gaussian_kde(log_values)
                                    x_range = np.linspace(log_values.min(), log_values.max(), 500)
                                    y_kde = kde(x_range)

                                    fig_density.add_trace(go.Scatter(
                                        x=x_range,
                                        y=y_kde,
                                        mode='lines',
                                        name=str(cls),
                                        line=dict(
                                            width=2,
                                            color=color_map.get(cls, f"#{hash(cls) % 0xFFFFFF:06x}")  # fallback si cls pas dans class_colors
                                        )
                                    ))
                        else:
                            # Cas "All Samples"
                            relevant_cols = [
                                col for col in df.columns
                                if pd.api.types.is_numeric_dtype(df[col])
                            ]
                            sub = df[relevant_cols]
                            values = sub.values.flatten()
                            values = values[~np.isnan(values)]
                            values = values[values > 0]
                            log_values = np.log2(values)

                            kde = gaussian_kde(log_values)
                            x_range = np.linspace(log_values.min(), log_values.max(), 500)
                            y_kde = kde(x_range)

                            fig_density.add_trace(go.Scatter(
                                x=x_range,
                                y=y_kde,
                                mode='lines',
                                name="All Samples",
                                line=dict(width=2, color="gray")
                            ))

                        fig_density.update_layout(
                            title="Density Plot of Log2 Feature Intensities",
                            xaxis_title="Log2 Intensity",
                            yaxis_title="Density",
                            plot_bgcolor="white",
                            yaxis=dict(gridcolor="#eeeeee"),
                            xaxis=dict(gridcolor="#eeeeee"),
                            height=420
                        )

                        st.plotly_chart(fig_density, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_density, "density_intensity_per_class")

            # -------------------- Outliers & Sample QC --------------------

                with _t1_sampleqc:
                    st.markdown(
                        "<p style='color: gray'>Sample-level QC using feature counts, total signal and IQR-based outlier detection.</p>",
                        unsafe_allow_html=True
                    )

                    if df is None:
                        st.warning("Please load a dataset first.")
                    else:
                        with st.form("full_analysis_form"):
                            run_analysis = st.form_submit_button("Run QC Analysis")

                        if run_analysis:
                            with st.spinner("Running QC analysis..."):
                                df_qc = df.copy()


                                feature_cols = df_qc.drop(columns=['Class'], errors='ignore')
                                df_qc['Features per sample'] = feature_cols.notna().sum(axis=1)
                                df_qc['Total signal'] = feature_cols.fillna(0).sum(axis=1)



                                # --- QC stats ---
                                fps_mean = df_qc['Features per sample'].mean()
                                fps_std  = df_qc['Features per sample'].std()
                                _sig_mean = df_qc['Total signal'].mean()
                                _sig_std  = df_qc['Total signal'].std()

                                # CV sur le nombre de features
                                if fps_mean > 0:
                                    features_cv = (fps_std / fps_mean) * 100
                                    _cv_feat_label = f"CV: {features_cv:.1f}%"
                                else:
                                    features_cv = None
                                    _cv_feat_label = "CV: N/A"

                                # CV sur le signal total (intensités)
                                if _sig_mean > 0:
                                    signal_cv = (_sig_std / _sig_mean) * 100
                                    _cv_label = f"CV: {signal_cv:.1f}%"
                                else:
                                    signal_cv = None
                                    _cv_label = "CV: N/A (signal contains negatives — use absolute intensities)"

                                # --- Min / Max sample-level stats ---
                                fps_min = df_qc['Features per sample'].min()
                                fps_max = df_qc['Features per sample'].max()

                                signal_min = df_qc['Total signal'].min()
                                signal_max = df_qc['Total signal'].max()


                                # Affichage
                                st.markdown("**Sample QC summary**")
                                st.info(
                                    f"Features/sample → "
                                    f"min: {fps_min:.0f} | max: {fps_max:.0f} | "
                                    f"mean: {fps_mean:.1f} | std: {fps_std:.1f} | {_cv_feat_label}\n"
                                    f"Total signal → "
                                    f"min: {signal_min:.2e} | max: {signal_max:.2e} | "
                                    f"mean: {_sig_mean:.2e} | std: {_sig_std:.2e} | {_cv_label}"
                                )



                                df_features = df.copy()



                                # --- IQR outlier detection ---
                                Q1 = df_qc['Features per sample'].quantile(0.25)
                                Q3 = df_qc['Features per sample'].quantile(0.75)
                                IQR = Q3 - Q1
                                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                                df_qc['IQR_outlier'] = (
                                    (df_qc['Features per sample'] < lower) |
                                    (df_qc['Features per sample'] > upper)
                                )

                                detected_outliers = df_qc[df_qc['IQR_outlier']].index.tolist()
                                st.session_state['detected_outliers'] = detected_outliers

                                if detected_outliers or signal_cv > 0.5:
                                    st.warning(
                                        "⚠️ Potential sample-level issues detected.\n"
                                        "- Inspect low-coverage or extreme samples\n"
                                        "- Manual removal recommended before modeling"
                                    )
                                else:
                                    st.success("✅ No critical sample-level QC issues detected.")

                                # -------------------- 📊 QC PLOT (AVANT SUPPRESSION) --------------------
                                import plotly.express as px



                                df_features["Features per sample"] = (
                                    df.drop(columns=['Class'], errors='ignore')
                                    .notna()
                                    .sum(axis=1)
                                )

                                color_map = None
                                if "Class" in df_features.columns and 'class_colors' in st.session_state:
                                    unique_classes = df_features["Class"].unique()
                                    color_map = {
                                        cls: st.session_state['class_colors'].get(
                                            cls, f"#{hash(cls) % 0xFFFFFF:06x}"
                                        )
                                        for cls in unique_classes
                                    }

                                fig = px.box(
                                    df_features,
                                    y="Features per sample",
                                    points="outliers",
                                    color="Class" if "Class" in df_features.columns else None,
                                    color_discrete_map=color_map,
                                    title="Features per Sample Distribution"
                                )

                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                                _capture_plotly(fig, "qc_features_per_sample")
                                st.session_state['qc_analysis_done'] = True
                        

                                # -------------------- 📋 TABLE --------------------


                                plot_df = df_qc.reset_index().rename(columns={"index": "Sample index"})



                                st.markdown("**QC table (used for selection below)**")
                                st.dataframe(
                                    plot_df[
                                        ["Sample index", "Class", "Features per sample", "Total signal", "IQR_outlier"]
                                        if "Class" in plot_df.columns
                                        else ["Sample index", "Features per sample", "Total signal", "IQR_outlier"]
                                    ],
                                    use_container_width=True,
                                    height=300
                                )

                                # Persister les options pour le widget de suppression (hors run_analysis)
                                st.session_state['_qc_sample_options'] = plot_df["Sample index"].tolist()

                        # -------------------- 🗑 MANUAL SAMPLE REMOVAL (outside run_analysis) --------------------
                        # Rendu en dehors du bloc if run_analysis: pour survivre au rerun Streamlit
                        if st.session_state.get('qc_analysis_done') and st.session_state.get('_qc_sample_options') is not None:
                            st.markdown("**🗑 Remove samples**")

                            _default_outliers = [
                                i for i in st.session_state.get('detected_outliers', [])
                                if i in st.session_state['_qc_sample_options']
                            ]

                            samples_to_remove = st.multiselect(
                                "Select sample indices to remove (pre-selected = IQR outliers):",
                                options=st.session_state['_qc_sample_options'],
                                default=_default_outliers,
                                key="outlier_removal_multiselect"
                            )

                            if samples_to_remove:
                                if st.button(f"Remove {len(samples_to_remove)} selected samples", key="outlier_remove_btn"):
                                    for key in ["data", "final_data", "overview_df", "preprocessed_data"]:
                                        if key in st.session_state and st.session_state[key] is not None:
                                            st.session_state[key] = (
                                                st.session_state[key]
                                                .drop(index=samples_to_remove, errors="ignore")
                                                .reset_index(drop=True)
                                            )
                                    # Re-sync DataLab cache
                                    _cur_src_qc = st.session_state.get('datalab_source', 'Raw / Edited')
                                    _sync_qc = {
                                        'Raw / Edited': st.session_state.get('final_data', st.session_state.get('data')),
                                        'Preprocessed': st.session_state.get('preprocessed_data'),
                                        'Oversampled':  st.session_state.get('oversampled_data'),
                                        'Undersampled': st.session_state.get('undersampled_data'),
                                    }.get(_cur_src_qc)
                                    if _sync_qc is not None:
                                        st.session_state['datalab_df']  = _sync_qc.copy()
                                        st.session_state['overview_df'] = _sync_qc.copy()

                                    st.success(f"✅ {len(samples_to_remove)} samples removed from all datasets.")
                                    st.session_state['detected_outliers'] = []
                                    st.session_state['qc_analysis_done'] = False
                                    st.session_state['_qc_sample_options'] = None
                                    st.rerun()


                with _t1_sampling:
                    st.markdown('<p style="color: gray; font-size: 14px">Class balancing strategies</p>', unsafe_allow_html=True)
                    with st.form("oversampling_form"):
                        source = st.selectbox("Select Data Source", ['Raw Data', 'Preprocessed'])
                        technique = st.selectbox("Oversampling Technique", ['None', 'SMOTE', 'ADASYN'])
                        apply_btn = st.form_submit_button("✅ Apply Oversampling")
                    if apply_btn and technique != 'None':
                        try:
                            data_use = st.session_state.get('preprocessed_data') if source=='Preprocessed' else st.session_state.get('final_data', st.session_state.get('data'))
                            if data_use is None: st.error("No valid data"); st.stop()
                            progress = st.progress(0)
                            st.session_state['oversampled_data'] = apply_sampling(data_use, technique.lower(), _progress_bar=progress)
                            # Desktop: session_state only
                            st.success("✅ Oversampling successful")
                        except Exception as e: st.error(f"Oversampling error: {e}")

            # ------------------ Undersampling ------------------
                    st.markdown('<p style="color: gray; font-size: 14px">Class balancing strategies</p>', unsafe_allow_html=True)
                    with st.form("undersampling_form"):
                        source = st.selectbox("Select Data Source", ['Raw Data', 'Preprocessed'])
                        technique = st.selectbox("Undersampling Technique", ['None', 'RandomUnderSampler', 'NearMiss'])
                        apply_btn = st.form_submit_button("✅ Apply Undersampling")
                    if apply_btn and technique != 'None':
                        try:
                            data_use = st.session_state.get('preprocessed_data') if source=='Preprocessed' else st.session_state.get('final_data', st.session_state.get('data'))
                            if data_use is None: st.error("No valid data"); st.stop()
                            X = data_use.drop([c for c in data_use.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')], axis=1, errors='ignore').select_dtypes(include='number'); y = data_use['Class']
                            if technique=='RandomUnderSampler': from imblearn.under_sampling import RandomUnderSampler; X_res, y_res = RandomUnderSampler(random_state=1).fit_resample(X,y)
                            elif technique=='NearMiss': from imblearn.under_sampling import NearMiss; X_res, y_res = NearMiss(version=1).fit_resample(X,y)
                            st.session_state['undersampled_data'] = pd.concat([X_res, y_res], axis=1)
                            # Desktop: session_state only
                            st.write(st.session_state['undersampled_data']['Class'].value_counts())
                            st.success("✅ Undersampling successful")

                        except Exception as e: st.error(f"Undersampling error: {e}")
    with tabs[2]:
        # ══════════════════════════════════════════════
        # DATA VIZ – shared single data loader
        # ══════════════════════════════════════════════
        _raw  = st.session_state.get('final_data', st.session_state.get('data'))
        _pp   = st.session_state.get('preprocessed_data')
        _os   = st.session_state.get('oversampled_data')
        _us   = st.session_state.get('undersampled_data')

        _viz_df = None
        if _raw is None and _pp is None:
            st.info("📂 Load a dataset from the sidebar and check your data in data lab before the visualizations.", icon="ℹ️")
        else:
            _available_sources = {k: v for k, v in {
                "Raw Data": _raw, "Preprocessed": _pp,
                "Oversampled": _os, "Undersampled": _us,
            }.items() if v is not None}

            _col_src, _col_info = st.columns([3, 5])
            with _col_src:
                _viz_keys = list(_available_sources.keys())
                _viz_default = _viz_keys.index("Preprocessed") if "Preprocessed" in _viz_keys else 0
                _viz_source_name = st.selectbox(
                    "🗂️ Data source (applies to all visualizations below)",
                    _viz_keys,
                    index=_viz_default,
                    key="global_viz_source",
                )
            with _col_info:
                _viz_df = _available_sources[_viz_source_name]
                if _viz_df is not None:
                    n_samp, n_feat = _viz_df.shape[0], _viz_df.shape[1] - (1 if 'Class' in _viz_df.columns else 0)
                    n_cls = _viz_df['Class'].nunique() if 'Class' in _viz_df.columns else 0
                    st.markdown(f"""
                    <div style='padding:8px 0 0 8px;color:#64748b;font-size:0.8rem;'>
                        <span style='color:#93c5fd;font-weight:700;'>{n_samp}</span> samples &nbsp;·&nbsp;
                        <span style='color:#86efac;font-weight:700;'>{n_feat}</span> features &nbsp;·&nbsp;
                        <span style='color:#fcd34d;font-weight:700;'>{n_cls}</span> classes
                    </div>""", unsafe_allow_html=True)

            # ── Ensure class_colors is in sync with the *current* class names ──
            # When classes are renamed/grouped, old keys may linger and new keys
            # may be missing. We rebuild the dict so that:
            #   1. New class names get a default colour (or inherit the colour of
            #      their predecessor via class_renaming, so user picks are kept).
            #   2. Stale keys (old names that no longer exist) are removed.
            import plotly.express as px
            if _viz_df is not None and 'Class' in _viz_df.columns:
                if 'class_colors' not in st.session_state:
                    st.session_state['class_colors'] = {}
                _current_classes = list(_viz_df['Class'].unique())
                # Build inverse renaming map: new_name → old_name (for colour inheritance)
                _rmap = st.session_state.get('class_renaming', {})
                _rmap_inv = {}
                for _old, _new in _rmap.items():
                    if _new not in _rmap_inv:
                        _rmap_inv[_new] = _old
                _palette = px.colors.qualitative.Plotly
                _new_colors = {}
                for _ci, _cls in enumerate(_current_classes):
                    if _cls in st.session_state['class_colors']:
                        _new_colors[_cls] = st.session_state['class_colors'][_cls]
                    elif _rmap_inv.get(_cls) in st.session_state['class_colors']:
                        # Inherit colour from the predecessor name after renaming
                        _new_colors[_cls] = st.session_state['class_colors'][_rmap_inv[_cls]]
                    else:
                        _new_colors[_cls] = _palette[_ci % len(_palette)]
                # Replace entirely so stale old-name keys are removed
                st.session_state['class_colors'] = _new_colors

            st.markdown("<hr style='margin:10px 0 14px 0;border:none;border-top:1px solid rgba(255,255,255,0.07);'>", unsafe_allow_html=True)

        _viz_subtabs = st.tabs(["  Class Colors", "  Feature Distribution", "  Multi-Feature", "  Signal Profile"])

        # ── Tab: Class Colors ──
        with _viz_subtabs[0]:
            if _viz_df is not None and 'Class' in _viz_df.columns:
                with st.form("class_colors_form"):
                    _unique_classes = list(_viz_df['Class'].unique())
                    st.write("Select a color for each class:")
                    for _class_name in _unique_classes:
                        st.session_state['class_colors'][_class_name] = st.color_picker(
                            f"Color for {_class_name}",
                            st.session_state['class_colors'].get(_class_name, "#636EFA"),
                            key=f"color_{_class_name}_custom"
                        )
                    if st.form_submit_button("✅ Apply Colors"):
                        st.success("Class colors updated successfully!")
            else:
                st.info("No class column found in the selected data.")

        # ── Tab: Feature Distribution ──
        with _viz_subtabs[1]:
            if _viz_df is not None:
                with st.form("feature_distribution_form"):
                    if 'available_features' not in st.session_state:
                        st.session_state['available_features'] = []
                    st.session_state['available_features'] = [
                        c for c in _viz_df.columns
                        if c not in _NON_FEATURE_COLS
                        and not str(c).endswith('_meta')
                        and pd.api.types.is_numeric_dtype(_viz_df[c])
                    ]
                    feature_to_explore = st.selectbox(
                        "Select Feature for Exploration",
                        st.session_state['available_features'],
                        key="feature_to_explore_single"
                    )
                    histfunc_single = st.selectbox(
                        "Aggregation Function",
                        ['sum', 'count', 'avg', 'min', 'max'],
                        key="histfunc_select_single"
                    )
                    apply_viz = st.form_submit_button("✅ Show Feature Distribution")
                if 'feature_distribution_plot' not in st.session_state:
                    st.session_state['feature_distribution_plot'] = go.Figure()
                if apply_viz and feature_to_explore:
                    fig = plot_feature_distribution(
                        _viz_df, feature_to_explore,
                        st.session_state['class_colors'], histfunc_single
                    )
                    st.session_state['feature_distribution_plot'] = fig
                if isinstance(st.session_state.get('feature_distribution_plot'), go.Figure):
                    st.plotly_chart(st.session_state['feature_distribution_plot'], use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                    _capture_plotly(st.session_state['feature_distribution_plot'], "feature_distribution_plot")
                gc.collect()
            else:
                st.info("Select a valid data source above.")

        # ── Tab: Multi-Feature ──
        with _viz_subtabs[2]:
            if _viz_df is not None:
                with st.form("multi_feature_form"):
                    if 'available_multi_features' not in st.session_state:
                        st.session_state['available_multi_features'] = []
                    st.session_state['available_multi_features'] = [c for c in _viz_df.columns if c != 'Class']
                    features_to_explore = st.multiselect(
                        "Select Features for Comparison",
                        st.session_state['available_multi_features'],
                        help="Choose two or more features."
                    )
                    plot_type = st.selectbox("Visualization Type",
                        ['Radar Chart', 'Line Plot', 'Bar Chart'], key="plot_type_select_multi")
                    available_histfuncs = ['sum', 'count', 'mean', 'min', 'max', 'percentage'] if plot_type == 'Bar Chart' else ['sum', 'mean', 'min', 'max', 'percentage']
                    histfunc_multi = st.selectbox("Aggregation Function", available_histfuncs, key="histfunc_multi_select")
                    error_type = st.selectbox("Error Bar (line plot only)", ['None', 'SEM', 'STD'], key="error_bar_type_select") if plot_type == 'Line Plot' else 'None'
                    if plot_type == 'Bar Chart':
                        if 'feature_colors' not in st.session_state:
                            st.session_state['feature_colors'] = {}
                        for feature in features_to_explore:
                            if feature not in st.session_state['feature_colors']:
                                st.session_state['feature_colors'][feature] = '#636EFA'
                            st.session_state['feature_colors'][feature] = st.color_picker(
                                f"Color for {feature}", st.session_state['feature_colors'][feature],
                                key=f"color_feature_{feature}")
                    apply_viz_multi = st.form_submit_button("Show Multi-Feature Comparison")
                if 'multi_feature_plot' not in st.session_state:
                    st.session_state['multi_feature_plot'] = go.Figure()
                if apply_viz_multi and len(features_to_explore) >= 2:
                    if plot_type == 'Bar Chart':
                        fig = plot_multiple_features_distribution(
                            _viz_df, features_to_explore, st.session_state['feature_colors'],
                            histfunc_multi, capture_name="multiple_feature_plot")
                    elif plot_type == 'Line Plot':
                        fig = plot_multiple_features_line(
                            _viz_df, features_to_explore, st.session_state['class_colors'],
                            histfunc=histfunc_multi,
                            error_type=error_type.lower() if error_type != 'None' else None,
                            capture_name="multi_feature_plot")
                    elif plot_type == 'Radar Chart':
                        fig = plot_multiple_features_radar(
                            _viz_df, features_to_explore, st.session_state['class_colors'], histfunc_multi)
                    st.session_state['multi_feature_plot'] = fig
                elif apply_viz_multi:
                    st.warning("⚠️ Please select at least two features.")
                if isinstance(st.session_state.get('multi_feature_plot'), go.Figure):
                    st.plotly_chart(st.session_state['multi_feature_plot'], use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                gc.collect()
            else:
                st.info("Select a valid data source above.")

        # ── Tab: Signal Profile ──
        with _viz_subtabs[3]:
            if _viz_df is not None:
                st.session_state['signal_data'] = _viz_df
                with st.form("signal_profile_form"):
                    class_options = _viz_df['Class'].unique().tolist() if 'Class' in _viz_df.columns else []
                    class_to_plot = st.multiselect("Select Class for Mean Profile", class_options, key="class_to_plot_signal")
                    apply_mean_profile = st.form_submit_button("Show Average Profile")
                    index_labels = {idx: f"Index {idx} (Class {row['Class']})" for idx, row in _viz_df.iterrows()}
                    selected_indices = st.multiselect(
                        "Select Individual Profiles",
                        options=list(index_labels.keys()),
                        format_func=lambda x: index_labels.get(x, str(x)),
                        key="selected_indices_signal")
                    apply_individual = st.form_submit_button("Show Individual Profiles")
                if apply_mean_profile and class_to_plot:
                    fig_mean = plot_mean_spectrum(
                        _viz_df, class_to_plot, st.session_state['class_colors'],
                        capture_name="mean_spectrum_plot")
                    st.session_state['mean_spectrum_plot'] = fig_mean
                if 'mean_spectrum_plot' in st.session_state and isinstance(st.session_state['mean_spectrum_plot'], go.Figure):
                    st.plotly_chart(st.session_state['mean_spectrum_plot'], use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                if apply_individual and selected_indices:
                    fig_ind = plot_individual_spectra(
                        _viz_df, st.session_state['class_colors'], selected_indices,
                        capture_name="individual_spectra_plot")
                    st.session_state['individual_spectra_plot'] = fig_ind
                if 'individual_spectra_plot' in st.session_state and isinstance(st.session_state['individual_spectra_plot'], go.Figure):
                    st.plotly_chart(st.session_state['individual_spectra_plot'], use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                gc.collect()
            else:
                st.info("Select a valid data source above.")

    with tabs[3]:
        # ══════════════════════════════════════════════
        # COMPARISONS – Correlation · Similarity · Venn/UpSet
        # ══════════════════════════════════════════════
        _raw  = st.session_state.get('final_data', st.session_state.get('data'))
        _pp   = st.session_state.get('preprocessed_data')
        _os   = st.session_state.get('oversampled_data')
        _us   = st.session_state.get('undersampled_data')

        _cmp_df = None
        if _raw is None and _pp is None:
            st.info("📂 Load a dataset from the sidebar to access the comparisons.", icon="ℹ️")
        else:
            _cmp_sources = {k: v for k, v in {
                "Raw Data": _raw, "Preprocessed": _pp,
                "Oversampled": _os, "Undersampled": _us,
            }.items() if v is not None}

            _cmp_col_src, _cmp_col_info = st.columns([3, 5])
            with _cmp_col_src:
                _cmp_keys = list(_cmp_sources.keys())
                _cmp_default = _cmp_keys.index("Preprocessed") if "Preprocessed" in _cmp_keys else 0
                _cmp_source_name = st.selectbox(
                    "🗂️ Data source (applies to all comparisons below)",
                    _cmp_keys,
                    index=_cmp_default,
                    key="global_cmp_source",
                )
            _cmp_df = _cmp_sources[_cmp_source_name]
            with _cmp_col_info:
                if _cmp_df is not None:
                    n_s, n_f = _cmp_df.shape[0], _cmp_df.shape[1] - (1 if 'Class' in _cmp_df.columns else 0)
                    n_c = _cmp_df['Class'].nunique() if 'Class' in _cmp_df.columns else 0
                    st.markdown(f"""<div style='padding:8px 0 0 8px;color:#64748b;font-size:0.8rem;'>
                        <span style='color:#93c5fd;font-weight:700;'>{n_s}</span> samples &nbsp;·&nbsp;
                        <span style='color:#86efac;font-weight:700;'>{n_f}</span> features &nbsp;·&nbsp;
                        <span style='color:#fcd34d;font-weight:700;'>{n_c}</span> classes
                    </div>""", unsafe_allow_html=True)

            st.markdown("<hr style='margin:10px 0 14px 0;border:none;border-top:1px solid rgba(255,255,255,0.07);'>", unsafe_allow_html=True)

        _cmp_subtabs = st.tabs(["  Correlation", "  Similarity", "  Venn / UpSet"])

        # ── Correlation ──
        with _cmp_subtabs[0]:
            st.markdown('<p style="color:gray;font-size:14px;">Compute correlations between the average feature vectors of each class using Pearson or Spearman methods.</p>', unsafe_allow_html=True)
            with st.form("correlation_form"):
                corr_method = st.selectbox("Correlation Method", ['None', 'Pearson', 'Spearman'], index=0, key="corr_method")
                apply_corr = st.form_submit_button("✅ Apply Correlation")
            if apply_corr:
                if _cmp_df is None or 'Class' not in _cmp_df.columns:
                    st.warning("No valid data available for correlation.")
                elif corr_method == 'None':
                    st.warning("Please select a correlation method.")
                else:
                    numeric_data = _cmp_df.drop(columns=['Class'], errors='ignore').select_dtypes(include='number')
                    if numeric_data.empty:
                        st.warning("No numeric features found for correlation.")
                    else:
                        grouped = _cmp_df.groupby('Class')[numeric_data.columns].mean()
                        # Use pandas corr (already numpy-backed for float data)
                        corr_matrix = grouped.T.corr(method=corr_method.lower())
                        # Ensure float32 to halve memory usage on large matrices
                        corr_matrix = corr_matrix.astype('float32')
                        plot_heatmap(corr_matrix, f"{corr_method} Correlation",
                            "Each cell shows the correlation coefficient between classes (averaged features).",
                            capture_name="corr_heatmap")
                del numeric_data
                gc.collect()

        # ── Similarity ──
        with _cmp_subtabs[1]:
            st.markdown("<p style='color:gray;font-size:14px;'>Compare class profiles using Cosine Similarity (continuous) or Cohen's Kappa (categorical after discretization).</p>", unsafe_allow_html=True)
            with st.form("similarity_form"):
                similarity_method = st.selectbox("Similarity Method", ["None", "Cosine Similarity", "Cohen's Kappa"], index=0, key="sim_method")
                apply_sim = st.form_submit_button("✅ Apply Similarity")
            if apply_sim:
                if _cmp_df is None or 'Class' not in _cmp_df.columns:
                    st.warning("No valid data available for similarity.")
                elif similarity_method == 'None':
                    st.warning("Please select a similarity method.")
                elif _cmp_df.drop(columns=['Class'], errors='ignore').isnull().values.any():
                    st.error("⚠️ Missing values detected. Please impute before running similarity analysis.")
                else:
                    numeric_data = _cmp_df.drop(columns=['Class'], errors='ignore').select_dtypes(include='number')
                    if numeric_data.empty:
                        st.warning("No numeric features found for similarity analysis.")
                    else:
                        grouped = _cmp_df.groupby('Class')[numeric_data.columns].mean()
                        classes = grouped.index
                        if similarity_method == "Cosine Similarity":
                            sim_matrix = cosine_similarity(grouped)
                            sim_df = pd.DataFrame(sim_matrix, index=classes, columns=classes)
                            plot_heatmap(sim_df, "Cosine Similarity",
                                "Cosine similarity measures the angle between feature vectors of each class (1=identical, 0=orthogonal).",
                                capture_name="sim_heatmap")
                        elif similarity_method == "Cohen's Kappa":
                            nb_bins = 3
                            kappa_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)
                            for i_k, class_i in enumerate(classes):
                                for j_k, class_j in enumerate(classes):
                                    vec1_cat = pd.qcut(grouped.loc[class_i].rank(method="first"), q=nb_bins, labels=False)
                                    vec2_cat = pd.qcut(grouped.loc[class_j].rank(method="first"), q=nb_bins, labels=False)
                                    kappa_matrix.iloc[i_k, j_k] = cohen_kappa_score(vec1_cat, vec2_cat)
                            plot_heatmap(kappa_matrix, "Cohen's Kappa Similarity",
                                "Cohen's Kappa evaluates the agreement in categorized feature profiles (1=perfect, 0=random, <0=disagreement).")
                gc.collect()

        # ── Venn / UpSet ──
        with _cmp_subtabs[2]:
            st.markdown('<p style="color:gray;font-size:14px;">Visualize class relationships and feature overlaps. Exclusive features per class can be extracted and sent directly to ORA enrichment.</p>', unsafe_allow_html=True)
            st.info(
                "💡 **Recommended source: Raw Data** — Venn & UpSet diagrams compute *presence/absence* "
                "of features per class based on non-null values. "
                "After preprocessing, missing values are often imputed (filled with 0 or an estimated "
                "value), which erases the natural sparsity pattern and inflates the overlap between "
                "classes. For proteomics especially, use the **Raw Data** source to correctly identify "
                "features that are exclusively detected in one condition.",
                icon="ℹ️",
            )
            with st.form("venn_upset_form"):
                col1_v, col2_v = st.columns(2)
                with col1_v:
                    show_venn = st.form_submit_button("Show Venn Diagram")
                with col2_v:
                    show_upset = st.form_submit_button("Show UpSet Plot")
            if show_venn or show_upset:
                if _cmp_df is not None:
                    venn_data = _cmp_df.drop(columns=['File', 'RT', 'Sum'], errors='ignore')
                    classes = venn_data['Class'].unique()
                    num_classes = len(classes)
                    st.write(f"Detected {num_classes} unique classes.")
                    if show_venn:
                        if num_classes <= 6:
                            fig = plot_venn_diagram(venn_data, 'Class', st.session_state['class_colors'],
                                _cmp_source_name, capture_name="venn_diagram_plot")
                            del fig
                        else:
                            st.error("⚠️ Too many classes (>6). Only UpSet plot is available.")
                    if show_upset:
                        if num_classes > 1:
                            fig = plot_upset(venn_data, 'Class', _cmp_source_name,
                                capture_name="upset_plot",
                                class_colors=st.session_state.get('class_colors', {}))
                            del fig
                        else:
                            st.warning("⚠️ At least 2 classes are required for UpSet plot.")

                    # ── Compute exclusive & shared features ───────────────
                    relevant_cols = [c for c in venn_data.columns if c != 'Class']
                    _cls_feats = {}
                    for _cls in classes:
                        _mask = venn_data['Class'] == _cls
                        _cls_feats[_cls] = set(
                            venn_data.loc[_mask, relevant_cols]
                            .columns[venn_data.loc[_mask, relevant_cols].notnull().any()]
                        )
                    _all_feats = set.union(*_cls_feats.values()) if _cls_feats else set()
                    _shared    = set.intersection(*_cls_feats.values()) if len(_cls_feats) > 1 else set()
                    _exclusive = {cls: feats - set.union(*(_cls_feats[c] for c in classes if c != cls))
                                  for cls, feats in _cls_feats.items()}

                    # Store in session state for ORA
                    st.session_state['_venn_exclusive_features'] = _exclusive
                    st.session_state['_venn_shared_features']    = _shared
                    st.session_state['_venn_classes']            = list(classes)

                    # ── Feature summary expander ──────────────────────────
                    st.markdown("---")
                    with st.expander("🔬 Exclusive & Shared Features", expanded=True):
                        _exc_cols = st.columns(min(len(classes), 4))
                        for _ci, _cls in enumerate(classes):
                            with _exc_cols[_ci % len(_exc_cols)]:
                                _exc = sorted(_exclusive.get(_cls, []))
                                _color = st.session_state.get('class_colors', {}).get(_cls, '#318CE7')
                                st.markdown(
                                    f"<div style='border:1px solid {_color};border-radius:8px;"
                                    f"padding:10px;margin-bottom:8px;'>"
                                    f"<b style='color:{_color}'>{_cls}</b> "
                                    f"<span style='font-size:0.75rem;color:#64748b;'>"
                                    f"({len(_exc)} exclusive features)</span></div>",
                                    unsafe_allow_html=True
                                )
                                if _exc:
                                    st.code(", ".join(_exc[:30]) + ("..." if len(_exc) > 30 else ""), language=None)
                                else:
                                    st.caption("No exclusive features")

                        if _shared:
                            st.markdown(f"**Shared by all classes:** {len(_shared)} features")
                            st.code(", ".join(sorted(_shared)[:30]) + ("..." if len(_shared) > 30 else ""), language=None)

                        st.info(
                            "💡 Exclusive features are now **directly available** in the "
                            "**Enrichment** tab → ORA as source **'🔷 Venn/UpSet — Exclusive per class'**.",
                            icon="ℹ️"
                        )

                    del venn_data, classes, num_classes
                    gc.collect()
                else:
                    st.warning("⚠️ No valid data available for Venn/UpSet.")

    with tabs[4]:
        # ═══════════════════════════════════════════════════
        # AI MODELING – single source selector + sub-tabs
        # ═══════════════════════════════════════════════════
        _t4_raw = st.session_state.get('final_data', st.session_state.get('data'))
        _t4_pp  = st.session_state.get('preprocessed_data')
        _t4_os  = st.session_state.get('oversampled_data')
        _t4_us  = st.session_state.get('undersampled_data')
        _t4_df = None
        _t4_src = None
        if _t4_raw is None and _t4_pp is None:
            st.info("📂 Load a dataset from the sidebar to access AI models.", icon="ℹ️")
        else:
            _t4_avail = {k:v for k,v in {"Raw Data":_t4_raw,"Preprocessed":_t4_pp,"Oversampled":_t4_os,"Undersampled":_t4_us}.items() if v is not None}
            _t4_c1, _t4_c2 = st.columns([3,5])
            with _t4_c1:
                _t4_keys = list(_t4_avail.keys())
                _t4_default = _t4_keys.index("Preprocessed") if "Preprocessed" in _t4_keys else 0
                _t4_src = st.selectbox("🗂️ Data source (applies to all analyses below)", _t4_keys, index=_t4_default, key="global_t4_source")
            _t4_df = _t4_avail[_t4_src]
            with _t4_c2:
                if _t4_df is not None:
                    _n4s,_n4f,_n4c = _t4_df.shape[0],_t4_df.shape[1]-(1 if 'Class' in _t4_df.columns else 0),(_t4_df['Class'].nunique() if 'Class' in _t4_df.columns else 0)
                    st.markdown(f"<div style='padding:8px 0 0 8px;color:#64748b;font-size:0.8rem;'><span style='color:#318CE7;font-weight:700;'>{_n4s}</span> samples &nbsp;·&nbsp;<span style='color:#10b981;font-weight:700;'>{_n4f}</span> features &nbsp;·&nbsp;<span style='color:#f59e0b;font-weight:700;'>{_n4c}</span> classes</div>",unsafe_allow_html=True)
            st.markdown("<hr style='margin:10px 0 14px;border:none;border-top:1px solid #e2e8f0;'>",unsafe_allow_html=True)
        (_t4_dimred, _t4_clust, _t4_ml, _t4_dl, _t4_long, _t4_save) = st.tabs([
            "  Dimensionality Reduction","  Clustering",
            "  Supervised ML","  Deep Learning",
            "  Longitudinal ⚠️",
            "  Save Model"])




    # --- Initialisation des variables de session ---
    for key, default in [
        ("X_reduction", None),
        ("y_reduction", None),
        ("compressed_data", None),
        ("reduction_method", None),
        ("svd_available", False),
        ("fig_initial", None),
        ("fig_feature", None),
        ("n_components_reduction", 2)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with _t4_dimred:
        st.markdown(
            '<p style="color: gray; font-size: 14px;">Reduce Dimensionality and Visualize Clusters Using PCA, UMAP or t-SNE</p>',
            unsafe_allow_html=True
        )

        # --- FORM 1 : Apply Reduction ---
        with st.form("dim_reduction_form"):
            method = st.selectbox("Visualization by Data Reduction", ['None', 'PCA', 'UMAP', 't-SNE'], key="reduction_choice")
            # Data source comes from the global selector above
            data_source = _t4_src if _t4_src else "Raw Data"  # "Raw Data", "Preprocessed", "Oversampled", "Undersampled"
            n_components = st.number_input("Number of Components", min_value=2, max_value=200,
                                        value=st.session_state["n_components_reduction"], step=1, key="n_components")

            # ── Colour-by selector (Class + _meta columns) ────────────────
            _dr_df_preview = {
                'Raw Data': st.session_state.get('final_data', st.session_state.get('data')),
                'Preprocessed': st.session_state.get('preprocessed_data'),
                'Oversampled': st.session_state.get('oversampled_data'),
                'Undersampled': st.session_state.get('undersampled_data'),
            }.get(data_source)
            _meta_avail_dr = (
                [c for c in _dr_df_preview.columns if str(c).endswith('_meta')]
                if _dr_df_preview is not None else []
            )
            _color_options = ['Class'] + _meta_avail_dr
            color_by = st.selectbox(
                "🎨 Color points by",
                options=_color_options,
                index=0,
                key="dim_color_by",
                help=(
                    "Choose how to colour samples on the plot.\n"
                    "• **Class** — use the main class labels\n"
                    "• **_meta columns** — colour by clinical/metadata variable "
                    "(categorical → distinct colours, numeric → gradient)"
                )
            )
            if _meta_avail_dr:
                st.caption(
                    f"📋 {len(_meta_avail_dr)} metadata column(s) available: "
                    + ", ".join(f"`{m}`" for m in _meta_avail_dr)
                )

            apply_reduction = st.form_submit_button("✅ Apply Reduction")

        if apply_reduction:
            # --- Récupération des données ---
            if data_source == 'Raw Data':
                df = st.session_state.get('final_data', st.session_state.get('data'))
            elif data_source == 'Preprocessed':
                df = st.session_state.get('preprocessed_data')
            elif data_source == 'Oversampled':
                df = st.session_state.get('oversampled_data')
            elif data_source == 'Undersampled':
                df = st.session_state.get('undersampled_data')
            else:
                df = None

            if df is None:
                st.warning("Selected data source not available.")
            else:
                drop_cols = [c for c in df.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')]
                try:
                    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include='number')
                    if X.empty or X.shape[1] == 0:
                        st.error("⚠️ No numeric feature columns found. Please verify the data source and format.")
                        st.stop()
                    if data_source == 'Raw Data' and X.isna().any().any():
                        from sklearn.impute import SimpleImputer
                        X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
                    y = df['Class']
                except Exception as e:
                    st.error(f"Error preparing data: {e}")
                    st.stop()

                if n_components > X.shape[1]:
                    st.error(f"Number of components ({n_components}) exceeds number of features ({X.shape[1]}).")
                    st.stop()

                st.session_state["X_reduction"] = X
                st.session_state["y_reduction"] = y
                st.session_state["n_components_reduction"] = n_components

                feature_col = "None"

                # --- PCA / UMAP / t-SNE ---
                try:
                    if method == "PCA":
                        svd_data, (loadings, explained_variance) = apply_pca(X, n_components, random_state=1)
                        st.session_state.update({
                            'compressed_data': svd_data,
                            'svd_model': (loadings, explained_variance),
                            'svd_available': True,
                            'reduction_method': "PCA"
                        })
                        # st.session_state["fig_initial"] = plot_pca(svd_data, y, st.session_state['class_colors'], feature_col, X)
                        # st.session_state["pca_fig"] = st.session_state["fig_initial"]

                        fig = plot_pca(
                            svd_data,
                            y,
                            st.session_state['class_colors'],
                            feature_col,
                            X,
                            capture_name="pca_fig",
                            color_by=st.session_state.get("dim_color_by", "Class"),
                            data_orig=df,
                        )

                        st.session_state["fig_initial"] = fig

                    elif method == "UMAP":
                        df_umap = X.assign(Class=y)
                        st.session_state.update({'compressed_data': df_umap, 'reduction_method': "UMAP"})
                        # st.session_state["fig_initial"] = plot_umap(df_umap, num_components=n_components,
                        #                                         custom_colors=st.session_state['class_colors'],
                        #                                         feature_intensity=feature_col)
                        # st.session_state["umap_fig"] = st.session_state["fig_initial"]

                        fig = plot_umap(
                            df_umap,
                            num_components=n_components,
                            custom_colors=st.session_state['class_colors'],
                            feature_intensity=feature_col,
                            capture_name="umap_fig",
                            color_by=st.session_state.get("dim_color_by", "Class"),
                            data_orig=df,
                        )

                        st.session_state["fig_initial"] = fig

                    elif method == "t-SNE":
                        df_tsne = X.assign(Class=y)
                        st.session_state.update({'compressed_data': df_tsne, 'reduction_method': "t-SNE"})
                        # st.session_state["fig_initial"] = plot_tsne(df_tsne, num_components=n_components,
                        #                                             custom_colors=st.session_state['class_colors'],
                        #                                             feature_intensity=feature_col)
                        # st.session_state["tsne_fig"] = st.session_state["fig_initial"]

                        fig = plot_tsne(
                            df_tsne,
                            num_components=n_components,
                            custom_colors=st.session_state['class_colors'],
                            feature_intensity=feature_col,
                            capture_name="tsne_fig",
                            color_by=st.session_state.get("dim_color_by", "Class"),
                            data_orig=df,
                        )

                        st.session_state["fig_initial"] = fig

                except Exception as e:
                    st.error(f"Error during dimensionality reduction: {e}")

        # --- Affichage du premier plot ---
        if st.session_state["fig_initial"] is not None:
            st.plotly_chart(st.session_state["fig_initial"], use_container_width=True, key="fig_initial")

        # --- FORM 2 : Feature Intensity + Regenerate Plot ---
        if st.session_state["compressed_data"] is not None:
            X = st.session_state["X_reduction"]
            y = st.session_state["y_reduction"]

            with st.form("feature_intensity_form"):
                feature_col = st.selectbox("Feature Intensity", ['None'] + list(X.columns), key="feature_intensity")
                refresh_plot = st.form_submit_button("✅  Show Feature")

            if refresh_plot:
                method = st.session_state.get("reduction_method")
                try:
                    if method == "PCA":
                        st.session_state["fig_feature"] = plot_pca(st.session_state["compressed_data"], y,
                                                                st.session_state['class_colors'], feature_col, X)
                    elif method == "UMAP":
                        st.session_state["fig_feature"] = plot_umap(st.session_state["compressed_data"],
                                                                    num_components=st.session_state["n_components_reduction"],
                                                                    custom_colors=st.session_state['class_colors'],
                                                                    feature_intensity=feature_col)
                    elif method == "t-SNE":
                        st.session_state["fig_feature"] = plot_tsne(st.session_state["compressed_data"],
                                                                    num_components=st.session_state["n_components_reduction"],
                                                                    custom_colors=st.session_state['class_colors'],
                                                                    feature_intensity=feature_col)
                except Exception as e:
                    st.error(f"Error regenerating plot: {e}")

        # --- Affichage du plot Feature Intensity ---
        if st.session_state.get("fig_feature") is not None:
            st.plotly_chart(st.session_state["fig_feature"], use_container_width=True, key="fig_feature")
            _capture_plotly(st.session_state["fig_feature"], "Features per Sample Feature Intensity")




        # --- FORM 3 : PCA Details ---
        if st.session_state.get("reduction_method") == "PCA" and st.session_state.get("svd_available", False):
            st.markdown("**PCA Details**")
            with st.form("pca_details_form"):
                loadings, explained_variance = st.session_state['svd_model']
                X = st.session_state["X_reduction"]

                pc_index = st.selectbox("Select Principal Component",
                                        [f"PC{i+1}" for i in range(len(explained_variance))],
                                        index=0, key="pc_index_select")
                top_n = st.number_input("Top Features to Display", min_value=5, max_value=50,
                                        value=10, step=1, key="top_loading_features_input")
                show_contrib = st.form_submit_button("✅ Show PCA Contributions")

            if show_contrib:
                try:
                    pc_idx = int(pc_index.replace("PC", "")) - 1

                    st.write("**Explained Variances**")
                    for i, var in enumerate(explained_variance):
                        st.write(f"PC{i + 1}: {var:.4f}")

                    contribs = pd.DataFrame({
                        'Feature': X.columns,
                        'Contribution': loadings[:, pc_idx],
                    }).assign(Abs_Contribution=lambda df: df['Contribution'].abs()) \
                        .sort_values(by='Abs_Contribution', ascending=False)

                    # st.dataframe(contribs, hide_index=True)
                    st.dataframe(contribs)

                    top_contribs = contribs.head(top_n).sort_values(by="Contribution")

                    # --- Plotly Bar Chart ---
                    fig = px.bar(top_contribs, x="Contribution", y="Feature", orientation='h',
                                text="Contribution", color="Contribution", color_continuous_scale='Blues')
                    fig.update_layout(title=f"Top {top_n} Feature Contributions to {pc_index}",
                                    xaxis_title="Loading (Contribution)",
                                    yaxis_title="Feature",
                                    yaxis=dict(autorange="reversed"),  # inverser l’ordre pour barh
                                    plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                    _capture_plotly(fig, "Features per Sample PCA Contributions")

                except Exception as e:
                    st.error(f"Error displaying PCA information: {e}")




    with _t4_clust:
        st.markdown(
            '<p style="color: gray; font-size: 14px">'
            'Assessing Group Formation and Heterogeneity'
            '</p>', unsafe_allow_html=True
        )

        # ----- Formulaire Silhouette -----
        with st.form("silhouette_form"):
            # --- Shared Inputs ---
            # Data source comes from the global selector above
            _clust_src_map2 = {"Raw Data": "Raw data", "Preprocessed": "Preprocessed",
                               "Oversampled": "Preprocessed + Oversampled", "Undersampled": "Preprocessed + Undersampled"}
            data_source = _clust_src_map2.get(_t4_src, "None")

            scaler_option = st.selectbox(
                "Select Scaler",
                ['StandardScaler', 'RobustScaler', 'MinMaxScaler'],
                help="Choose a method to normalize your features before clustering.",
                key="form_scaler"
            )

            cluster_range_input = st.text_input("Enter Range of Clusters for Silhouette (e.g., 2-10)", "2-10")
            run_silhouette = st.form_submit_button("Run Silhouette Analysis")

            # --- Préparation des données ---
            df, X_scaled, n_samples = None, None, 0
            if data_source != 'None':
                source_map = {
                    'Raw data': 'data',
                    'Preprocessed': 'preprocessed_data',
                    'Preprocessed + Oversampled': 'oversampled_data',
                    'Preprocessed + Undersampled': 'undersampled_data'
                }
                data_kmeans = st.session_state.get(source_map.get(data_source))

                # Si Raw data mais final_data existe, utiliser final_data à la place
                if data_source == 'Raw data' and st.session_state.get('final_data') is not None:
                    data_kmeans = st.session_state['final_data']

                if data_kmeans is not None:
                    df = data_kmeans.copy()
                    drop_cols = [c for c in data_kmeans.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')]
                    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore').select_dtypes(include='number').fillna(0)
                    if X.empty or X.shape[1] == 0:
                        st.warning("⚠️ No numeric feature columns found in the selected data source. "
                                   "If you loaded MaxQuant or another specialised format, "
                                   "please ensure the data was loaded correctly.")
                    else:
                        scaler_cls = {'StandardScaler': StandardScaler, 'RobustScaler': RobustScaler, 'MinMaxScaler': MinMaxScaler}[scaler_option]
                        X_scaled = scaler_cls().fit_transform(X)
                        n_samples = X_scaled.shape[0]

            # --- Exécution silhouette ---
            if run_silhouette:
                if X_scaled is None:
                    st.warning("No data available for the selected source.")
                else:
                    with st.spinner("Running Silhouette Analysis..."):
                        try:
                            start, end = map(int, cluster_range_input.split('-'))
                            scores = []
                            progress = st.progress(0)
                            total = end - start + 1
                    
                            for i, n in enumerate(range(start, end + 1)):
                                km = KMeans(n_clusters=n, n_init="auto", max_iter=300, tol=0.01, random_state=1)
                                labels = km.fit_predict(X_scaled)
                                score = silhouette_score(X_scaled, labels)
                                scores.append(score)
                                progress.progress((i + 1) / total)
                                time.sleep(0.05)
                            import plotly.express as px
                            fig = px.line(
                                x=range(start, end + 1),
                                y=scores,
                                markers=True,
                                title='Silhouette Score vs. Number of Clusters'
                            )
                            fig.update_layout(xaxis_title='Number of Clusters', yaxis_title='Silhouette Score')
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                            _capture_plotly(fig, "Silhouette Score vs. Number of Clusters")

                        except ValueError:
                            st.error("Please enter a valid range (e.g., 2-10).")
                        finally:
                            del scores, km, labels, score
                            gc.collect()

        # ----- Formulaire Clustering Spécifique -----
        with st.form("specific_clustering_form"):
            # Même data_source et scaler que le formulaire silhouette
            k = st.number_input("Enter Specific Number of Clusters", min_value=2, max_value=10, value=4)
            method = st.selectbox("Select Dimensionality Reduction Method for vizualisation", ['PCA', 't-SNE', 'UMAP'])
            dims = st.selectbox("Select Number of Dimensions for vizualisation", [2, 3])
            apply_clustering = st.form_submit_button("Apply Specific Clustering")

            if apply_clustering:
                if X_scaled is None:
                    st.warning("No data available for the selected source.")
                else:
                    with st.spinner("Applying Clustering..."):
                        progress = st.progress(0.3)
                        time.sleep(0.05)

                        km = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=0.01, random_state=1)
                        labels = km.fit_predict(X_scaled)
                        label_chars = [chr(ord('A') + i) for i in labels]
                        df['Class'] = label_chars

                        if method == 'PCA':
                            reducer = PCA(n_components=dims)
                        elif method == 't-SNE':
                            perplexity = max(5, min(int(np.sqrt(n_samples)), 50))
                            reducer = TSNE(n_components=dims, perplexity=perplexity, n_jobs=-1, n_iter=500, method="barnes_hut")
                        elif method == 'UMAP':
                            n_neighbors = max(2, min(int(np.log2(n_samples)), 100))
                            reducer = umap.UMAP(n_components=dims, n_neighbors=n_neighbors, n_jobs=-1, low_memory=False)

                        progress.progress(0.6)
                        time.sleep(0.05)

                        reduced = reducer.fit_transform(X_scaled)

                        if dims == 2:
                            import plotly.express as px
                            fig = px.scatter(
                                x=reduced[:, 0],
                                y=reduced[:, 1],
                                color=label_chars,
                                title=f'{method} - Clustering (k={k})'
                            )
                            fig.update_layout(xaxis_title='Component 1', yaxis_title='Component 2')
                        else:
                            fig = px.scatter_3d(
                                x=reduced[:, 0],
                                y=reduced[:, 1],
                                z=reduced[:, 2],
                                color=label_chars,
                                title=f'{method} - Clustering (k={k})'
                            )
                            fig.update_layout(
                                scene=dict(
                                    xaxis_title='Component 1',
                                    yaxis_title='Component 2',
                                    zaxis_title='Component 3'
                                )
                            )

                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig, f"Features per Sample {method} Visualization with Clusters")
                        progress.progress(1.0)

                        st.write("###### Updated DataFrame with Cluster Labels")
                        # st.dataframe(df, hide_index=True)
                        st.dataframe(df)
                        st.write("###### Cluster Distribution")
                        st.write(df['Class'].value_counts())

                        del reduced, reducer, km, labels, label_chars, fig
                        gc.collect()

        if df is not None:
            del X, X_scaled, df
            gc.collect()




    # Expander pour Machine Learning
    with _t4_ml:
        st.markdown(
            '<p style="color: gray; font-size: 14px">'
            'Over 20 Machine Learning Models Available – Classification & Regression'
            '</p>',
            unsafe_allow_html=True
        )

        source_map = {
            'Raw data': 'data',
            'Preprocessed': 'preprocessed_data',
            'Preprocessed + Oversampled': 'oversampled_data',
            'Preprocessed + Undersampled': 'undersampled_data'
        }

        # ======================================================
        # FORMULAIRE ML TRAINING
        # ======================================================

        with st.form("ml_training_form"):

            # Data source comes from the global selector above
            _ml_src_map2 = {"Raw Data": "Raw data", "Preprocessed": "Preprocessed",
                            "Oversampled": "Preprocessed + Oversampled", "Undersampled": "Preprocessed + Undersampled"}
            data_source = _ml_src_map2.get(_t4_src, "None")

            # -------------------------
            # Smart task detection
            # -------------------------
            task_options = ["Classification"]
            sample_data = None

            if data_source != "None":
                sample_data = st.session_state.get(source_map.get(data_source))

            # ── Target column selector (Class + _meta options) ──────────
            _target_options = ["Class"]
            if sample_data is not None:
                _meta_avail = [c for c in sample_data.columns if str(c).endswith("_meta")]
                _target_options = ["Class"] + _meta_avail

            selected_target_col = st.selectbox(
                "Target Column",
                _target_options,
                index=0,
                key="form_target_col",
                help="Default: Class column. Select a _meta column to train on clinical metadata instead."
            )

            if sample_data is not None and selected_target_col in sample_data.columns:
                y_sample = sample_data[selected_target_col]

                if sample_data is not None and selected_target_col in sample_data.columns:
                    y_sample = sample_data[selected_target_col]

                    if pd.api.types.is_numeric_dtype(y_sample):
                        n_unique = y_sample.nunique()

                        if n_unique <= 10:
                            task_options = ["Classification", "Regression"]
                            st.info(f"Numeric target detected ({n_unique} unique values). Choose task type below.")
                        else:
                            task_options = ["Regression"]
                            st.info(f"Continuous target detected ({n_unique} unique values) → Regression recommended.")
                    else:
                        st.info(f"Categorical target detected ({y_sample.nunique()} classes) → Classification.")

            task_type = st.selectbox(
                "Task Type",
                task_options,
                help="Classification for discrete labels, Regression for continuous targets."
            )

            # -------------------------
            # Dimensionality Reduction
            # -------------------------
            apply_reduction = st.checkbox(
                "Apply Dimensionality Reduction",
                key="form_apply_reduction",
                help="Reduce feature dimensions using PCA, UMAP, or t-SNE."
            )

            reduction_choice, n_components = None, None

            if apply_reduction:
                reduction_choice = st.selectbox(
                    "Reduction Technique",
                    ['PCA', 'UMAP', 't-SNE'],
                    key="form_reduction_choice"
                )
                n_components = st.number_input(
                    "Number of Components",
                    min_value=2, max_value=200, value=2, step=1,
                    key="form_n_components"
                )

            # Plus rapide par défaut
            n_splits = st.number_input(
                "Number of Splits for Cross-Validation",
                min_value=2, max_value=50, value=3, step=1,
                key="form_n_splits",
                help="Number of folds for cross-validation. (3 is faster, 5 is more stable)"
            )

            #  Calibration (optionnelle, classification uniquement)
            calibrate_models = st.checkbox(
                "Calibrate models (slower but better probabilities)",
                value=False,
                key="form_calibrate_models",
                help=(
                    "Calibration improves probability estimates (confidence scores). "
                    "⚠️ This makes training much slower because it adds an internal cross-validation. "
                    "Only applies to Classification."
                )
            )

            st.caption(
                "💡 Tip: Keep calibration OFF for faster training. "
                "Turn it ON only if you really need to extract probabilities for: "
                "RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier."
            )

            train_models_btn = st.form_submit_button("Train Machine Learning Models")

        # ======================================================
        # PRÉPARATION DES DONNÉES
        # ======================================================

        X, y = None, None

        if data_source != 'None':
            data_train = st.session_state.get(source_map.get(data_source))

            # Raw data → remplacer par final_data si existant
            if data_source == 'Raw data' and st.session_state.get('final_data') is not None:
                data_train = st.session_state['final_data']

            if data_train is not None:
                # Use selected_target_col if it exists, else fallback to 'Class'
                _tgt = st.session_state.get("form_target_col", "Class")
                if _tgt not in data_train.columns:
                    _tgt = "Class"
                if _tgt not in data_train.columns:
                    st.error("No target column found in the dataset.")
                else:
                    # exclude metadata+ID but keep selected target for y
                    drop_cols = [c for c in data_train.columns
                                 if c in _NON_FEATURE_COLS
                                 or (str(c).endswith('_meta') and c != _tgt)]
                    X = data_train.drop(
                        columns=[col for col in drop_cols if col in data_train.columns],
                        errors='ignore'
                    ).select_dtypes(include='number')
                    y = data_train[_tgt]

        # ======================================================
        # EXÉCUTION DU TRAINING
        # ======================================================

        if train_models_btn:

            if X is None or y is None:
                st.error("Dataset invalid or missing 'Class' column. Please select a valid data source.")
                st.stop()

            # Safety checks
            if task_type == "Regression":
                if not pd.api.types.is_numeric_dtype(y):
                    st.error("Regression requires a numeric target column.")
                    st.stop()

            if task_type == "Classification":
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
                    st.error("Target appears continuous (>20 unique values). Please choose Regression.")
                    st.stop()

            feature_names = X.columns.tolist()

            # -------------------------
            # Réduction dimensionnelle
            # -------------------------
            if apply_reduction and reduction_choice:

                if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                    st.error("Dimensionality reduction requires numeric features only.")
                    st.stop()

                n_samples = X.shape[0]

                if reduction_choice == 'PCA':
                    reducer = PCA(n_components=n_components)
                    X = reducer.fit_transform(X)

                elif reduction_choice == 'UMAP':
                    reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=max(2, min(int(np.log2(n_samples)), 100)),
                        random_state=1,
                        n_jobs=-1,
                        low_memory=False,
                        verbose=False
                    )
                    X = reducer.fit_transform(X)

                elif reduction_choice == 't-SNE':
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=max(5, min(int(np.sqrt(n_samples)), 50)),
                        random_state=1,
                        n_jobs=-1,
                        n_iter=500 if n_samples > 2000 else 1000,
                        method='barnes_hut'
                    )
                    X = tsne.fit_transform(X)

                feature_names = [f"Component {i+1}" for i in range(n_components)]

            st.session_state['reduced_data'] = pd.DataFrame(X, columns=feature_names)
            X = st.session_state['reduced_data']

            # Save X/y for report & explainability (SHAP/LIME)
            st.session_state["ml_X_used"] = X
            st.session_state["ml_y_used"] = y
            st.session_state["ml_task_type"] = task_type
            st.session_state["ml_n_splits"] = n_splits

            try:
                if task_type == "Classification":
                    class_counts = Counter(y)
                    too_few_classes = [cls for cls, count in class_counts.items() if count < n_splits]

                    if too_few_classes:
                        st.error(
                            f"The following class(es) have fewer samples than the number of CV splits ({n_splits}): "
                            f"{too_few_classes}"
                        )
                        st.stop()

                    if len(class_counts) < 2:
                        st.error("At least two classes are required for classification.")
                        st.stop()

                st.info("⏳ Training models... This may take a few minutes.")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("Starting training...")

                if task_type == "Classification":
                    model_results = train_models(
                        X,
                        y,
                        n_splits=n_splits,
                        progress_bar=progress_bar,
                        calibrate=calibrate_models
                    )
                else:
                    model_results = train_regression_models(
                        X,
                        y,
                        n_splits=n_splits,
                        progress_bar=progress_bar
                    )

                st.session_state['models'] = model_results
                # Desktop: session_state only
                status_text.success("Training complete!")
                st.success("✅ Models trained successfully!")

            except RuntimeError as e:
                st.error(f"Runtime error: {e}")
            except MemoryError:
                st.error("MemoryError: Reduce dataset size or number of models.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                with st.expander("Show full traceback"):
                    st.code(traceback.format_exc(), language="python")

        del X, y
        gc.collect()

        # ==========================================================
        # 📊 MODEL ANALYSIS & COMPARISON
        # ==========================================================

        if 'models' in st.session_state and st.session_state['models']:

            models_dict = st.session_state['models']
            task_type   = st.session_state.get("ml_task_type", "Classification")
            n_splits    = st.session_state.get("ml_n_splits", 3)
            X_used      = st.session_state.get("ml_X_used")
            y_used      = st.session_state.get("ml_y_used")

            with st.form("ml_report_form"):

                show_comparison_btn = st.form_submit_button("Show Model Comparison")

                model_options = ['None'] + list(models_dict.keys())

                selected_model = st.selectbox(
                    "Select ML Model for Report",
                    model_options,
                    key="form_selected_model"
                )

                # Synchroniser avec la clé utilisée par SHAP/LIME
                st.session_state["selected_model"] = selected_model

                show_report_btn = st.form_submit_button("Show Model Report")

            # ======================================================
            # REPORT
            # ======================================================

            if show_report_btn and selected_model != 'None':

                model_data = models_dict[selected_model]

                # =========================
                # CLASSIFICATION
                # =========================
                if task_type == "Classification":

                    st.subheader("Classification Report")
                    report_data = model_data['classification_report']

                    if isinstance(report_data, dict):
                        report_df = pd.DataFrame(report_data).transpose()

                        global_metrics = report_df.loc[
                            [r for r in ['accuracy', 'macro avg', 'weighted avg'] if r in report_df.index]
                        ]
                        class_metrics = report_df.drop(
                            [r for r in ['accuracy', 'macro avg', 'weighted avg'] if r in report_df.index]
                        )

                        combined_df = pd.concat([
                            class_metrics,
                            pd.DataFrame([[''] * len(class_metrics.columns)], columns=class_metrics.columns),
                            global_metrics
                        ])

                        numeric_cols = [c for c in ['precision', 'recall', 'f1-score', 'support'] if c in combined_df.columns]
                        combined_df[numeric_cols] = combined_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                        st.dataframe(
                            combined_df.style
                            .format("{:.4f}", subset=numeric_cols)
                            .highlight_max(subset=numeric_cols, axis=0, color='lightgreen')
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', '#f0f2f6'),
                                                             ('font-size', '18px'),
                                                             ('text-align', 'center'),
                                                             ('font-weight', 'bold')]},
                                {'selector': 'td', 'props': [('font-size', '16px'),
                                                             ('text-align', 'center')]},
                                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
                                {'selector': 'tr:nth-child(odd)',  'props': [('background-color', 'white')]}
                            ])
                            .set_properties(**{'border': '1px solid #ddd', 'padding': '8px'}),
                            use_container_width=True,
                            height=400
                        )

                    elif isinstance(report_data, str):
                        st.code(report_data)
                    else:
                        st.code(report_data)

                    # ---------- Confusion Matrices ----------
                    FIG_SIZE       = 850
                    TITLE_SIZE     = 20
                    AXIS_TITLE_SIZE = 24
                    TICK_SIZE      = 20
                    FONT_FAMILY    = "Arial"

                    labels = model_data['label_encoder'].classes_
                    cm     = model_data['confusion_matrix']

                    fig = go.Figure(data=go.Heatmap(
                        z=cm, x=labels, y=labels,
                        colorscale='Viridis',
                        text=cm,
                        texttemplate="<b>%{text}</b>",
                        textfont=dict(size=26, family=FONT_FAMILY),
                        hoverinfo="z"
                    ))
                    fig.update_layout(
                        title=dict(text="Confusion Matrix",
                                   font=dict(size=TITLE_SIZE, color="black", family=FONT_FAMILY)),
                        autosize=True,
                        margin=dict(l=50, r=50, b=50, t=80),
                        xaxis=dict(
                            title=dict(text="Predicted label",
                                       font=dict(size=AXIS_TITLE_SIZE, color="black", family=FONT_FAMILY)),
                            tickfont=dict(size=TICK_SIZE, color="black", family=FONT_FAMILY),
                            scaleanchor="y", constrain="domain"
                        ),
                        yaxis=dict(
                            title=dict(text="True label",
                                       font=dict(size=AXIS_TITLE_SIZE, color="black", family=FONT_FAMILY)),
                            tickfont=dict(size=TICK_SIZE, color="black", family=FONT_FAMILY),
                            scaleanchor="x", constrain="domain"
                        ),
                        font=dict(size=TICK_SIZE, color="black", family=FONT_FAMILY)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                    _capture_plotly(fig, "ml_confusion")

                    # Normalized
                    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
                    fig_norm = go.Figure(data=go.Heatmap(
                        z=cm_norm, x=labels, y=labels,
                        colorscale='jet',
                        text=np.round(cm_norm, 2),
                        texttemplate="<b>%{text}%</b>",
                        textfont=dict(size=11, family=FONT_FAMILY),
                        hoverinfo="z"
                    ))
                    fig_norm.update_layout(
                        title=dict(text="Normalized Confusion Matrix (%)",
                                   font=dict(size=TITLE_SIZE, color="black", family=FONT_FAMILY)),
                        autosize=True,
                        margin=dict(l=50, r=50, b=50, t=80),
                        xaxis=dict(
                            title=dict(text="Predicted label",
                                       font=dict(size=AXIS_TITLE_SIZE, color="black", family=FONT_FAMILY)),
                            tickfont=dict(size=TICK_SIZE, color="black", family=FONT_FAMILY),
                            scaleanchor="y", constrain="domain"
                        ),
                        yaxis=dict(
                            title=dict(text="True label",
                                       font=dict(size=AXIS_TITLE_SIZE, color="black", family=FONT_FAMILY)),
                            tickfont=dict(size=TICK_SIZE, color="black", family=FONT_FAMILY),
                            scaleanchor="x", constrain="domain"
                        ),
                        font=dict(size=TICK_SIZE, color="black", family=FONT_FAMILY)
                    )
                    st.plotly_chart(fig_norm, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                    _capture_plotly(fig_norm, "ml_confusion_norm")

                    # ---------- ROC Curves ----------
                    try:
                        st.markdown("#### ROC Curves")
                        # Use stored probas from model_data if available
                        if model_data.get('probas') is not None:
                            fig_roc = plot_roc_curves({selected_model: model_data})
                        else:
                            # Fallback: plot for all models that have probas
                            models_with_probas = {k: v for k, v in models_dict.items() if v.get('probas') is not None}
                            fig_roc = plot_roc_curves(models_with_probas) if models_with_probas else None
                        if fig_roc is not None:
                            st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                            _capture_plotly(fig_roc, "ml_roc")
                        else:
                            st.info("ROC curves not available (model does not support probability output).")
                    except Exception as e:
                        st.warning(f"ROC curves could not be computed: {e}")

                    # ---------- Learning Curve ----------
                    try:
                        learning_curve_fig = plot_learning_curve(
                            model_data['model'], X_used, y_used, n_splits=n_splits
                        )
                        st.plotly_chart(learning_curve_fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(learning_curve_fig, "ml_learning_curve")
                    except Exception as e:
                        st.error(f"Error plotting learning curve: {e}")

                # =========================
                # REGRESSION
                # =========================
                else:

                    st.subheader("Regression Report")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("R²",   round(model_data["r2_score"], 4))
                    col2.metric("MAE",  round(model_data["mae"],      4))
                    col3.metric("RMSE", round(model_data["rmse"],     4))

                    y_true = model_data.get("y_true")
                    y_pred = model_data.get("y_pred")

                    if y_true is not None and y_pred is not None:
                        fig = px.scatter(
                            x=y_true, y=y_pred,
                            labels={"x": "True Values", "y": "Predicted Values"},
                            title="Predicted vs True Values",
                            trendline="ols"
                        )
                        fig.add_shape(
                            type="line",
                            x0=min(y_true), y0=min(y_true),
                            x1=max(y_true), y1=max(y_true),
                            line=dict(color="red", dash="dash"),
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig, "pred_vs_true_fig")

                        # Residuals
                        residuals = np.array(y_pred) - np.array(y_true)
                        fig_res = px.scatter(
                            x=y_pred, y=residuals,
                            labels={"x": "Predicted", "y": "Residuals"},
                            title="Residual Plot"
                        )
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_res, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_res, "residual_plot_fig")

            # ======================================================
            # COMPARISON
            # ======================================================

            if show_comparison_btn:

                if task_type == "Classification":

                    comparison_df = pd.DataFrame({
                        name: {
                            "Accuracy": m.get("accuracy"),
                            "F1-score": m.get("f1_score")
                        }
                        for name, m in models_dict.items()
                    }).T

                else:

                    comparison_df = pd.DataFrame({
                        name: {
                            "R²":   m.get("r2_score"),
                            "MAE":  m.get("mae"),
                            "RMSE": m.get("rmse")
                        }
                        for name, m in models_dict.items()
                    }).T

                # fig = px.bar(
                #     comparison_df.reset_index().rename(columns={"index": "Model"}),
                #     x="Model",
                #     y=comparison_df.columns.tolist(),
                #     barmode="group",
                #     title=f"Model Comparison – {task_type}"
                # )

                # st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                # _capture_plotly(fig, "model_comparison_fig")
                if task_type == "Classification":
                    fig = compare_models(models_dict)
                else:
                    fig_r2, fig_rmse = compare_regression_models(models_dict)
                    fig = fig_r2
                    st.plotly_chart(fig_rmse, use_container_width=True,
                                    config={'displayModeBar': True, 'displaylogo': False,
                                            'scrollZoom': True})
                    _capture_plotly(fig_rmse, "model_comparison_rmse_fig")

                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': True, 'displaylogo': False,
                                        'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'],
                                        'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                _capture_plotly(fig, "model_comparison_fig")


    with _t4_dl:
        st.markdown('<p style="color: gray; font-size: 14px ">CNN, RNN, and MLP Deep Learning Models for Advanced Tasks</p>', unsafe_allow_html=True)
        n_splits = st.number_input("Number of Splits", min_value=2, max_value=10, value=2, step=1, key="dl_n_splits", help="Higher values provide better generalization estimates but increase training time.")
        #epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1, key="dl_epochs")
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="dl_epochs",
            help=(
                "Number of complete passes through the training dataset. "
                "Too few may lead to underfitting, too many may cause overfitting. "
                "Start with 10–20 and adjust based on performance."
            )
        )                
        # batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32, step=1, key="dl_batch_size")
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=128,
            value=32,
            step=1,
            key="dl_batch_size",
            help=(
                "Number of samples used per gradient update. "
                "Smaller batches give noisier but more frequent updates. "
                "Larger batches are more stable but require more memory. "
                "A batch size of 32 is a common starting point."
            )
        )                
        learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.9, value=0.001, step=0.0001, key="dl_learning_rate", format="%.3f", help="Lower values make learning slower but more stable. Try 0.001 as a starting point.")

        # Data source comes from the global selector above
        _dl_src_map2 = {"Raw Data": "Raw data", "Preprocessed": "Preprocessed",
                        "Oversampled": "Preprocessed + Oversampled", "Undersampled": "Preprocessed + Undersampled"}
        data_source = _dl_src_map2.get(_t4_src, "None")

        X, y = None, None
        if data_source == 'Raw data' and 'data' in st.session_state and st.session_state['data'] is not None:
            _tmp_df = st.session_state['data']
            X = _tmp_df.drop([c for c in _tmp_df.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')], axis=1, errors='ignore').select_dtypes(include='number')
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')  
            X = imputer.fit_transform(X)  
            y = st.session_state['data']['Class']
        if data_source == 'Preprocessed' and 'preprocessed_data' in st.session_state and st.session_state['preprocessed_data'] is not None:
            _tmp_df = st.session_state['preprocessed_data']
            X = _tmp_df.drop([c for c in _tmp_df.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')], axis=1, errors='ignore').select_dtypes(include='number')
            y = st.session_state['preprocessed_data']['Class']
        elif data_source == 'Preprocessed + Oversampled' and 'oversampled_data' in st.session_state and st.session_state['oversampled_data'] is not None:
            _tmp_df = st.session_state['oversampled_data']
            X = _tmp_df.drop([c for c in _tmp_df.columns if c in {'Class','ID','File','RT','Sum'} or str(c).endswith('_meta')], axis=1, errors='ignore')
            y = st.session_state['oversampled_data']['Class']
        elif data_source == 'Preprocessed + Undersampled' and 'undersampled_data' in st.session_state and st.session_state['undersampled_data'] is not None:
            _tmp_df = st.session_state['undersampled_data']
            X = _tmp_df.drop([c for c in _tmp_df.columns if c in {'Class','ID','File','RT','Sum'} or str(c).endswith('_meta')], axis=1, errors='ignore')
            y = st.session_state['undersampled_data']['Class']
        elif data_source == 'None':
            st.warning("Please select a valid data source.")
        if X is not None:
            if st.button("Train Deep Learning Models", key="train_dl_models"):
                try:
                    progress_bar = st.progress(0)
                    model_results = train_DL(X, y, n_splits=n_splits, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                    st.session_state['dl_models'] = model_results
                    # Desktop: session_state only
                    st.success("Deep Learning models trained successfully!")
                except Exception as e:
                    st.error(f"Error during DL model training: {e}")

        # if st.button("Deep Learning Model Comparison", key="show_dl_model_comparison") and 'dl_models' in st.session_state:
        #     st.plotly_chart(compare_DL(st.session_state['dl_models'], config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}}))
    

        if st.button("Deep Learning Model Comparison", key="show_dl_model_comparison") and 'dl_models' in st.session_state:
            fig = compare_DL(st.session_state['dl_models'])

            st.plotly_chart(
                fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['downloadImage'],
                    'toImageButtonOptions': {'format': 'png', 'scale': 2}
                }
            )

        selected_dl_model = st.selectbox("Select DL Model for Report", ['None'] + list(st.session_state['dl_models'].keys()), key="selected_dl_model")
        if selected_dl_model != 'None':
            dl_model_data = st.session_state['dl_models'][selected_dl_model]
            if st.button("DL Model Results", key="show_dl_model_results"):
                X_test = dl_model_data['X_test']
                y_test = dl_model_data['y_test']
                history = dl_model_data['history']
                label_encoder = dl_model_data['label_encoder']  # Récupérez le label_encoder à partir des résultats
                display_model_results(dl_model_data['model'], X_test, y_test, history, label_encoder)

            if st.button("Show Cross-validated Results", key="show_global_results"):
                display_global_results(dl_model_data)
        del X
        del y
        gc.collect()



    # ════════════════════════════════════════════════════════
    # LONGITUDINAL TAB
    # ════════════════════════════════════════════════════════
    with _t4_long:
        # Prefer dedicated longitudinal_df (preserves Time/Subject_ID as strings)
        # Fall back to "data" for backwards compatibility
        # NOTE: cannot use `or` on DataFrames (ambiguous truth value) — use explicit check
        _long_df_candidate = st.session_state.get("longitudinal_df")
        if _long_df_candidate is not None:
            _long_data = _long_df_candidate
        elif st.session_state.get("is_longitudinal"):
            _long_data = st.session_state.get("data")
        else:
            _long_data = None
        if _long_data is None:
            st.markdown("""
    <div style='background:#fffbeb;border-left:4px solid #d97706;border-radius:8px;
    padding:12px 16px;margin:12px 0;'>
    <b style='color:#92400e;'>📈 Longitudinal Analysis</b><br>
    <span style='color:#78350f;font-size:0.85rem;'>
    No longitudinal dataset is currently loaded.<br><br>
    To use this module:<br>
    1. Go to the sidebar → <b>Step 5 — Longitudinal / Time-Series Data</b><br>
    2. Upload your dataset (one row per sample-timepoint)<br>
    3. Your file must contain a <code>Subject_ID</code> column and a <code>Time</code> column<br>
    4. Click <b>Load Longitudinal Data</b>
    </span>
    </div>
    """, unsafe_allow_html=True)
            st.markdown("#### Expected format")
            st.markdown("""
    | Subject_ID | Time | Class | Protein_A | Protein_B | age_meta |
    |---|---|---|---|---|---|
    | P01 | T0 | Ctrl | 1257 | 0.45 | 58 |
    | P01 | T1 | Ctrl | 1389 | 0.50 | 58 |
    | P02 | T0 | Case | 752 | 1.30 | 62 |
    | P02 | T1 | Case | 810 | 1.20 | 62 |

    **Requirements:**
    - `Subject_ID` (or `Patient_ID`, `patient`, `subject`) — identifies each individual across timepoints
    - `Time` (or `Timepoint`, `Visit`, `Week`) — numeric or categorical (T0, T1, Week4…)
    - `Class` — group/condition label
    - Any number of numeric feature columns (proteins, genes, metabolites…)
    """)
        else:
            render_longitudinal_tab(_long_data)

    with _t4_save:
        st.markdown('<p style="color: gray; font-size: 14px;">Storing Your Trained Models for Future Use.</p>', unsafe_allow_html=True)

        model_type_save = st.selectbox("Select Model Type", ['Machine Learning', 'Deep Learning'], key="model_type_save", help="Select which type of model you want to save and export.")

        selected_model = None
        selected_dl_model = None

        if model_type_save == 'Machine Learning':
            available_models = list(st.session_state.get('models', {}).keys()) if isinstance(st.session_state.get('models', {}), dict) else []
            selected_model = st.selectbox("Select Model", available_models, key="selected_ml_model") if available_models else None
        else:
            available_dl_models = list(st.session_state.get('dl_models', {}).keys()) if isinstance(st.session_state.get('dl_models', {}), dict) else []
            selected_dl_model = st.selectbox("Select Deep Learning Model", available_dl_models, key="selected_dl_model_save") if available_dl_models else None

        save_directory = "saved_models"
        os.makedirs(save_directory, exist_ok=True)
        auto_save =True

        if auto_save and (selected_model or selected_dl_model):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = selected_model if selected_model else selected_dl_model
            model_path = os.path.join(save_directory, f"{model_name}_{timestamp}_model.pkl")
            features_path = os.path.join(save_directory, f"{model_name}_{timestamp}_features.pkl")
            label_encoder_path = os.path.join(save_directory, f"{model_name}_{timestamp}_label_encoder.pkl")
        else:
            model_path = os.path.join(save_directory, "model.pkl")
            features_path = os.path.join(save_directory, "features.pkl")
            label_encoder_path = os.path.join(save_directory, "label_encoder.pkl")




        if st.button("Save Model", key="save_model"):
            try:
                if model_type_save == 'Machine Learning' and selected_model:
                    model_data = st.session_state['models'][selected_model]
                    model_name = selected_model
                elif model_type_save == 'Deep Learning' and selected_dl_model:
                    model_data = st.session_state['dl_models'][selected_dl_model]
                    model_name = selected_dl_model
                else:
                    st.error("No model selected!")
                    st.stop()

                model_accuracy = model_data.get('accuracy', None)


                reduced_data = st.session_state.get('reduced_data', None)
                preprocessed_data = st.session_state.get('preprocessed_data', None)

                if reduced_data is not None and hasattr(reduced_data, 'empty') and not reduced_data.empty:
                    feature_names = reduced_data.columns.tolist()
                elif preprocessed_data is not None and hasattr(preprocessed_data, 'empty') and not preprocessed_data.empty:
                    feature_names = preprocessed_data.drop([c for c in preprocessed_data.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')], axis=1, errors='ignore').columns.tolist()
                elif model_data.get('features', None) is not None:
                    feature_names = model_data['features']

                else:
                    st.error("No feature data available for saving.")
                    st.stop()

                model_buffer = io.BytesIO()
                joblib.dump({
                    'model': model_data['model'],
                    'accuracy': model_accuracy 
                }, model_buffer)
                model_buffer.seek(0)

                features_buffer = io.BytesIO()
                joblib.dump(feature_names, features_buffer)
                features_buffer.seek(0)

                label_encoder_buffer = io.BytesIO()
                joblib.dump(model_data.get('label_encoder', None), label_encoder_buffer)
                label_encoder_buffer.seek(0)

                # Session state pour download
                st.session_state.model_buffer = model_buffer
                st.session_state.features_buffer = features_buffer
                st.session_state.label_encoder_buffer = label_encoder_buffer
                st.session_state.model_name = model_name
                st.session_state.timestamp = timestamp
                st.session_state.show_model_downloads = True

                joblib.dump({'model': model_data['model'], 'accuracy': model_accuracy}, model_path)
                joblib.dump(feature_names, features_path)
                joblib.dump(model_data.get('label_encoder', None), label_encoder_path)

            except Exception as e:
                st.error(f"Error saving model: {e}")


        # Show download buttons if available
        if st.session_state.get("show_model_downloads", False):
            st.download_button(
                label="Download Model",
                data=st.session_state.model_buffer,
                file_name=f"{st.session_state.model_name}_{st.session_state.timestamp}_model.pkl",
                mime="application/octet-stream"
            )
            st.download_button(
                label="Download Features",
                data=st.session_state.features_buffer,
                file_name=f"{st.session_state.model_name}_{st.session_state.timestamp}_features.pkl",
                mime="application/octet-stream"
            )
            st.download_button(
                label="Download Label Encoder",
                data=st.session_state.label_encoder_buffer,
                file_name=f"{st.session_state.model_name}_{st.session_state.timestamp}_label_encoder.pkl",
                mime="application/octet-stream"
            )
            st.success(f"Model '{st.session_state.model_name}' is available for download!")








    with tabs[5]:
        # ═══════════════════════════════════════════════════
        # BIOMARKERS – single source selector + sub-tabs
        # ═══════════════════════════════════════════════════
        _t5_raw = st.session_state.get('final_data', st.session_state.get('data'))
        _t5_pp  = st.session_state.get('preprocessed_data')
        _t5_os  = st.session_state.get('oversampled_data')
        _t5_us  = st.session_state.get('undersampled_data')
        _t5_df = None
        _t5_src = None
        if _t5_raw is None and _t5_pp is None:
            st.info("📂 Load a dataset from the sidebar to access the biomarkers.", icon="ℹ️")
        else:
            _t5_avail = {k:v for k,v in {"Raw Data":_t5_raw,"Preprocessed":_t5_pp,"Oversampled":_t5_os,"Undersampled":_t5_us}.items() if v is not None}
            _t5_c1, _t5_c2 = st.columns([3,5])
            with _t5_c1:
                _t5_keys = list(_t5_avail.keys())
                _t5_default = _t5_keys.index("Preprocessed") if "Preprocessed" in _t5_keys else 0
                _t5_src = st.selectbox("🗂️ Data source (applies to all analyses below)", _t5_keys, index=_t5_default, key="global_t5_source")
            _t5_df = _t5_avail[_t5_src]
            with _t5_c2:
                if _t5_df is not None:
                    _n5s,_n5f,_n5c = _t5_df.shape[0],_t5_df.shape[1]-(1 if 'Class' in _t5_df.columns else 0),(_t5_df['Class'].nunique() if 'Class' in _t5_df.columns else 0)
                    st.markdown(f"<div style='padding:8px 0 0 8px;color:#64748b;font-size:0.8rem;'><span style='color:#318CE7;font-weight:700;'>{_n5s}</span> samples &nbsp;·&nbsp;<span style='color:#10b981;font-weight:700;'>{_n5f}</span> features &nbsp;·&nbsp;<span style='color:#f59e0b;font-weight:700;'>{_n5c}</span> classes</div>",unsafe_allow_html=True)
            st.markdown("<hr style='margin:10px 0 14px;border:none;border-top:1px solid #e2e8f0;'>",unsafe_allow_html=True)
        (_t5_volcano, _t5_heatmap, _t5_boxplot, _t5_shap, _t5_lime) = st.tabs([
            "  Volcano Plot","  Heatmap Clustering",
            "  Feature Boxplots","  SHAP Explainability","  LIME"])

    # ---------------- Volcano Plot Expander ----------------
    with _t5_volcano:
        st.markdown(
            '<p style="color: gray; font-size: 14px;">'
            'Significant features between conditions using p-value and fold change thresholds '
            'for both binary and multi-class.'
            '</p>',
            unsafe_allow_html=True
        )

        selected_features = []

        # ---------------- FORM ----------------
        with st.form("volcano_form"):
            # Data source comes from the global selector above
            _vol_src_map = {"Raw Data": "Raw data", "Preprocessed": "Preprocessed",
                            "Oversampled": "Oversampled", "Undersampled": "Undersampled"}
            data_source = _vol_src_map.get(_t5_src, "None")

            # Feature selection
            select_all_features_volcano = st.checkbox(
                "Select All Features for Volcano Plot",
                key="select_all_features_volcano",
                help="Check this box to select all features automatically."
            )

            features_inputvp = st.text_input(
                "Features",
                key="features_inputvp",
                help="Enter a comma-separated list of features to include in the Volcano Plot."
            )

            # Peak picking
            use_peak_picking = st.checkbox(
                "Use Feature Detection",
                key="use_peak_picking",
                help="Enable feature detection to automatically identify peaks."
            )

            intensity_threshold = st.number_input(
                "Peak Intensity Threshold",
                min_value=0.0, max_value=100000.0, value=0.01, step=1.0, format="%.5f",
                help="Set the intensity threshold for peak detection."
            )

            # Thresholds
            p_value_threshold = st.number_input(
                "Select P-Value Threshold",
                min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f",
                help="Set the p-value threshold to filter significant features."
            )
            fold_change_threshold = st.number_input(
                "Select Fold Change Threshold",
                min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f",
                help="Set the fold change threshold to filter significant features."
            )

            correction_method = st.selectbox(
                "Multiple-Testing Correction Method",
                ["FDR (Benjamini–Hochberg)", "Bonferroni", "None"],
                index=0,
                help=(
                    "Choose the method for correcting p-values for multiple comparisons:\n"
                    "- **FDR (Benjamini–Hochberg)**: less strict, recommended for omics data.\n"
                    "- **Bonferroni**: more conservative.\n"
                    "- **None**: no correction applied."
                )
            )

            # Highlight feature names
            highlight_features = st.checkbox(
                "Highlight Feature Names",
                value=False,
                key="highlight_features",
                help="Check this box to highlight feature names in the Volcano Plot."
            )

            submitted = st.form_submit_button("Display Volcano Plot")

        # ── Control class selector (outside form — always reactive) ──────────
        _vol_src_key_preview = {
            "Raw data": st.session_state.get('data'),
            "Preprocessed": st.session_state.get('preprocessed_data'),
            "Oversampled": st.session_state.get('oversampled_data'),
            "Undersampled": st.session_state.get('undersampled_data'),
        }.get(_vol_src_map.get(_t5_src, "None"))

        _vol_control_class = None
        if _vol_src_key_preview is not None and 'Class' in _vol_src_key_preview.columns:
            _vol_classes = list(_vol_src_key_preview['Class'].dropna().unique())
            if len(_vol_classes) == 2:
                st.markdown(
                    "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;"
                    "padding:10px 14px;margin:6px 0;font-size:0.85rem;color:#1e40af;'>"
                    "🔁 <b>Binary comparison detected</b> — choose which class is the <b>Control</b> "
                    "(denominator in Fold Change). Features <em>upregulated</em> are higher in the <em>other</em> class.</div>",
                    unsafe_allow_html=True
                )
                _vol_control_class = st.selectbox(
                    "Control class (reference / denominator)",
                    options=_vol_classes,
                    index=0,
                    key="volcano_control_class",
                    help=(
                        "The selected class is used as the denominator when computing Fold Change.\n"
                        "Fold Change = mean(Other) / mean(Control)\n"
                        "Upregulated = higher in Other vs Control."
                    )
                )

        # ---------------- ACTION ----------------
        if submitted:
            # Load selected data source
            data_vol = None
            if data_source == 'Raw data':
                data_vol = st.session_state.get('data')
            elif data_source == 'Preprocessed':
                data_vol = st.session_state.get('preprocessed_data')
            elif data_source == 'Oversampled':
                data_vol = st.session_state.get('oversampled_data')
            elif data_source == 'Undersampled':
                data_vol = st.session_state.get('undersampled_data')

            # Stocker la source choisie pour l'utiliser dans RESULTS
            st.session_state["volcano_data_source_df"] = data_vol
            # Stocker les paramètres pour le nom du fichier
            st.session_state["volcano_p_threshold"] = p_value_threshold
            st.session_state["volcano_fc_threshold"] = fold_change_threshold

            class_column = st.session_state.get('class_column', 'Class')

            if data_vol is None:
                st.warning("Please select a valid data source.")
            else:
                # --- Feature selection ---
                if select_all_features_volcano:
                    selected_features = [
                        col for col in data_vol.columns
                        if col not in _NON_FEATURE_COLS
                        and not str(col).endswith('_meta')
                        and pd.api.types.is_numeric_dtype(data_vol[col])
                    ]
                else:
                    selected_features = [f.strip() for f in features_inputvp.split(',')] if features_inputvp else []

                # Peak picking
                peak_features = detect_peaks(data_vol, intensity_threshold) if use_peak_picking else []
                selected_features = list(set(selected_features + peak_features))

                if not selected_features:
                    st.warning("Please select features to display the Volcano Plot.")
                else:
                    try:
                        with st.spinner("Generating Volcano Plot..."):

                            method_map = {
                                "FDR (Benjamini–Hochberg)": "fdr_bh",
                                "Bonferroni": "bonferroni",
                                "None": "none"
                            }

                            volcano_data = calculate_volcano_data(
                                data_vol, class_column, selected_features, p_value_threshold,
                                correction_method=method_map[correction_method],
                                control_class=st.session_state.get("volcano_control_class")
                            )

                            # Apply fold change filter
                            filtered_volcano_data = volcano_data[
                                (volcano_data['Log2 Fold Change'] >= fold_change_threshold) |
                                (volcano_data['Log2 Fold Change'] <= -fold_change_threshold)
                            ]

                            # Plot
                            volcano_plot = plot_volcano(
                                filtered_volcano_data,
                                highlight_features,
                                p_value_threshold,
                                fold_change_threshold,capture_name="volcano_fig",
                            )

                            # Store results first, then rerun so the persistent block renders
                            st.session_state["volcano_data"] = filtered_volcano_data
                            st.session_state["show_volcano"] = True
                            st.session_state["volcano_fig"] = volcano_plot
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error generating Volcano Plot: {e}")
                    finally:
                        del data_vol
                        gc.collect()

        # ── Re-display stored volcano figure (persists across ANY rerun) ────────
        if st.session_state.get("show_volcano"):
            _stored_vol_fig = st.session_state.get("volcano_fig")
            if _stored_vol_fig is not None:
                st.plotly_chart(_stored_vol_fig, use_container_width=True, config={
                    'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True,
                    'modeBarButtonsToAdd': ['downloadImage'],
                    'toImageButtonOptions': {'format': 'png', 'scale': 2}
                })

        # ---------------- RESULTS ----------------
        if "volcano_data" in st.session_state:
            try:
                volcano_data = st.session_state["volcano_data"]
                data_vol = st.session_state.get("volcano_data_source_df")  
                comparisons = volcano_data["Comparison"].unique()
                significant_features = set()

                _volcano_up_by_comparison   = {}
                _volcano_down_by_comparison = {}

                for comparison in comparisons:
                    comparison_data = volcano_data[volcano_data["Comparison"] == comparison]
                    upregulated = comparison_data[comparison_data["Regulation Type"] == "Upregulated"]["Feature"].tolist()
                    downregulated = comparison_data[comparison_data["Regulation Type"] == "Downregulated"]["Feature"].tolist()

                    _volcano_up_by_comparison[comparison]   = upregulated
                    _volcano_down_by_comparison[comparison] = downregulated

                    st.write(f"**Comparison: {comparison}**")

                    if upregulated:
                        st.info("**Upregulated Features:**")
                        st.success(", ".join(upregulated))
                        significant_features.update(upregulated)
                    else:
                        st.warning("No specifically upregulated features found.")

                    if downregulated:
                        st.info("**Downregulated Features:**")
                        st.success(", ".join(downregulated))
                        significant_features.update(downregulated)
                    else:
                        st.warning("No specifically downregulated features found.")

                # Store per-comparison up/down for ORA auto-detection
                st.session_state["volcano_up_by_comparison"]   = _volcano_up_by_comparison
                st.session_state["volcano_down_by_comparison"] = _volcano_down_by_comparison

                # Display dataframe of significant features from the selected data source
                if significant_features and data_vol is not None:
                    significant_features = list(significant_features)
                    class_column = st.session_state.get("class_column", "Class")

                    # ── Résoudre str vs float columns (m/z mzML) ─────────────────
                    import pandas as _pd_vol
                    _data_vol_norm = data_vol.copy()
                    _data_vol_norm.columns = _data_vol_norm.columns.astype(str)
                    significant_features = [str(f) for f in significant_features]
                    significant_features = _resolve_features(_data_vol_norm, significant_features)

                    missing_cols = [f for f in significant_features if f not in _data_vol_norm.columns]
                    if missing_cols:
                        st.warning(f"Missing columns in selected data source: {', '.join(missing_cols)}")
                        significant_features = [f for f in significant_features if f not in missing_cols]

                    if class_column in _data_vol_norm.columns:
                        significant_data = _data_vol_norm[[class_column] + significant_features]
                
                        st.write("**DataFrame with Significant Features from Selected Source:**")
                        st.info(
                            "💡 **Feature Extraction Tip:** This table of significant features can be used as a "
                            "feature extraction step. You can download it and re-import it into Profiler or any "
                            "other pipeline to perform machine learning classification (Random Forest, SVM, etc.), "
                            "dimensionality reduction and visualization (PCA, t-SNE, UMAP), or unsupervised "
                            "clustering (K-Means, hierarchical clustering, etc.)."
                        )
                
                        # Layout avec bouton de téléchargement
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.dataframe(significant_data)
                        with col2:
                            # Récupérer les paramètres pour le nom de fichier
                            p_thresh = st.session_state.get("volcano_p_threshold", 0.05)
                            fc_thresh = st.session_state.get("volcano_fc_threshold", 0.0)
                    
                            import csv

                            # Préparer le CSV avec le même format que ton export "cleaned dataset"
                            csv_volcano = significant_data.to_csv(
                                index=False,
                                sep=';',  # point important : séparateur ';'
                                quoting=csv.QUOTE_NONNUMERIC,  # pour mettre les chaînes entre guillemets
                                encoding='utf-8-sig'
                            )

                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_volcano.encode("utf-8-sig"),  # encoder pour Windows/Excel
                                file_name=f"volcano_significant_features_p{p_thresh}_fc{fc_thresh}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Download the significant features table as CSV file"
                            )

                    else:
                        st.error("Class column missing from the selected data source.")
                elif not significant_features:
                    st.warning("No significant features found.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


    # Expander Heatmap
    with _t5_heatmap:
        st.markdown(
            '<p style="color: gray; font-size: 14px;">'
            'Feature and sample clustering-heatmap can be performed on all or selected features, '
            'with statistical significance tested on original or log2 intensities.'
            '</p>',
            unsafe_allow_html=True
        )

        # 🔥 Formulaire pour configurer le heatmap
        with st.form("heatmap_form"):
            select_all_features = st.checkbox(
                "Select All Features",
                key="select_all_features",
                help="Check this box to select all features automatically."
            )

            # Data source comes from the global selector above
            _hm_src_map = {"Raw Data": "Raw data", "Preprocessed": "Preprocessed",
                           "Oversampled": "Oversampled", "Undersampled": "Undersampled"}
            data_source = _hm_src_map.get(_t5_src, "None")

            # Couleurs
            default_colors = ["#00FF00", "#000000", "#FF0000"]  # lime, black, red
            color_labels = ["Underexpression", "Neutral", "Overexpression"]
            custom_colors = [
                st.color_picker(f"🎨 Color of {label}", default_colors[i], help=f"Select a custom color for {label}.")
                for i, label in enumerate(color_labels)
            ]

            average_by_class = st.checkbox(
                "Average by Class",
                key="average_by_class",
                help="Check this box to average the feature values by class."
            )

            perform_stat_test = st.checkbox(
                "Perform Statistical Test",
                key="perform_stat_test",
                help="Check this box to perform a statistical test on the selected features."
            )

            if perform_stat_test:
                p_value_threshold = st.number_input(
                    "P-value",
                    min_value=0.0000001,
                    max_value=1.0,
                    value=0.01,
                    step=0.0001,
                    key="p_value_threshold",
                    help="Set the p-value threshold for the statistical test."
                )
                data_type = st.selectbox(
                    "Select Data Type for Statistical Test",
                    ['Original Intensity', 'Log2'],
                    key="data_type",
                    help="Choose the data type for the statistical test (original or transformed log2)."
                )
            else:
                p_value_threshold = 1
                data_type = 'Original Intensity'

            # Sélection manuelle des features
            features_input = ""
            if not select_all_features:
                features_input = st.text_input(
                    "Features (comma-separated)",
                    key="features_input",
                    help="Enter a comma-separated list of features to include in the Heatmap."
                )

            show_sample_names = st.checkbox(
                "Show sample names on heatmap",
                value=True,
                help="Display sample names on the heatmap if dataset is small."
            )

            # ── Meta annotation bars ──────────────────────────────────
            _hm_df_preview = {
                'Raw data': st.session_state.get('data'),
                'Preprocessed': st.session_state.get('preprocessed_data'),
                'Oversampled': st.session_state.get('oversampled_data'),
                'Undersampled': st.session_state.get('undersampled_data'),
            }.get(data_source)
            _meta_avail_hm = (
                [c for c in _hm_df_preview.columns if str(c).endswith('_meta')]
                if _hm_df_preview is not None else []
            )
            if _meta_avail_hm:
                st.markdown("**🏷️ Metadata annotation bars** *(optional — displayed above the Class bar)*")
                meta_annotation_cols = st.multiselect(
                    "Add metadata annotation bars",
                    options=_meta_avail_hm,
                    default=[],
                    key="heatmap_meta_annotations",
                    help=(
                        "Each selected column adds a colour bar above the heatmap. "
                        "Categorical columns get distinct colours; "
                        "numeric columns get a gradient. "
                        "A legend is shown to the right."
                    )
                )
            else:
                meta_annotation_cols = []

            # Bouton de validation
            submitted = st.form_submit_button("Show Heatmap")
            _hm_clear = st.form_submit_button("🗑️ Clear Heatmap", help="Remove the current heatmap so you can configure a new one.")

        # Clear heatmap if requested
        if _hm_clear:
            for _k in ("show_heatmap", "heatmap_fig", "heatmap_overexpressed_features",
                       "heatmap_exclusive_features", "heatmap_significant_features",
                       "heatmap_data_source_df", "heatmap_over_df", "heatmap_n_features",
                       "heatmap_p_threshold", "heatmap_avg_by_class", "heatmap_class_labels"):
                st.session_state.pop(_k, None)
            st.rerun()



        if submitted:
            try:
                st.session_state["show_heatmap"] = True
                # Charger la source des données
                if data_source == 'Raw data':
                    data_heat = st.session_state.get('data')
                    data_source_df = st.session_state.get('data') 
                elif data_source == 'Preprocessed':
                    data_heat = st.session_state.get('preprocessed_data')
                    data_source_df = st.session_state.get('preprocessed_data') 
                elif data_source == 'Oversampled':
                    data_heat = st.session_state.get('oversampled_data')
                    data_source_df = st.session_state.get('oversampled_data')  
                elif data_source == 'Undersampled':
                    data_heat = st.session_state.get('undersampled_data')
                    data_source_df = st.session_state.get('undersampled_data') 
                else:
                    data_heat = None
                    data_source_df = None
                if data_heat is None:
                    st.warning("Please select a valid data source.")
                    st.stop()
                # Déterminer les features sélectionnées
                if select_all_features:
                    selected_features = [col for col in data_heat.columns if col not in ['Class', 'File', 'RT', 'Sum']]
                else:
                    selected_features = [f.strip() for f in features_input.split(',')] if features_input else []
                    if not selected_features:
                        st.warning("Please select features for the heatmap.")
                        st.stop()
                # Vérifier que les features existent vraiment
                selected_features = [f for f in selected_features if f in data_heat.columns]
                if not selected_features:
                    st.error("No valid features found in the dataset. Please check your selection.")
                    st.stop()
                # Forcer la conversion numérique des colonnes features (évite str/int TypeError)
                data_heat = data_heat.copy()
                for _f in selected_features:
                    data_heat[_f] = pd.to_numeric(data_heat[_f], errors='coerce')
                # Appliquer log2 sur tout le DataFrame si demandé (évite NaN/-inf)
                if perform_stat_test and data_type == 'Log2':
                    data_heat[selected_features] = np.log2(data_heat[selected_features].clip(lower=1e-6))
                # Test statistique si demandé
                if select_all_features and perform_stat_test:
                    significant_features = []
                    classes = data_heat['Class'].unique()
                    if len(classes) < 2:
                        st.error("The 'Class' column must contain at least two unique classes.")
                        st.stop()
                    for feature in selected_features:
                        groups = [data_heat[data_heat['Class'] == c][feature] for c in classes]
                        # Vérifier qu'il y a assez de données
                        if all(len(g) > 1 for g in groups):
                            if len(classes) > 2:
                                _, p_value = f_oneway(*groups)
                            else:
                                _, p_value = ttest_ind(groups[0], groups[1])
                            if p_value < p_value_threshold:
                                significant_features.append(feature)
                        else:
                            st.warning(f"Not enough data for feature '{feature}' in one of the classes.")
                    if not significant_features:
                        st.warning("No significant features found. Please adjust the p-value threshold.")
                        st.stop()
                    selected_features = significant_features

                if average_by_class:
                    data_heat = data_heat.groupby("Class").mean().reset_index()
                st.markdown(f"**{len(selected_features)} feature(s)** selected for clustering.")
                if perform_stat_test:
                    st.markdown(f"P-value threshold: `{p_value_threshold}` on `{data_type}` data")
                # Affichage heatmap
                with st.spinner("Generating heatmap..."):
                    plot_heatmap_samples(
                        data_heat,
                        st.session_state['class_colors'],
                        selected_features,
                        custom_colors,
                        show_sample_names=show_sample_names,
                        capture_name="heatmap_fig",
                        meta_annotation_cols=meta_annotation_cols,
                    )
                st.markdown("**Overexpressed Features by Class**")
                if average_by_class:
                    df_for_overexpr = data_heat.set_index("Class")
                else:
                    df_for_overexpr = data_heat.copy()
                classes = data_heat['Class'].unique()
                overexpressed_features = {}
                for cls in classes:
                    cls_mask = df_for_overexpr.index == cls if average_by_class else data_heat['Class'] == cls
                    cls_values = df_for_overexpr.loc[cls_mask, selected_features].mean() if average_by_class else df_for_overexpr.loc[cls_mask, selected_features].mean()
                    other_values = df_for_overexpr.loc[~cls_mask, selected_features].mean() if average_by_class else df_for_overexpr.loc[~cls_mask, selected_features].mean()

                    diff = cls_values - other_values
                    # Features plus hautes que les autres classes
                    overexpressed_features[cls] = list(diff[diff > 0].sort_values(ascending=False).index)

                # Store overexpressed features for ORA auto-detection
                st.session_state["heatmap_overexpressed_features"] = {
                    cls: feats for cls, feats in overexpressed_features.items() if feats
                }

                # Compute heatmap-exclusive features: non-zero/non-null ONLY in one class
                _hm_feats_per_class = {}
                for _hcls in classes:
                    _mask = data_heat['Class'] == _hcls
                    _hm_feats_per_class[_hcls] = set(
                        f for f in selected_features
                        if data_heat.loc[_mask, f].notnull().any() and
                           (data_heat.loc[_mask, f] != 0).any()
                    )
                _hm_exclusive = {
                    cls: sorted(feats - set.union(*(_hm_feats_per_class[c] for c in classes if c != cls)))
                    for cls, feats in _hm_feats_per_class.items()
                }
                st.session_state["heatmap_exclusive_features"] = {
                    cls: feats for cls, feats in _hm_exclusive.items() if feats
                }

                # ── Persist results for re-display across reruns ───────────────
                _over_df_rows = []
                _feat_cols_for_over = selected_features
                _df_for_over = data_source_df[['Class'] + [f for f in _feat_cols_for_over if f in data_source_df.columns]].copy()
                for _f in _feat_cols_for_over:
                    if _f in _df_for_over.columns:
                        _df_for_over[_f] = pd.to_numeric(_df_for_over[_f], errors='coerce')
                _classes_over = _df_for_over['Class'].unique()
                for _cls in _classes_over:
                    _cls_vals = _df_for_over[_df_for_over['Class'] == _cls][[f for f in _feat_cols_for_over if f in _df_for_over.columns]].mean()
                    _other_vals = _df_for_over[_df_for_over['Class'] != _cls][[f for f in _feat_cols_for_over if f in _df_for_over.columns]].mean()
                    _diff = _cls_vals - _other_vals
                    for _feat, _score in _diff[_diff > 0].sort_values(ascending=False).items():
                        _over_df_rows.append({"Class": _cls, "Feature": _feat, "Overexpression Score": round(_score, 4)})
                st.session_state["heatmap_over_df"] = pd.DataFrame(_over_df_rows) if _over_df_rows else pd.DataFrame()
                st.session_state["heatmap_n_features"]   = len(selected_features)
                st.session_state["heatmap_p_threshold"]  = p_value_threshold if perform_stat_test else None
                st.session_state["heatmap_avg_by_class"] = average_by_class
                st.session_state["heatmap_class_labels"] = list(classes)
                if perform_stat_test and 'significant_features' in locals():
                    st.session_state["heatmap_significant_features"] = significant_features
                    st.session_state["heatmap_data_source_df"]       = data_source_df

                for var in ["data_heat", "data_source_df", "selected_features",
                            "custom_colors", "_df_for_over", "_over_df_rows"]:
                    try: del locals()[var]
                    except: pass
                gc.collect()
                # Hand off to the persistent display block below (avoids
                # double-render and ensures all reruns show the heatmap).
                st.rerun()

            except Exception as e:
                import traceback
                st.error(f"An error occurred while generating the Heatmap: {e}")
                st.text(traceback.format_exc())

        # ── Re-display stored heatmap results (persists across ANY rerun) ────
        # The figure is stored in session_state["heatmap_fig"] by
        # plot_heatmap_samples() itself. We always re-render it here so
        # download_button clicks (or any other widget interaction) never
        # cause it to disappear.
        if st.session_state.get("show_heatmap"):
            _stored_fig = st.session_state.get("heatmap_fig")
            if _stored_fig is not None:
                # Re-display the interactive plotly chart (persists across all reruns)
                st.plotly_chart(_stored_fig, use_container_width=True, config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["downloadImage"],
                    "toImageButtonOptions": {"format": "png", "scale": 3},
                })
                # PNG download button — uses matplotlib bytes pre-generated by
                # _build_static_png_original (stored in session_state), no kaleido needed.
                _hm_png_bytes = st.session_state.get("heatmap_fig_png_bytes")
                if _hm_png_bytes:
                    st.download_button(
                        label="📥 Download Heatmap (PNG)",
                        data=_hm_png_bytes,
                        file_name="heatmap_clustering.png",
                        mime="image/png",
                        use_container_width=True,
                        key="heatmap_png_download",
                        help="Download the heatmap as a high-resolution PNG image (seaborn clustermap, 200 dpi).",
                    )
                else:
                    st.info("Heatmap PNG not available. Please regenerate the heatmap.")

            # ── Overexpressed summary ──────────────────────────────────────────
            _over_feats = st.session_state.get("heatmap_overexpressed_features", {})
            _exc_feats  = st.session_state.get("heatmap_exclusive_features", {})
            _cls_color_map = st.session_state.get('class_colors', {})
            _n_feat = st.session_state.get("heatmap_n_features", "?")
            _p_thr  = st.session_state.get("heatmap_p_threshold")

            st.markdown(f"**{_n_feat} feature(s)** used in the heatmap.")
            if _p_thr is not None:
                st.markdown(f"P-value threshold: `{_p_thr}`")

            st.markdown("**Overexpressed Features by Class**")
            for cls, feats in _over_feats.items():
                exc = _exc_feats.get(cls, [])
                _col = _cls_color_map.get(cls, "#318CE7")
                if feats:
                    st.markdown(
                        f"<div style='background:rgba(49,140,231,0.07);border-left:4px solid {_col};"
                        f"border-radius:6px;padding:8px 14px;margin:4px 0;font-size:0.88rem;'>"
                        f"<b style='color:{_col};'>{cls}</b> — "
                        f"<b>{len(feats)}</b> overexpressed features&nbsp;·&nbsp;"
                        f"<b>{len(exc)}</b> exclusive features</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:#f8fafc;border-left:4px solid #94a3b8;"
                        f"border-radius:6px;padding:8px 14px;margin:4px 0;font-size:0.88rem;color:#64748b;'>"
                        f"<b>{cls}</b> — No overexpressed features detected</div>",
                        unsafe_allow_html=True
                    )

            # ── Overexpressed detail table ─────────────────────────────────────
            _over_df = st.session_state.get("heatmap_over_df", pd.DataFrame())
            if not _over_df.empty:
                st.markdown("**Over-expressed Features by Class**")
                import csv as _csv_mod
                _col1, _col2 = st.columns([3, 1])
                with _col1:
                    st.dataframe(_over_df, use_container_width=True)
                with _col2:
                    _csv_over = _over_df.to_csv(index=False).encode('utf-8')
                    _p_label = st.session_state.get("heatmap_p_threshold", "all")
                    st.download_button(
                        label="📥 Download CSV",
                        data=_csv_over,
                        file_name=f"overexpressed_features_p{_p_label}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Download the overexpressed features table as CSV file"
                    )
            else:
                st.info("No overexpressed features detected for selected classes.")

            # ── Significant features table (stat test) ─────────────────────────
            _sig_feats   = st.session_state.get("heatmap_significant_features")
            _sig_src_df  = st.session_state.get("heatmap_data_source_df")
            _p_thr_sig   = st.session_state.get("heatmap_p_threshold")
            if _sig_feats and _sig_src_df is not None:
                _valid_sig = [f for f in _sig_feats if f in _sig_src_df.columns]
                if _valid_sig:
                    significant_df = _sig_src_df[['Class'] + _valid_sig].copy()
                    st.markdown("**Significant Features**")
                    st.info(
                        "💡 **Feature Extraction Tip:** This table of significant features can be used as a "
                        "feature extraction step. You can download it and re-import it into Profiler or any "
                        "other pipeline to perform machine learning classification (Random Forest, SVM, etc.), "
                        "dimensionality reduction and visualization (PCA, t-SNE, UMAP), or unsupervised "
                        "clustering (K-Means, hierarchical clustering, etc.)."
                    )
                    import csv
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.dataframe(significant_df)
                    with col2:
                        csv_significant = significant_df.to_csv(
                            index=False, sep=';',
                            quoting=csv.QUOTE_NONNUMERIC,
                            encoding='utf-8-sig'
                        )
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_significant.encode("utf-8-sig"),
                            file_name=f"significant_features_p{_p_thr_sig}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    st.markdown("**Common Overexpressed Features Between Classes**")
                    _valid_over = {cls: feats for cls, feats in _over_feats.items() if feats}
                    if len(_valid_over) < 2:
                        st.info("Not enough classes with overexpressed features to compute intersections.")
                    else:
                        _common_all = set.intersection(*map(set, _valid_over.values()))
                        if _common_all:
                            st.success(f"**Common to ALL Classes ({len(_common_all)}):** {', '.join(sorted(_common_all))}")
                        else:
                            st.warning("No features are commonly overexpressed across ALL classes.")
                        st.markdown("**Pairwise Intersections**")
                        for i, cls1 in enumerate(_valid_over):
                            for cls2 in list(_valid_over)[i + 1:]:
                                inter = set(_valid_over[cls1]).intersection(_valid_over[cls2])
                                if inter:
                                    st.info(f"**{cls1} ∩ {cls2}** ({len(inter)} features): " + ", ".join(sorted(inter)))
                                else:
                                    st.info(f"**{cls1} ∩ {cls2}**: None")
                
        gc.collect()





    if "show_boxplots" not in st.session_state:
        st.session_state["show_boxplots"] = False

    with _t5_boxplot:
        st.markdown(
            '<p style="color: gray; font-size: 14px;">Visualize and statistically compare the distribution '
            'of selected features across sample classes. Includes p-value correction and filtering.</p>',
            unsafe_allow_html=True
        )

        # ── Résoudre le dataset AVANT la form (pour la liste de features) ──
        _box_src_map2 = {"Raw Data": "raw_bp", "Preprocessed": "preprocessed_data",
                         "Oversampled": "oversampled_data", "Undersampled": "undersampled_data"}
        _bp_ds_key = {"Raw Data": "data", "Preprocessed": "preprocessed_data",
                      "Oversampled": "oversampled_data", "Undersampled": "undersampled_data"
                      }.get(_t5_src, "data")
        _bp_df_preview = st.session_state.get(_bp_ds_key)

        # Liste des features disponibles (hors colonnes méta/structurelles)
        _BP_EXCL = {'Class', 'File', 'RT', 'Sum', 'ID', 'id', 'Index'}
        _bp_all_features = (
            [c for c in _bp_df_preview.columns
             if c not in _BP_EXCL and not str(c).endswith('_meta')]
            if _bp_df_preview is not None else []
        )

        # ── Feature selection OUTSIDE form (live filtering) ───────────────────
        st.markdown("**Feature Selection**")

        # ── Paste features ────────────────────────────────────────────────────
        with st.expander("📋 Paste a feature list", expanded=False):
            _bp_paste_raw = st.text_area(
                "Paste feature names (separated by spaces, commas, or newlines)",
                key="bp_paste_features",
                placeholder="e.g.  feature_123, feature_456  or  feature_123 feature_456",
                height=90,
            )
            # Parse: split on commas, spaces, newlines — strip whitespace — deduplicate
            import re as _re
            _bp_paste_tokens = [
                t.strip() for t in _re.split(r'[\s,]+', _bp_paste_raw.strip()) if t.strip()
            ] if _bp_paste_raw.strip() else []

            # Validate against available features (exact match first, then case-insensitive)
            _bp_all_lower_map = {str(f).lower(): f for f in _bp_all_features}
            _bp_pasted_valid, _bp_pasted_invalid = [], []
            for _tok in _bp_paste_tokens:
                if _tok in _bp_all_features:
                    _bp_pasted_valid.append(_tok)
                elif _tok.lower() in _bp_all_lower_map:
                    _bp_pasted_valid.append(_bp_all_lower_map[_tok.lower()])
                else:
                    _bp_pasted_invalid.append(_tok)

            # Deduplicate while preserving order
            _seen = set()
            _bp_pasted_valid = [
                f for f in _bp_pasted_valid
                if not (_seen.add(f) or f in _seen - {f})
            ]

            if _bp_paste_tokens:
                if _bp_pasted_valid:
                    st.success(f"✅ {len(_bp_pasted_valid)} feature(s) recognised and ready to add to selection.")
                if _bp_pasted_invalid:
                    st.warning(f"⚠️ {len(_bp_pasted_invalid)} not found in dataset: {', '.join(_bp_pasted_invalid[:10])}"
                                + (" …" if len(_bp_pasted_invalid) > 10 else ""))

        _c_srch, _c_topn = st.columns([3, 1])
        with _c_srch:
            _bp_search = st.text_input(
                "Filter features",
                key="bp_feature_search",
                placeholder="Type substring to narrow the list…",
            )
        with _c_topn:
            _bp_topn = st.number_input(
                "Quick top-N", min_value=0, max_value=200,
                value=0, step=5, key="bp_top_n",
                help="Auto-select first N filtered features."
            )

        _bp_filtered = (
            [f for f in _bp_all_features
            if _bp_search.strip().lower() in str(f).lower()]
            if _bp_search.strip() else _bp_all_features
        )

        # Merge pasted valid features into default selection (union with top-N)
        _bp_default_topn = _bp_filtered[:int(_bp_topn)] if _bp_topn > 0 else []
        _bp_default = list(dict.fromkeys(_bp_pasted_valid + _bp_default_topn))  # pasted first, dedup

        # Ensure options list contains all pasted valid features (even if filtered out)
        _bp_options = list(dict.fromkeys(_bp_pasted_valid + _bp_filtered))

        mz_values_selected = st.multiselect(
            f"Features ({len(_bp_filtered):,} available"
            + (f", filtered from {len(_bp_all_features):,}" if _bp_search.strip() else "")
            + (f" · {len(_bp_pasted_valid)} pasted" if _bp_pasted_valid else "")
            + ")",
            options=_bp_options,
            default=_bp_default,
            key="bp_features_multiselect",
            help="Select one or more features to compare across classes. Use the paste box above to add features quickly.",
        )

        # ── Analysis params INSIDE form ────────────────────────────────────────
        with st.form("boxplot_stat_form", clear_on_submit=False):
            data_source = {"Raw Data": "Raw data", "Preprocessed": "Preprocessed",
                        "Oversampled": "Oversampled", "Undersampled": "Undersampled"
                        }.get(_t5_src, "Raw data")

            st.markdown("---")
            st.markdown("##### Analysis Options")
            _co1, _co2, _co3 = st.columns(3)
            with _co1:
                test = st.selectbox(
                    "Statistical Test",
                    ['Kruskal', 'Mann-Whitney', 't-test_ind', 'ANOVA'],
                    index=0, key="statistical_test",
                    help="Kruskal/ANOVA: 2+ groups. Mann-Whitney/t-test: exactly 2 groups."
                )
            with _co2:
                pval_correction = st.selectbox(
                    "P-value Correction",
                    ['None', 'Bonferroni', 'FDR (Benjamini-Hochberg)'],
                    key="pval_correction_method",
                )
            with _co3:
                plot_type = st.selectbox(
                    "Plot Type",
                    ['Box Plot', 'Violin Plot', 'Bar Plot'],
                    key="plot_type",
                )

            _co4, _co5 = st.columns(2)
            with _co4:
                show_scatter = st.checkbox("Show Individual Points", key="show_scatter")
            with _co5:
                use_log2 = st.checkbox("Apply log2 Transform", key="use_log2")

            # ── Class selection ────────────────────────────────────────────────
            data_sig_key = {
                'Raw data': 'data', 'Preprocessed': 'preprocessed_data',
                'Oversampled': 'oversampled_data', 'Undersampled': 'undersampled_data'
            }.get(data_source)
            data_sig = st.session_state.get(data_sig_key)
            can_submit = True
            selected_classes = []

            if data_sig is not None:
                class_col = st.session_state.get("label_column", "Class")
                all_classes = data_sig[class_col].dropna().unique().tolist()
                compare_all = st.checkbox(
                    "Compare all classes", value=True, key="compare_all_classes"
                )
                if compare_all:
                    selected_classes = all_classes
                else:
                    selected_classes = st.multiselect(
                        "Select Classes to Compare",
                        options=all_classes,
                        default=all_classes[:2] if len(all_classes) >= 2 else all_classes,
                        key="selected_classes",
                        help="Mann-Whitney / t-test require exactly 2 classes."
                    )
                    if test in ['Mann-Whitney', 't-test_ind'] and len(selected_classes) != 2:
                        st.warning("⚠️ This test requires exactly 2 classes.")
                        can_submit = False
                    elif len(selected_classes) < 2:
                        st.warning("⚠️ Select at least 2 classes.")
                        can_submit = False
            else:
                st.error("❌ Dataset not loaded.")
                can_submit = False

            # NOTE: we do NOT disable submit based on mz_values_selected here
            # because mz_values_selected is now outside the form and its value
            # IS available at submit time via st.session_state["bp_features_multiselect"]
            if can_submit and len(mz_values_selected) == 0:
                st.caption("⬆️ Select at least one feature above to enable Run.")

            submitted = st.form_submit_button(
                "▶ Run Analysis",
                disabled=not can_submit,
                use_container_width=True,
                type="primary",
            )

        # ── Post-submission ────────────────────────────────────────────────────
        if submitted:
            # Re-read mz_values_selected from session state (outside-form value is preserved)
            mz_values_selected = st.session_state.get("bp_features_multiselect", mz_values_selected)
            if not mz_values_selected:
                st.warning("⚠️ Please select at least one feature.")
            else:


                data_sig = data_sig.copy()
                class_col = st.session_state.get("label_column", "Class")
                data_filtered = data_sig[data_sig[class_col].isin(selected_classes)].copy()
                # Multiselect already validated — just confirm they exist in filtered df
                mz_values = _resolve_features_df(data_filtered, mz_values_selected)

                if not mz_values:
                    st.warning("⚠️ None of the selected features found in this data source.")
                else:
                    # Calcul des statistiques
                    # Pre-extract all groups as dict for speed
                    _grp_cache = {g: data_filtered[data_filtered[class_col] == g] for g in selected_classes}

                    def _run_test(mz):
                        groups = [pd.to_numeric(_grp_cache[g][mz], errors="coerce").dropna() for g in selected_classes]
                        try:
                            if test == 'Kruskal':
                                return {"feature": mz, "pvalue": stats.kruskal(*groups).pvalue}
                            elif test == 'Mann-Whitney' and len(groups) == 2:
                                return {"feature": mz, "pvalue": stats.mannwhitneyu(groups[0], groups[1]).pvalue}
                            elif test == 't-test_ind' and len(groups) == 2:
                                return {"feature": mz, "pvalue": stats.ttest_ind(groups[0], groups[1]).pvalue}
                            elif test == 'ANOVA':
                                return {"feature": mz, "pvalue": stats.f_oneway(*groups).pvalue}
                        except Exception:
                            pass
                        return {"feature": mz, "pvalue": None}

                    _n_jobs = min(10, max(1, len(mz_values)))
                    results = joblib.Parallel(n_jobs=_n_jobs, prefer="threads")(
                        joblib.delayed(_run_test)(mz) for mz in mz_values
                    )

                    # Correction des p-values
                    result_df = pd.DataFrame(results).dropna()
            
                    if len(result_df) == 0:
                        st.error("❌ No valid p-values computed. Check your data.")
                    else:
                        if pval_correction == 'Bonferroni':
                            result_df["adj_pvalue"] = multipletests(result_df["pvalue"], method="bonferroni")[1]
                        elif pval_correction == 'FDR (Benjamini-Hochberg)':
                            result_df["adj_pvalue"] = multipletests(result_df["pvalue"], method="fdr_bh")[1]
                        else:
                            result_df["adj_pvalue"] = result_df["pvalue"]

                        # Affichage des résultats
                        sig_df = result_df[result_df["adj_pvalue"] < 0.05]
                        nonsig_df = result_df[result_df["adj_pvalue"] >= 0.05]
                        st.info(f"**Significant features (p < 0.05)**: {len(sig_df)} / {len(result_df)}")
                        st.info(f"**Non-significant features**: {len(nonsig_df)} / {len(result_df)}")

                        # Stocker les arguments du plot dans session_state
                        # pour pouvoir le ré-afficher sans re-soumettre le formulaire
                        st.session_state["_boxplot_args"] = dict(
                            data=data_filtered,
                            mz_values=mz_values,
                            class_colors=st.session_state.get("class_colors", None),
                            test=test,
                            plot_type=plot_type.lower().split()[0],
                            show_scatter=show_scatter,
                            use_log2=use_log2,
                            pval_correction=pval_correction,
                            significance_dict=dict(zip(result_df.feature, result_df.adj_pvalue)),
                            capture_name="boxplots_fig"
                        )

        # ── Toujours ré-afficher le plot si des args sont stockés ─────────────────
        # (permet aux widgets dans _make_feature_subplots de fonctionner sans re-submit)
        if not submitted and st.session_state.get("_boxplot_args"):
            _args = st.session_state["_boxplot_args"]
            plot_significant_features(**_args)
        elif submitted and st.session_state.get("_boxplot_args"):
            _args = st.session_state["_boxplot_args"]
            plot_significant_features(**_args)





    if "show_shap" not in st.session_state:
        st.session_state["show_shap"] = False

    with _t5_shap:
        st.markdown('<p style="color: gray; font-size: 14px;">Visualize how each feature contributes to model predictions using SHAP.</p>', unsafe_allow_html=True)

        model_type_for_interpretation = st.selectbox(
            "Select Model Type for Interpretation",
            ['None', 'Machine Learning', 'Deep Learning'],
            key="model_type_for_interpretation_shap",
            help="Choose the type of model you want to interpret using SHAP."
        )

        # Data source comes from the global selector above — use _t5_df directly
        data_sh = _t5_df
        _shap_unused = {
            'Preprocessed': st.session_state.get('preprocessed_data'),
            'Oversampled': st.session_state.get('oversampled_data'),
            'Undersampled': st.session_state.get('undersampled_data'),
            'Raw data': st.session_state.get('data'),
        }

        top_n_shap = st.number_input("Top N features (SHAP)", min_value=5, max_value=100, value=20, step=5, key="shap_top_n", help="Number of top features in SHAP plots")
        if st.button("Show SHAP Values Importance"):
            if data_sh is None:
                st.warning("Selected data source is not available.")
                st.stop()

            st.session_state["show_shap"] = True  # Keep expander open
            st.write(f"Selected Model Type: {model_type_for_interpretation}")

            # Load appropriate model
            if model_type_for_interpretation == 'Machine Learning':
                model_name = st.session_state.get('selected_model')
                if model_name == 'None' or model_name not in st.session_state.get('models', {}):
                    st.warning("Please select a valid machine learning model.")
                    st.stop()
                model = st.session_state['models'][model_name]['model']

            elif model_type_for_interpretation == 'Deep Learning':
                model_name = st.session_state.get('selected_dl_model')
                if model_name == 'None' or model_name not in st.session_state.get('dl_models', {}):
                    st.warning("Please select a valid deep learning model.")
                    st.stop()
                model = st.session_state['dl_models'][model_name]['model']

            else:
                st.warning("Please select a model type.")
                st.stop()


            # Prepare features and labels
            y = data_sh["Class"]
            X = data_sh.drop([c for c in data_sh.columns if c in _NON_FEATURE_COLS or str(c).endswith('_meta')], axis=1, errors='ignore').select_dtypes(include='number')

            # Define a prefix for capturing figures
            capture_prefix = f"shap_{model_name.lower().replace(' ', '_')}"

            # Clear cached SHAP values if data/model changed
            if "cached_X" in st.session_state and not X.equals(st.session_state["cached_X"]):
                st.session_state.pop("cached_shap_values", None)
                st.session_state.pop("cached_X", None)
                st.session_state.pop("cached_feature_names", None)

            # Run the full SHAP plotting function with capture_prefix
            with st.spinner("Computing SHAP values..."):
                plot_shap_values(
                    model,
                    X,
                    st.session_state.get('class_colors'),
                    sorted(y.unique()),
                    capture_prefix="shap",
                    top_n=st.session_state.get('shap_top_n', 20)
                )
            for _sk in ['shap_beeswarm', 'shap_bar']:
                _sv = st.session_state.get(_sk)
                if _sv is not None:
                    st.session_state[f'_report_{_sk}'] = _sv
            st.session_state['shap_values'] = True

            # Free memory
            del data_sh, X, y
            gc.collect()


    # Expander LIME
    with _t5_lime:
        st.markdown('<p style="color: gray; font-size: 14px;">🔍 Model-based interpretation using LIME. For binary classification, note that class orientation and top features may vary per sample.</p>', unsafe_allow_html=True)

        model_type_for_interpretation = st.selectbox(
            "Select Model Type for LIME Interpretation",
            ['None', 'Machine Learning', 'Deep Learning'],
            key="model_type_for_interpretation_lime"
        )

        # Data source comes from the global selector above — use _t5_df directly
        data_li = _t5_df
        if False:  # placeholder kept for structure
            pass
        elif data_source == 'Raw data':
            data_li = st.session_state.get('data')

        if data_li is None:
            st.warning("Selected data source is not available.")
        else:
            top_n_lime = st.number_input(
                "Top N features (LIME)", min_value=5, max_value=100, value=20, step=5,
                key="lime_top_n",
                help="Number of top features ranked by absolute LIME weight to display."
            )
            if st.button("Show LIME Feature Importance"):
                st.session_state["show_lime"] = True

                if model_type_for_interpretation == 'None':
                    st.warning("Please select a model type.")
                    st.stop()

                model = None
                if model_type_for_interpretation == 'Machine Learning':
                    model_data = st.session_state['models'].get(st.session_state.get('selected_model'))
                    if model_data:
                        model = model_data['model']
                        label_encoder = model_data['label_encoder']
                    else:
                        st.warning("Selected machine learning model is not available.")
                        st.stop()
                elif model_type_for_interpretation == 'Deep Learning':
                    model_data = st.session_state['dl_models'].get(st.session_state.get('selected_dl_model'))
                    if model_data:
                        model = model_data['model']
                        label_encoder = model_data['label_encoder']
                    else:
                        st.warning("Selected deep learning model is not available.")
                        st.stop()

                with st.spinner("Computing LIME Explanations..."):
                    try:
                        html_contribution, lime_df = eli5_feature_importance(
                            model, label_encoder, data_li, top_features=int(top_n_lime)
                        )
                        st.components.v1.html(html_contribution, height=600, scrolling=True)
                        import plotly.express as px

                        # ── Top-N bar chart (mirrors SHAP bar layout) ──────────────
                        _lime_plot_df = (
                            lime_df
                            .assign(abs_weight=lambda d: d['weight'].abs())
                            .nlargest(int(top_n_lime), 'abs_weight')
                            .sort_values('abs_weight', ascending=True)
                        )

                        _lime_colors = _lime_plot_df['weight'].apply(
                            lambda w: '#ef4444' if w < 0 else '#3b82f6'
                        ).tolist()

                        fig = px.bar(
                            _lime_plot_df,
                            x='weight',
                            y='feature',
                            orientation='h',
                            color='weight',
                            color_continuous_scale='RdYlGn',
                            title=f"Top {int(top_n_lime)} LIME Features — Mean Contribution Weight"
                        )
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=max(420, 28 * int(top_n_lime)),
                            font=dict(size=14, color='black', family='Arial'),
                            title=dict(font=dict(size=18, color='black', family='Arial')),
                            yaxis=dict(
                                title='Features',
                                title_font=dict(size=15, color='black'),
                                tickfont=dict(size=13, color='black'),
                                gridcolor='#eeeeee', linecolor='#333', linewidth=1.5, mirror=True,
                            ),
                            xaxis=dict(
                                title='Contribution (Weight)',
                                title_font=dict(size=15, color='black'),
                                tickfont=dict(size=13, color='black'),
                                gridcolor='#eeeeee', zeroline=True,
                                zerolinecolor='#888', zerolinewidth=1.5,
                                linecolor='#333', linewidth=1.5, mirror=True,
                            ),
                            coloraxis_colorbar=dict(
                                title=dict(text='Weight', font=dict(size=13, color='black')),
                                tickfont=dict(size=12, color='black'),
                            ),
                            legend=dict(font=dict(size=13, color='black'),
                                        bordercolor='#ccc', borderwidth=1),
                        )
                        fig.add_vline(x=0, line_dash='solid', line_color='#888', line_width=1)

                        st.plotly_chart(fig, use_container_width=True, config={
                            'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True,
                            'modeBarButtonsToAdd': ['downloadImage'],
                            'toImageButtonOptions': {'format': 'png', 'scale': 2}
                        })
                        _capture_plotly(fig, "lime_feature_contrib")
                    except Exception as e:
                        st.error(f"LIME analysis failed: {e}")

            # Free memory
            del data_li
            gc.collect()



    with tabs[6]:
        # ═══════════════════════════════════════════════════
        # ENRICHMENT – ORA + GSEA via render_enrichment_tab
        # ═══════════════════════════════════════════════════
        from app.analysis.profiler_genes_enrichment import render_enrichment_tab
        (_t6_enrich,) = st.tabs(["  Enrichment Analysis"])
        with _t6_enrich:
            render_enrichment_tab()

    with tabs[7]:
        # ═══════════════════════════════════════════════════
        # SURVIVAL – sub-tabs
        # ═══════════════════════════════════════════════════
        (_t7_km, _t7_cox, _t7_pred) = st.tabs(["  Kaplan-Meier","  Cox Regression","  Survival Prediction"])
        # Vérifier si les données de survie sont disponibles
        survival_data_available = 'survival_data' in st.session_state
        if survival_data_available:
            survie = st.session_state['survival_data']
        else:
            survie = None



        # Expander pour l'analyse Kaplan-Meier 
        with _t7_km:
            st.markdown(
                '<p style="color: gray; font-size: 14px;">Kaplan-Meier survival analysis to assess time to a specific event (death/relapse...).</p>',
                unsafe_allow_html=True,
            )
            st.caption("ℹ️ Requires: 'Overall survival', 'State', and 'Class' columns.")

            if survie is not None:
                if {'Overall survival', 'State', 'Class'}.issubset(survie.columns):
                    if survie['Overall survival'].dtype == 'O':
                        survie['Overall survival'] = survie['Overall survival'].str.replace(',', '.').astype(float)

                    kmf = KaplanMeierFitter()
                    classes = survie['Class'].unique()
                    from plotly import express as px
                    _km_palette = px.colors.qualitative.Plotly

                    # Couleurs
                    if 'class_colors' in st.session_state and st.session_state['class_colors']:
                        class_colors = st.session_state['class_colors']
                    else:
                        class_colors = {cls: _km_palette[i % len(_km_palette)] for i, cls in enumerate(classes)}

                    if st.button("Run Kaplan-Meier Analysis", help="Generate survival curves and perform log-rank tests."):
                        # ── Build KM step functions manually for Plotly ──────────────
                        p_values = {}
                        fig_km = go.Figure()
                        _px_colors = px.colors.qualitative.Plotly

                        for i, cls in enumerate(classes):
                            group = survie['Class'] == cls
                            kmf.fit(
                                survie['Overall survival'][group],
                                survie['State'][group],
                                label=f'Group {cls}'
                            )
                            # KM timeline → step function
                            _t = kmf.timeline
                            _s = kmf.survival_function_at_times(_t).values.flatten()
                            _color = class_colors.get(cls, _px_colors[i % len(_px_colors)])
                            fig_km.add_trace(go.Scatter(
                                x=_t, y=_s, mode="lines", name=f"Group {cls}",
                                line=dict(color=_color, width=2.5, shape="hv"),
                                hovertemplate="t=%{x:.1f}<br>S(t)=%{y:.3f}<extra>" + cls + "</extra>"
                            ))
                            # Censors
                            _cens_mask = survie.loc[group, 'State'] == 0
                            _cens_times = survie.loc[group & _cens_mask, 'Overall survival']
                            if not _cens_times.empty:
                                _cens_s = kmf.survival_function_at_times(_cens_times).values.flatten()
                                fig_km.add_trace(go.Scatter(
                                    x=_cens_times, y=_cens_s, mode="markers",
                                    name=f"{cls} censored", legendgroup=cls,
                                    showlegend=False,
                                    marker=dict(symbol="line-ew", size=8,
                                                color=_color, line=dict(width=2, color=_color))
                                ))

                        # Log-rank test
                        for i in range(len(classes)):
                            for j in range(i + 1, len(classes)):
                                group_A = survie['Class'] == classes[i]
                                group_B = survie['Class'] == classes[j]
                                _lr = logrank_test(
                                    durations_A=survie['Overall survival'][group_A],
                                    durations_B=survie['Overall survival'][group_B],
                                    event_observed_A=survie['State'][group_A],
                                    event_observed_B=survie['State'][group_B]
                                )
                                p_values[f'{classes[i]} vs {classes[j]}'] = _lr.p_value

                        fig_km.update_layout(
                            title="Kaplan-Meier Survival Curves",
                            xaxis=dict(title="Survival Time", showgrid=True, gridcolor="#f1f5f9"),
                            yaxis=dict(title="Survival Probability", range=[0, 1.05],
                                       showgrid=True, gridcolor="#f1f5f9"),
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=500, legend=dict(font=dict(size=13)),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_km, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_km, "km_fig")

                        # P-values table
                        st.subheader("Log-rank Test P-values")
                        _pv_df = pd.DataFrame(
                            [(k, f"{v:.4f}", "✅" if v >= 0.05 else "⚠️ significant")
                             for k, v in p_values.items()],
                            columns=["Comparison", "p-value", ""]
                        )
                        st.dataframe(_pv_df, use_container_width=True, hide_index=True)
                else:
                    st.error("The survival data must contain 'Overall survival', 'State', and 'Class' columns.")
            else:
                st.warning("Please upload survival data to perform Kaplan-Meier analysis.")





        # Expander for Cox Model Analysis
        # Expander for Cox Model Analysis
        with _t7_cox:
            st.markdown('<p style="color: gray; font-size: 14px;">Cox model to analyze the impact of covariates on survival.</p>', unsafe_allow_html=True)
            st.caption("ℹ️ Requires: 'Overall survival', 'State', and covariates such as age, BMI, markers... (numeric or categorical).")

            if survie is not None:
                if 'Class' not in survie.columns:
                    st.error("The uploaded file must contain the column 'Class' for Cox Model analysis.")
                else:
                    pipeline = create_cox_pipeline(survie)
                    cph = None  # Initialize cph to None

                    if st.button("Run Cox Model Analysis", help="Fit a Cox model with regularization and display results"):
                        # Fit the pipeline to the data
                        pipeline.fit(survie)

                        # Retrieve the transformed columns
                        num_columns = pipeline.named_steps['preprocessor'].transformers_[0][2]  # Unchanged numeric columns

                        # Get the column names after OneHotEncoding
                        ohe = pipeline.named_steps['preprocessor'].transformers_[1][1]  # Instance of the OneHotEncoder
                        cat_columns = ohe.get_feature_names_out(input_features=pipeline.named_steps['preprocessor'].transformers_[1][2])

                        # Create the final list of columns
                        columns_after_transformation = list(num_columns) + list(cat_columns)

                        # Transform the data
                        X_transformed = pipeline.transform(survie)
                        X_transformed_df = pd.DataFrame(X_transformed, columns=columns_after_transformation)

                        X_transformed_df.fillna(X_transformed_df.median(), inplace=True)  # Handle NaN values

                        # Add 'Overall survival' and 'State' columns
                        X_transformed_df[['Overall survival', 'State']] = survie[['Overall survival', 'State']]

                        X_numeric = X_transformed_df.drop(columns=['Overall survival', 'State'])  # Exclude survival columns

                        # Skip collinearity detection if only one covariate is present
                        if X_numeric.shape[1] > 1:
                            collinearity_detected = detect_collinearity(X_numeric)
                        else:
                            collinearity_detected = False  # Cannot have collinearity with one covariate

                        # Apply penalty only if collinearity is detected
                        penalizer_value = 0.1 if collinearity_detected else 0.0
                        # Create and train the Cox model
                        cph = CoxPHFitter(penalizer=penalizer_value)
                        cph.fit(X_transformed_df, duration_col='Overall survival', event_col='State')
                        st.session_state.cph = cph

                        st.markdown("**Cox Model Summary**")
                        st.write(cph.summary)

                        st.write(f"Concordance = {cph.concordance_index_}")

                        # Adjust figure size dynamically based on the number of variables
                        num_vars = len(X_numeric.columns)
                        fig_height = max(6, num_vars * 0.3)  # Adjust height based on the number of variables

                        # ── Plotly forest plot from cph.summary ──────────────
                        _summary = cph.summary.copy()
                        _vars = _summary.index.tolist()
                        _coef = _summary["coef"].values
                        _lo = _summary["coef lower 95%"].values
                        _hi = _summary["coef upper 95%"].values
                        _p   = _summary["p"].values

                        fig_cox = go.Figure()
                        for _vi, (_v, _c, _l, _h, _pv) in enumerate(zip(_vars, _coef, _lo, _hi, _p)):
                            _col = "#dc2626" if _pv < 0.05 else "#2563eb"
                            fig_cox.add_trace(go.Scatter(
                                x=[_l, _c, _h], y=[_v, _v, _v],
                                mode="lines+markers",
                                name=_v, showlegend=False,
                                line=dict(color=_col, width=2),
                                marker=dict(symbol=["line-ew", "square", "line-ew"],
                                            size=[8, 10, 8], color=_col),
                                hovertemplate=f"<b>{_v}</b><br>coef={_c:.3f}<br>95% CI [{_l:.3f},{_h:.3f}]<br>p={_pv:.4f}<extra></extra>"
                            ))
                        fig_cox.add_vline(x=0, line=dict(dash="dash", color="#94a3b8", width=1))
                        fig_cox.update_layout(
                            title="Forest Plot — Cox Model Coefficients",
                            xaxis=dict(title="Coefficient (log HR)", showgrid=True, gridcolor="#f1f5f9", zeroline=False),
                            yaxis=dict(autorange="reversed"),
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=max(350, len(_vars) * 32 + 100),
                            margin=dict(l=20, r=20, t=50, b=20),
                        )
                        st.plotly_chart(fig_cox, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_cox, "cox_fig")

                    model_name = st.text_input("Enter model name:", value="cox_model", help="Custom name for saving your Cox model")

                    if st.button("Save Cox Model", help="Save the trained Cox model and its preprocessing pipeline for blind predictions"):
                        if 'cph' in st.session_state:
                            pipeline.fit(survie)

                            # Save the model to a bytes buffer
                            model_buffer = io.BytesIO()
                            joblib.dump(st.session_state.cph, model_buffer)
                            model_buffer.seek(0)
                            st.session_state.model_buffer = model_buffer

                            # Save the pipeline to a bytes buffer
                            pipeline_buffer = io.BytesIO()
                            joblib.dump(pipeline, pipeline_buffer)
                            pipeline_buffer.seek(0)
                            st.session_state.pipeline_buffer = pipeline_buffer

                            # Set a flag to show the download buttons
                            st.session_state.show_downloads = True
                        else:
                            st.warning("Please run the Cox Model analysis before saving.")

                    # Show download buttons if ready
                    if st.session_state.get("show_downloads", False):
                        st.download_button(
                            label="Download Cox Model",
                            data=st.session_state.model_buffer,
                            file_name=f"{model_name}.pkl",
                            mime="application/octet-stream"
                        )
                        st.download_button(
                            label="Download Preprocessor Pipeline",
                            data=st.session_state.pipeline_buffer,
                            file_name="preprocessor_pipeline.pkl",
                            mime="application/octet-stream"
                        )                        
            else:
                st.warning("Please upload survival data to perform Cox Model analysis.")


        with _t7_pred:
            st.markdown('<p style="color: gray; font-size: 14px;">Import a CSV or Excel file and use a Saved Cox model to make predictions.</p>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
            model_file = st.file_uploader("Upload your pre-saved Cox model (.pkl)", type=["pkl"])
            pipeline_file = st.file_uploader("Upload your preprocessor pipeline (.pkl)", type=["pkl"])

            if uploaded_file and model_file and pipeline_file:
                # Load the CSV or Excel file
                if uploaded_file.name.endswith('.csv'):
                    data_pred = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data_pred = pd.read_excel(uploaded_file)

                # Load the pre-saved Cox model and pipeline
                cph = joblib.load(model_file)
                pipeline = joblib.load(pipeline_file)

                try:
                    if not hasattr(pipeline, "transform"):
                        raise NotFittedError("The pipeline has not been fitted.")

                    # Transform the data using the pipeline
                    X_transformed = pipeline.transform(data_pred)
                    X_transformed_df = pd.DataFrame(
                        X_transformed,
                        columns=pipeline.named_steps['preprocessor'].transformers_[0][2] +
                                list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
                    )

                    if st.button("Make Predictions", key="make_predictions_test_data"):
                        # Predict key survival metrics
                        median_survival = cph.predict_median(X_transformed_df)
                        overall_survival = cph.predict_expectation(X_transformed_df)  # Expected overall survival
                        hazard_ratios = cph.predict_partial_hazard(X_transformed_df)

                        results_df = data_pred.copy()
                        results_df["Predicted Overall Survival"] = overall_survival
                        results_df["Median Survival Time"] = median_survival
                        results_df["Hazard Ratio"] = hazard_ratios
                        # results_df["P-Value"] = p_values
                        st.success('Survival prediction successful')
                        # Display results
                        st.write("#### Survival Predictions Summary")
                        st.dataframe(results_df)
                        st.write("#### Descriptive statistics")
                        st.write(results_df[["Predicted Overall Survival", "Hazard Ratio"]].describe())
                        # Visualizations
                        st.write("#### Visualizations")

                        fig_med = px.histogram(
                            x=median_survival.values, nbins=30,
                            labels={"x": "Median Survival Time", "y": "Count"},
                            title="Distribution of Median Survival Time",
                            marginal="rug", opacity=0.75,
                            color_discrete_sequence=["#2563eb"],
                        )
                        fig_med.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                              height=380, showlegend=False)
                        st.plotly_chart(fig_med, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_med, "median_survival_distribution")

                        fig_hr = px.histogram(
                            x=hazard_ratios.values, nbins=30,
                            labels={"x": "Hazard Ratio", "y": "Count"},
                            title="Distribution of Hazard Ratios",
                            marginal="rug", opacity=0.75,
                            color_discrete_sequence=["#dc2626"],
                        )
                        fig_hr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                             height=380, showlegend=False)
                        st.plotly_chart(fig_hr, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_hr, "hazard_ratios_distribution")
                        # st.write(results_df.describe())
                


                except NotFittedError:
                    st.error("The uploaded pipeline has not been fitted. Please ensure it has been fitted before using it for predictions.")


    with tabs[8]:
        # ═══════════════════════════════════════════════════
        # REAL-TIME – sub-tabs
        # ═══════════════════════════════════════════════════
        (_t8_live, _t8_post, _t8_load) = st.tabs(["  Real-Time & Post-Acquisition","  Predict with Tabular Data", "  Load & Verify Model"])
        with _t8_live:
            watch_directory = st.text_input("Directory to monitor:")
            output_directory = st.text_input("Output directory for mzML files:")

            model_files = st.file_uploader("Upload model files", accept_multiple_files=True, type=["pkl"])
            feature_file = st.file_uploader("Upload feature file", type=["pkl"])
            label_encoder_file = st.file_uploader("Upload label encoder file", type=["pkl"])

            if st.button("Load All"):
                load_all_rt(model_files, feature_file, label_encoder_file)

            if st.session_state['label_encoder']:
                labels = st.session_state['label_encoder'].classes_
                for label in labels:
                    st.session_state['label_colors'][label] = st.color_picker(f"Color for {label}", "#FFFFFF")

            if st.button("Start Monitoring Directory"):
                if not st.session_state['numeric_features']:
                    st.warning("Please load the numeric features before starting monitoring.")
                else:
                    st.session_state['monitoring'] = True
                    st.write("Started monitoring...")

            if st.button("Stop Monitoring"):
                st.session_state['monitoring'] = False
                st.write("Stopped monitoring.")

            if st.session_state['monitoring'] and watch_directory and output_directory:
                new_files = convert_raw_to_mzml_rt(watch_directory, output_directory)
                if new_files:
                    mzml_files = [f for f in os.listdir(output_directory) if f.endswith(".mzML")]
                    for mzml_file in mzml_files:
                        if mzml_file in st.session_state['processed_files']:
                            continue
                        data = load_data_single_file_rt(os.path.join(output_directory, mzml_file))
                        if not data.empty:
                            processed_data = preprocess_data_rt(data, normalization_type=normalization_type)
                            # 🔹 Removed SVD choice — process directly
                            st.session_state['latest_result'] = decision_rt(processed_data)
                        st.session_state['processed_files'].add(mzml_file)

                if st.session_state['latest_result'] is not None:
                    st.write("Latest Prediction Results:")
                    st.dataframe(st.session_state['latest_result'])
                    visualize_predictions_circles_rt(st.session_state['latest_result'])

                time.sleep(2)
                # st.rerun()
                st.rerun()







        with _t8_post:

            uploaded_file = st.file_uploader("Upload CSV/XLSX File", type=["csv", "xlsx"], help="Upload your tabular data file. It can be labeled (with a 'Class' column) or unlabeled.")
            model_file = st.file_uploader("Upload Trained Model (Pickle)", type=["pkl"], help="Upload the trained classification model saved as a .pkl file.")
            feature_file = st.file_uploader("Upload Feature Names (Pickle)", type=["pkl"], help="Upload the file containing the ordered list of features used to train the model as a .pkl files.")
            label_encoder_file = st.file_uploader("Upload Label Encoder (Pickle)", type=["pkl"],help="Upload the label encoder as .pkl format used during training to map class labels.")

            col1, col2 = st.columns(2)
            with col1:
                predict_with_gt = st.button("Predict with Ground Truth",help="Use this if your file includes a 'Class' column to evaluate predictions with true labels.")
            with col2:
                predict_without_gt = st.button("Predict without Ground Truth",help="Use this if your file does not have true labels and you only want predictions.")

            def run_prediction(has_ground_truth):
                # pandas already imported
                import numpy as np
                import joblib
                from sklearn.metrics import confusion_matrix, classification_report
                import matplotlib.pyplot as plt
                import seaborn as sns

                if uploaded_file and model_file and feature_file and label_encoder_file:

                    file_extension = uploaded_file.name.split(".")[-1].lower()

                    if file_extension == "csv":
                        # 🔍 Détection automatique du séparateur
                        uploaded_file.seek(0)
                        sample = uploaded_file.read(4096).decode("utf-8-sig")
                        uploaded_file.seek(0)

                        sep = ";" if sample.count(";") > sample.count(",") else ","

                        df = pd.read_csv(
                            uploaded_file,
                            sep=sep,
                            encoding="utf-8-sig",
                            engine="python"
                        )

                    else:
                        df = pd.read_excel(uploaded_file)



                    # Load components
                    model_data = joblib.load(model_file)
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']  # ✅ extrait le vrai pipeline
                    else:
                        model = model_data  # ancien format (directement un pipeline)

                    feature_names = joblib.load(feature_file)
                    label_encoder = joblib.load(label_encoder_file)

                    if has_ground_truth and "Class" not in df.columns:
                        st.error("Missing 'Class' column in file.")
                        return

                    # Separate features
                    X = df.drop(columns=[c for c in df.columns if c in {"Class","ID","File","RT","Sum"} or str(c).endswith("_meta")], errors="ignore") if has_ground_truth else df.drop(columns=[c for c in df.columns if c in {"ID","File","RT","Sum"} or str(c).endswith("_meta")], errors="ignore")

                    # Check features
                    missing = set(feature_names) - set(X.columns)
                    extra = set(X.columns) - set(feature_names)

                    # Ajouter les features manquants avec valeur 0
                    if missing:
                        st.warning(f"Missing features detected and filled with 0: {missing}")
                        for col in missing:
                            X[col] = 0

                    # Supprimer les colonnes en trop
                    if extra:
                        st.warning(f"Extra features will be ignored: {extra}")
                        X = X.drop(columns=list(extra))

                    # Réordonner les colonnes pour correspondre au modèle
                    X = X[feature_names]

                    # Prédictions
                    predictions = model.predict(X)  # supposé aligné avec les lignes de df
                    predicted_labels = label_encoder.inverse_transform(predictions)
                    df.insert(0, "Predicted_Class", predicted_labels)



                    # ---------------- Confidence Scores (robuste tous modèles) ----------------
                    confidence_scores = None

                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(X)
                        confidence_scores = np.max(probas, axis=1)

                    elif hasattr(model, "decision_function"):
                        decision_scores = model.decision_function(X)

                        # Cas binaire → sortie 1D (Ridge, SGD, Perceptron…)
                        if decision_scores.ndim == 1:
                            confidence_scores = 1 / (1 + np.exp(-decision_scores))  # sigmoid

                        # Cas multi-classe → sortie 2D
                        else:
                            min_ = decision_scores.min(axis=1, keepdims=True)
                            max_ = decision_scores.max(axis=1, keepdims=True)
                            denom = np.where(max_ - min_ == 0, 1, max_ - min_)
                            norm_scores = (decision_scores - min_) / denom
                            confidence_scores = np.max(norm_scores, axis=1)

                    else:
                        confidence_scores = np.full(len(X), np.nan)



                    df.insert(1, "Confidence_Score", confidence_scores)

                    # Nettoyage (au cas où)
                    conf_clean = pd.Series(confidence_scores).dropna()

                    if len(conf_clean) == 0:
                        st.warning("No confidence scores available for this model.")
                    else:
                        high_conf_threshold = 0.7
                        high_conf_pct = (conf_clean >= high_conf_threshold).mean() * 100

                        st.info(
                            f"High-confidence predictions (≥ {high_conf_threshold}): "
                            f"**{high_conf_pct:.1f}%** of samples"
                        )

                        fig_conf = px.histogram(
                            conf_clean,
                            nbins=30,
                            opacity=0.85,
                            title="Distribution of Prediction Confidence Scores",
                            labels={"value": "Confidence Score"},
                            color_discrete_sequence=["#FF9800"]  # 
                        )

                        fig_conf.add_vline(
                            x=high_conf_threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="High confidence"
                        )

                        fig_conf.update_layout(
                            bargap=0.05,
                            height=400
                        )

                        st.plotly_chart(fig_conf, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                        _capture_plotly(fig_conf, "Features per Sample Confidence Scores Distribution")

                    # Affichage général
                    st.write("#### Prediction Results")
                    st.dataframe(df)

                    if has_ground_truth:
                        # --- Gestion des labels ground-truth inconnus ---
                        gt_raw = df["Class"]
                        # Nettoyage simple (supprime espaces en tête/queue)
                        gt_clean_str = gt_raw.astype(str).str.strip()

                        # Map stringified known classes vers leur valeur d'origine
                        classes_map = {str(c): c for c in label_encoder.classes_}
                        unseen_str = set(gt_clean_str.unique()) - set(classes_map.keys())

                        if unseen_str:
                            st.warning(f"Found unseen ground-truth labels (these rows will be excluded from metrics): {unseen_str}")

                        # mask des échantillons utilisables pour l'évaluation
                        mask_eval = ~gt_clean_str.isin(unseen_str)

                        if mask_eval.sum() == 0:
                            st.error("None of the ground-truth labels match the label encoder classes. Cannot compute evaluation metrics.")
                        else:
                            # Remapper les valeurs nettoyées vers leurs valeurs d'origine (types originaux)
                            gt_mapped = gt_clean_str.map(classes_map)

                            # Encodage pour la métrique
                            true_encoded = label_encoder.transform(gt_mapped[mask_eval])

                            # Récupérer les prédictions correspondantes (predictions est un ndarray aligné avec df)
                            pred_for_eval = predictions[mask_eval.values]

                            # Matrice de confusion et rapport
                            cm = confusion_matrix(true_encoded, pred_for_eval)
                            report = classification_report(
                                true_encoded,
                                pred_for_eval,
                                labels=list(range(len(label_encoder.classes_))),
                                target_names=list(label_encoder.classes_),
                                output_dict=True
                            )

                            # Construire la colonne Correct et Eval_included
                            correct = np.zeros(len(df), dtype=bool)
                            correct_pos = (true_encoded == pred_for_eval)
                            correct[mask_eval.values] = correct_pos
                            df["Correct"] = correct
                            df["Eval_included"] = mask_eval.values

                            misclassified = df[mask_eval & (~df["Correct"])]

                            # Plot confusion matrix (Plotly)
                            fig_cm = px.imshow(
                                cm,
                                x=list(label_encoder.classes_),
                                y=list(label_encoder.classes_),
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                color_continuous_scale="Blues", text_auto=True, aspect="auto",
                                title="Confusion Matrix"
                            )
                            fig_cm.update_layout(
                                height=400, plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                            _capture_plotly(fig_cm, "confusion_matrix")

                            st.markdown("**Classification Report**")
                            st.dataframe(pd.DataFrame(report).T)

                            st.write("#### Misclassified Samples (only among evaluated rows)")
                            st.dataframe(misclassified)

                    # Export
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
                else:
                    st.error("Please upload all required files (data, model, feature names, and label encoder).")



            # Trigger logic
            if predict_with_gt:
                run_prediction(has_ground_truth=True)

            if predict_without_gt:
                run_prediction(has_ground_truth=False)

        with _t8_load:
            st.markdown(
                '<p style="color: gray; font-size: 14px;">Load a previously saved Machine Learning or Deep Learning model and inspect its features, classes, and accuracy (if available).</p>',
                unsafe_allow_html=True
            )

            model_type_load = st.selectbox("Select Model Type", ['Machine Learning', 'Deep Learning'], key="load_model_type")
            model_file = st.file_uploader("Upload Model File (.pkl)", type=["pkl"], key="load_model_file")
            features_file = st.file_uploader("Upload Features File (.pkl)", type=["pkl"], key="load_features_file")
            label_encoder_file = st.file_uploader("Upload Label Encoder File (.pkl, optional)", type=["pkl"], key="load_label_encoder_file")

            if st.button("Load & Inspect Model", key="load_verify_btn"):
                if model_file is None or features_file is None:
                    st.error("Please upload both the model and features files.")
                else:
                    try:
                        # Chargement du modèle (dict ou direct)
                        loaded_obj = joblib.load(model_file)

                        if isinstance(loaded_obj, dict):
                            model = loaded_obj.get('model', None)
                            accuracy = loaded_obj.get('accuracy', None)
                        else:
                            model = loaded_obj
                            accuracy = None

                        # Chargement des features et du label_encoder
                        features = joblib.load(features_file)
                        label_encoder = joblib.load(label_encoder_file) if label_encoder_file else None

                        st.success("✅ Model loaded successfully!")

                        # Affichage des features
                        st.markdown("**🔹 Features used in training**")
                        st.dataframe(pd.DataFrame(features, columns=["Feature"]))

                        # Affichage modèle ML ou DL
                        if model_type_load == 'Machine Learning':
                            st.markdown("**🔹 Model Details**")
                            st.text(str(model))
                        elif model_type_load == 'Deep Learning':
                            st.markdown("**🔹 Model Architecture**")
                            try:
                                model_summary = []
                                model.summary(print_fn=lambda x: model_summary.append(x))
                                st.text("\n".join(model_summary))
                            except Exception as e:
                                st.warning(f"Cannot display architecture: {e}")

                        # Labels / classes
                        st.markdown("**🔹 Classes / Labels**")
                        if label_encoder:
                            st.write(label_encoder.classes_)
                        elif hasattr(model, "classes_"):
                            st.write(model.classes_)
                        else:
                            st.info("No label encoder or embedded class list found.")

                        # Accuracy si disponible
                        if accuracy is not None:
                            st.markdown(f"**Model Accuracy** {accuracy:.4f}")
                        else:
                            st.info("No accuracy stored in this model.")

                    except Exception as e:
                        st.error(f"Error loading model or metadata: {e}")

                    # Cleanup
                    try:
                        del model, features
                        gc.collect()
                    except:
                        pass



if __name__ == "__main__":
    import matplotlib
    # matplotlib.use('TkAgg')
    matplotlib.use('Agg')
    main()



