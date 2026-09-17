"""
Software Name: Profiler
Module Name: Features importance
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 16/06/2026
Version: 1.2.7

Context:
This module is part of the "Profiler" project, originally developed for a web version (https://prism-profiler.univ-lille.fr) and now adapted for a desktop version (profiler_desktop_GUI).
It is designed for archiving on Zenodo and integration into GitHub releases.

License: l’Agence pour la Protection des Programmes IDDN (InterDeposit Digital Number) : FR2 .0013 .0300044 .0005 .S6 .C7 .20258 .0009 .312301
Citation:
If Profiler or this module (a part of Profiler) is used in a publication, please cite:
Zirem, Y. (2025). Profiler: an open web platform for multi-omics analysis. Journal of Bioinformatics. doi:10.1093/bioinformatics/btaf644

Links:
- GitHub temporary Repository: https://github.com/yanisZirem/Profiler_v1_requests_datatests

"""

# --- Standard library ---
import io
import gc
import os
import profiler_perf  # noqa: F401 — budget CPU centralisé, importé avant numpy/TF/joblib
from itertools import combinations

# ── Desktop: budget CPU centralisé (voir profiler_perf.py) ───────────────────
# Ce module fixait auparavant ses propres OMP/MKL/OPENBLAS *et* réclamait
# tous les threads TensorFlow, indépendamment de profiler_preprocessing.py
# et profiler_training.py qui faisaient la même chose de leur côté : sur un
# PC normal, ces réglages concurrents empilés = sur-souscription (voir la
# note détaillée en tête de profiler_training.py). On délègue maintenant à
# profiler_perf, qui ne fixe ces variables qu'une seule fois pour tout le
# process.
_N_CPUS = profiler_perf.LOGICAL_CPUS
_N_JOBS = profiler_perf.OUTER_JOBS
profiler_perf.configure_tensorflow()


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: résoudre les noms de features (str ou float) vers les vraies colonnes
#  Nécessaire quand les colonnes sont des floats m/z (ex: 590.30254) issus de mzML
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_features(df, features):
    """
    Résout une liste de features (str ou float) vers les colonnes réelles du df.

    Cas d'usage : les colonnes du df sont des floats m/z (590.30254) mais les
    features passées sont des strings ('590.30254') ou inversement.

    Stratégie (par ordre de priorité) :
      1. Correspondance directe (colonne déjà présente telle quelle).
      2. Correspondance après conversion str→float (tolérance exacte numpy).
      3. Correspondance par round à 5 décimales pour absorber les micro-diffs
         de représentation flottante.

    Retourne la liste des colonnes réelles (dans l'ordre des features demandées).
    """
    col_index = {c: c for c in df.columns}           # exact match dict
    str_to_col = {str(c): c for c in df.columns}     # str(float) → col
    # round-5 index pour absorber les diffs de précision
    round5_to_col = {}
    for c in df.columns:
        try:
            round5_to_col[round(float(c), 5)] = c
        except (ValueError, TypeError):
            pass

    resolved = []
    seen = set()
    for f in features:
        if f in col_index and f not in seen:
            resolved.append(f); seen.add(f); continue
        sf = str(f)
        if sf in str_to_col and str_to_col[sf] not in seen:
            resolved.append(str_to_col[sf]); seen.add(str_to_col[sf]); continue
        try:
            fv = round(float(f), 5)
            if fv in round5_to_col and round5_to_col[fv] not in seen:
                resolved.append(round5_to_col[fv]); seen.add(round5_to_col[fv]); continue
        except (ValueError, TypeError):
            pass
        # Pas trouvé → on l'ignore silencieusement (évite KeyError)
    return resolved


# --- Scientific computing ---
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# --- Machine learning ---
import tensorflow as tf
import shap
import eli5
from eli5 import explain_prediction, explain_weights
from eli5.formatters import format_as_dataframe

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import (
    SGDClassifier, LogisticRegression, RidgeClassifier,
    PassiveAggressiveClassifier, Perceptron, Lasso
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
from statannotations.Annotator import Annotator

# --- Web and parsing ---
from bs4 import BeautifulSoup

# --- Streamlit app ---
import streamlit as st
import os




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




def _capture_plotly(fig, key: str):
    """Store a Plotly figure in session_state for the HTML report."""
    if fig is not None:
        st.session_state[f"_report_{key}"] = ("plotly", fig)


def eli5_feature_importance(model, label_encoder, data, top_features=50):
    """
    ELI5 / LIME feature importance.
    Robustly drops ALL non-numeric / meta columns before building X so that
    feature_names always matches the model input size.
    """
    # ── Drop every non-feature column regardless of position ─────────────────
    _NON_FEAT = {'Class', 'File', 'RT', 'Sum', 'ID', 'id',
                 'Index', 'sample_id', 'SampleID', 'patient_id', 'PatientID'}
    cols_to_drop = [c for c in data.columns
                    if c in _NON_FEAT or str(c).endswith('_meta')]
    X = (data.drop(columns=cols_to_drop, errors='ignore')
             .select_dtypes(include='number'))

    # ── Standardize ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.fillna(0).astype('float32')),
        columns=X.columns,
    )

    # ── Extract base estimator from pipeline ──────────────────────────────────
    if hasattr(model, 'named_steps'):
        fitted_model = model.named_steps[model.steps[-1][0]]
    else:
        fitted_model = model

    feature_names = [str(c) for c in X_scaled.columns]
    target_names  = list(label_encoder.classes_)

    # ── ELI5 weights explanation ──────────────────────────────────────────────
    explanation = explain_weights(
        fitted_model,
        feature_names=feature_names,
        top=top_features,
        target_names=target_names,
    )

    html_raw = eli5.format_as_html(explanation)
    df_contribution = format_as_dataframe(explanation)
    # Strip caveats section
    html_raw = html_raw.split('<div class="caveats">')[0] + '</div>'

    # ── Professional styling wrapper ──────────────────────────────────────
    styled = f"""
    <style>
      .eli5-wrap {{
        font-family: Arial, sans-serif;
        font-size: 13px;
        max-height: 520px;
        overflow-y: auto;
        border: 1px solid #dde1e7;
        border-radius: 8px;
        padding: 14px 18px;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      }}
      .eli5-wrap table {{
        border-collapse: collapse;
        width: 100%;
      }}
      .eli5-wrap th {{
        background: #f0f4f9;
        color: #222;
        font-weight: 700;
        padding: 7px 12px;
        border-bottom: 2px solid #c8d0dc;
        text-align: left;
        position: sticky;
        top: 0;
        z-index: 1;
      }}
      .eli5-wrap td {{
        padding: 5px 12px;
        border-bottom: 1px solid #edf0f4;
        color: #333;
      }}
      .eli5-wrap tr:hover td {{
        background: #f5f8ff;
      }}
      .eli5-wrap .eli5-pos-color {{ background: #c6efce; border-radius: 3px; }}
      .eli5-wrap .eli5-neg-color {{ background: #ffc7ce; border-radius: 3px; }}
    </style>
    <div class="eli5-wrap">{html_raw}</div>
    """

    return styled, df_contribution


def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 400)


def plot_shap_values(model, X, class_colors=None, class_names=None,
                     capture_prefix="shap", top_n: int = 20):
    """
    Fully Plotly-native SHAP visualisation.
    • Beeswarm (strip chart sorted by mean |SHAP|, coloured by feature value)
    • Bar chart (mean |SHAP| per feature)
    Both figures are stored in st.session_state for HTML report embedding,
    and offered as interactive Plotly downloads.
    """
    import shap
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import streamlit as st

    X = X.fillna(0)

    # ── Extract fitted model + transform X ────────────────────────────────
    if hasattr(model, "named_steps"):
        fitted_model = model.named_steps[model.steps[-1][0]]
        if len(model.steps) > 1:
            from sklearn.pipeline import Pipeline as _P
            try:
                X_transformed = _P(model.steps[:-1]).transform(X)
            except Exception:
                X_transformed = X.values if hasattr(X, "values") else X
        else:
            X_transformed = X.values if hasattr(X, "values") else X
    else:
        fitted_model = model
        X_transformed = X.values if hasattr(X, "values") else X

    _unsupported = {
        "AdaBoostClassifier", "BaggingClassifier", "SVC", "NuSVC", "LinearSVC",
        "GaussianNB", "BernoulliNB", "DummyClassifier", "NearestCentroid",
        "KNeighborsClassifier", "QuadraticDiscriminantAnalysis",
    }
    if type(fitted_model).__name__ in _unsupported:
        st.error(f"SHAP not supported for: {type(fitted_model).__name__}")
        return

    # ── Explainer selection ────────────────────────────────────────────────
    try:
        if isinstance(fitted_model, (
            RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier,
            ExtraTreeClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
        )):
            explainer = shap.TreeExplainer(fitted_model)
        elif isinstance(fitted_model, LGBMClassifier):
            explainer = shap.TreeExplainer(fitted_model, data=X_transformed)
        elif isinstance(fitted_model, (
            LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
            PassiveAggressiveClassifier, Lasso, LinearDiscriminantAnalysis,
        )):
            explainer = shap.LinearExplainer(fitted_model, X_transformed)
        else:
            st.warning("Using KernelExplainer (may be slow on large datasets).")
            # Desktop: jusqu'à 150 samples de background → meilleure précision SHAP
            _bg_size = min(150, X_transformed.shape[0])
            bg = shap.kmeans(X_transformed, _bg_size)
            fn = (fitted_model.predict_proba
                  if hasattr(fitted_model, "predict_proba")
                  else fitted_model.predict)
            explainer = shap.KernelExplainer(fn, bg)
    except Exception as e:
        st.error(f"SHAP explainer error: {e}")
        return

    with st.spinner("Computing SHAP values…"):
        shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    feat_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(sv.shape[1])]
    top_n = min(top_n, len(feat_names))

    # ── Rank features by mean |SHAP| ──────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]
    top_feat = [feat_names[i] for i in order]
    top_sv   = sv[:, order]          # (n_samples, top_n)
    top_Xv   = X_transformed[:, order] if hasattr(X_transformed, '__getitem__') else np.array(X_transformed)[:, order]

    PALETTE = px.colors.sequential.RdBu   # blue=low, red=high

    # ════════════════════════════════════════════════════════════════════════
    # 1. BEESWARM (strip chart)
    # ════════════════════════════════════════════════════════════════════════
    # Normalise feature values [0,1] for colour mapping
    col_min = top_Xv.min(axis=0, keepdims=True)
    col_max = top_Xv.max(axis=0, keepdims=True)
    col_rng = np.where((col_max - col_min) == 0, 1, col_max - col_min)
    norm_Xv  = (top_Xv - col_min) / col_rng          # [0,1]

    fig_bee = go.Figure()
    n_pts = top_sv.shape[0]

    for fi, fname in enumerate(top_feat):
        jitter = np.random.uniform(-0.30, 0.30, n_pts)
        feat_norm = norm_Xv[:, fi]

        # Map [0,1] → colour from RdBu palette (11 stops)
        palette_rgb = px.colors.sample_colorscale("RdBu", feat_norm.tolist())

        fig_bee.add_trace(go.Scatter(
            x=top_sv[:, fi],
            y=np.full(n_pts, fi) + jitter,
            mode="markers",
            marker=dict(
                color=feat_norm,
                colorscale="RdBu",
                size=6,
                opacity=0.75,
                line=dict(width=0),
                showscale=(fi == 0),
                colorbar=dict(
                    title=dict(text="Feature value<br>(normalised)",
                               font=dict(size=11, family="Arial")),
                    tickfont=dict(size=10, family="Arial"),
                    len=0.4, x=1.01,
                ) if fi == 0 else {},
            ),
            customdata=np.column_stack([
                [fname] * n_pts,
                np.round(top_sv[:, fi], 4),
                np.round(top_Xv[:, fi], 4),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "SHAP: %{customdata[1]}<br>"
                "Value: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
            name=fname,
        ))

    # Zero line
    fig_bee.add_vline(x=0, line=dict(color="black", width=1.2, dash="dot"))

    fig_bee.update_layout(
        title=dict(
            text="<b>SHAP Beeswarm</b>",
            font=dict(size=20, color="black", family="Arial Black"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="<b>SHAP value (impact on model output)</b>",
            titlefont=dict(size=14, color="black", family="Arial Black"),
            tickfont=dict(size=12, color="black", family="Arial"),
            showgrid=True, gridcolor="#ececec",
            zeroline=False, showline=True, linecolor="black", mirror=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(top_n)),
            ticktext=[f"<b>{f}</b>" for f in top_feat],
            tickfont=dict(size=11, color="black", family="Arial"),
            showgrid=False, showline=True, linecolor="black", mirror=True,
            autorange="reversed",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(420, 32 * top_n),
        width=820,
        margin=dict(l=170, r=80, t=65, b=60),
        legend=dict(font=dict(size=11)),
    )

    st.markdown("**SHAP Beeswarm**")
    st.plotly_chart(fig_bee, use_container_width=True)
    st.session_state[f"{capture_prefix}_beeswarm"] = ("plotly", fig_bee)

    # ════════════════════════════════════════════════════════════════════════
    # 2. BAR CHART — mean |SHAP|
    # ════════════════════════════════════════════════════════════════════════
    bar_vals  = mean_abs[order]          # already sorted desc
    bar_colors = px.colors.sample_colorscale(
        "Blues", [(v - bar_vals.min()) / (bar_vals.max() - bar_vals.min() + 1e-9)
                  for v in bar_vals]
    )

    fig_bar = go.Figure(go.Bar(
        x=bar_vals,
        y=top_feat,
        orientation="h",
        marker=dict(
            color=bar_vals,
            colorscale="Blues",
            line=dict(color="black", width=0.8),
            colorbar=dict(
                title=dict(text="Mean |SHAP|",
                           font=dict(size=11, family="Arial")),
                tickfont=dict(size=10, family="Arial"),
            ),
        ),
        text=[f"{v:.4f}" for v in bar_vals],
        textposition="outside",
        textfont=dict(size=11, color="black", family="Arial"),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))

    fig_bar.update_layout(
        title=dict(
            text="<b>SHAP Feature Importance</b>",
            font=dict(size=20, color="black", family="Arial Black"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="<b>Mean |SHAP value|</b>",
            titlefont=dict(size=14, color="black", family="Arial Black"),
            tickfont=dict(size=12, color="black", family="Arial"),
            showgrid=True, gridcolor="#ececec",
            zeroline=False, showline=True, linecolor="black", mirror=True,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="black", family="Arial"),
            showgrid=False, showline=True, linecolor="black", mirror=True,
            autorange="reversed",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(400, 30 * top_n),
        width=780,
        margin=dict(l=170, r=80, t=65, b=60),
    )

    st.markdown("**SHAP Feature Importance (Bar)**")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.session_state[f"{capture_prefix}_bar"] = ("plotly", fig_bar)

from scipy.stats import f_oneway
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import streamlit as st



# ─── Shared Plotly subplot builder ────────────────────────────────────────────
import plotly.subplots as _subplots
from scipy.stats import kruskal, mannwhitneyu, ttest_ind as _ttest
import gc as _gc

def _stat_test(group_a, group_b, test):
    """Return p-value for a pair of groups."""
    try:
        a = group_a.dropna().values
        b = group_b.dropna().values
        if len(a) < 2 or len(b) < 2:
            return 1.0
        if test in ("Kruskal", "Mann-Whitney"):
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        else:  # t-test / ANOVA fallback
            _, p = _ttest(a, b)
        return float(p)
    except Exception:
        return 1.0


def _pval_to_stars(p):
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def _make_feature_subplots(data, mz_values, class_colors, test, show_scatter,
                            use_log2, plot_type, capture_name, significance_dict=None,
                            pval_correction='None'):
    """
    Publication-ready subplot grid — box / violin / bar.

    Design rules
    ────────────
    • NO subplot_titles → causes overlap. Titles drawn as paper-ref
      annotations ABOVE each subplot domain with a guaranteed gap.
    • Fixed pixel dimensions per subplot (360×360) → square, never
      distorted by window width. use_container_width=False.
    • Scatter points drawn as separate Scatter traces layered ON top of
      box/violin (boxpoints=False / points=False on main traces).
    • Explicit y-axis range per subplot = data range + bracket headroom.
    • Brackets use data-coord y, domain-fraction x for correct centering.
    """
    MAX_PER_PAGE = 20
    data = data.copy()
    data.columns = data.columns.astype(str)
    # Résoudre les mz_values : str ou float → colonnes réelles (maintenant str)
    mz_values = [str(m) for m in mz_values]
    mz_values = _resolve_features(data, mz_values)
    # Forcer la conversion numérique pour éviter str/int TypeError
    for _f in mz_values:
        data[_f] = pd.to_numeric(data[_f], errors="coerce")
    label     = "Class"
    # Force class labels to plain str — numpy.int64 labels crash plotly
    # legendgroup, name, and other string-typed properties.
    data[label] = data[label].astype(str)
    classes   = sorted(data[label].dropna().unique())
    pairs     = list(combinations(classes, 2))
    n_cls     = len(classes)
    n_pairs   = len(pairs)
    color_map = {str(c): class_colors.get(c, class_colors.get(str(c), "#636EFA")) for c in classes}

    # ── Pagination ────────────────────────────────────────────────────────────
    if len(mz_values) > MAX_PER_PAGE:
        st.info(f"Showing {len(mz_values)} features — split into pages of {MAX_PER_PAGE}.")
        for b in range(0, len(mz_values), MAX_PER_PAGE):
            batch = mz_values[b:b + MAX_PER_PAGE]
            st.markdown(f"**Features {b+1}–{min(b+MAX_PER_PAGE, len(mz_values))}**")
            _make_feature_subplots(data, batch, class_colors, test, show_scatter,
                                   use_log2, plot_type,
                                   f"{capture_name}_p{b//MAX_PER_PAGE}" if capture_name else None,
                                   significance_dict=significance_dict,
                                   pval_correction=pval_correction)
        return

    # ── Condition selector — filtre uniquement les brackets p-value ──────────
    # Les boxplots de TOUTES les conditions restent affichés.
    # Seules les paires entre les conditions sélectionnées auront des brackets.
    if len(classes) > 2:
        # Clé stable : basée sur le nom du plot + les classes disponibles
        # JAMAIS sur id() ou len() qui changent à chaque re-render
        _stable_key = f"{capture_name or 'plot'}_{'_'.join(sorted(str(c) for c in classes))}"

        st.markdown("**Condition selection for p-value brackets**")
        _col1, _col2 = st.columns([2, 1])
        with _col1:
            _selected_classes = st.multiselect(
                "Select conditions for p-value brackets (all pairs shown if empty):",
                options=classes,
                default=st.session_state.get(f"cond_select_{_stable_key}", []),
                key=f"cond_select_{_stable_key}",
                help="All boxplots stay visible. Only the brackets between selected conditions are drawn. Select 2+ conditions to restrict brackets.",
            )
        with _col2:
            _bracket_mode = st.radio(
                "Brackets to show:",
                options=["All pairs", "Significant only"],
                index=0,
                key=f"brk_mode_{_stable_key}",
                help="'Significant only' hides ns brackets to reduce clutter.",
            )

        # Paires autorisées pour les brackets : uniquement entre conditions sélectionnées
        # Les données et les boxplots ne sont PAS filtrés
        if len(_selected_classes) >= 2:
            _bracket_pairs = set(combinations(_selected_classes, 2)) | \
                             set(combinations(reversed(_selected_classes), 2))
        elif len(_selected_classes) == 1:
            st.warning("Please select at least 2 conditions for brackets.")
            _bracket_pairs = None  # aucune restriction
        else:
            _bracket_pairs = None  # vide = tout afficher

        _show_ns_brackets = (_bracket_mode == "All pairs")
    else:
        _bracket_pairs    = None
        _show_ns_brackets = True

    n     = len(mz_values)
    ncols = min(4, max(1, n))
    nrows = max(1, int(np.ceil(n / ncols)))

    # ── Fixed square pixel size per subplot ───────────────────────────────────
    # Each subplot cell = 360 px wide × 360 px tall (square).
    # For multi-class (many brackets), add extra height per pair.
    # Margins: left 70 (y-axis label), right 120 (legend), top 60, bottom 50.
    CELL_W    = 360
    _extra    = max(0, min(n_pairs - 1, 8) * 28)   # cap at 8 brackets worth of extra height
    CELL_H    = 360 + _extra
    H_GAP_PX  = 80    # horizontal gap between cells (for y-axis labels + padding)
    V_GAP_PX  = 80 + max(0, min(n_pairs - 1, 4) * 10)   # capped extra vertical gap
    MARGIN_L  = 70
    MARGIN_R  = 130
    MARGIN_T  = 50    # small — titles drawn inside this space above first row
    MARGIN_B  = 55

    fig_w = MARGIN_L + ncols * CELL_W + (ncols - 1) * H_GAP_PX + MARGIN_R
    fig_h = MARGIN_T + nrows * CELL_H + (nrows - 1) * V_GAP_PX + MARGIN_B

    # horizontal / vertical spacing as fractions of total figure size
    h_spacing = H_GAP_PX / fig_w if ncols > 1 else 0.0
    v_spacing = V_GAP_PX / fig_h if nrows > 1 else 0.0

    # ── Build subplot domains (same formula as make_subplots) ─────────────────
    # We need these to place paper-ref annotations precisely.
    plot_area_w = fig_w - MARGIN_L - MARGIN_R
    plot_area_h = fig_h - MARGIN_T - MARGIN_B
    col_w_frac  = (CELL_W) / plot_area_w
    row_h_frac  = (CELL_H) / plot_area_h
    h_gap_frac  = H_GAP_PX  / plot_area_w
    v_gap_frac  = V_GAP_PX  / plot_area_h

    def _domain(row_1idx, col_1idx):
        """Return (x0, x1, y0, y1) in paper [0-1] for subplot (row, col)."""
        # Plotly paper coords: x=0 is left edge of plot area, y=0 is bottom
        # BUT margins shift things: paper 0→1 spans the entire figure.
        # We compute relative to the plot area then convert.
        ml_frac = MARGIN_L / fig_w
        mb_frac = MARGIN_B / fig_h
        c = col_1idx - 1
        r = row_1idx - 1
        x0 = ml_frac + c * (col_w_frac + h_gap_frac) * (plot_area_w / fig_w)
        x1 = x0 + col_w_frac * (plot_area_w / fig_w)
        # y in paper: row 0 is top, paper y grows upward
        y1 = 1.0 - (MARGIN_T / fig_h) - r * (row_h_frac + v_gap_frac) * (plot_area_h / fig_h)
        y0 = y1 - row_h_frac * (plot_area_h / fig_h)
        return x0, x1, y0, y1

    # ── Pre-scan: ALL pairwise comparisons for every feature ──────────────────
    # For 2 classes  → 1 bracket, use significance_dict p if available.
    # For 3+ classes → compute ALL pairwise Mann-Whitney p-values with
    #                  Bonferroni correction; display one bracket per pair,
    #                  stacked at increasing heights to avoid overlap.

    def _sig_for(mz):
        """Return list of (class_a, class_b, p_raw_pairwise) for all pairs.

        Strategy — consistent with the significance counter in Profiler.py:
        • Binary (1 pair): use significance_dict[mz] directly — same value
          that determined significant/non-significant in the counter.
        • Multi-class (≥3 classes): the counter uses a GLOBAL test (Kruskal/ANOVA)
          on all groups at once. Brackets show POST-HOC pairwise p-values using
          the same pairwise test, with the user-chosen correction applied across
          the pairs of THIS feature only (not across features).
          We do NOT apply a second cross-feature correction here — that would
          produce different results from the counter.
        """
        raw = data[mz].replace([np.inf, -np.inf], np.nan)
        if use_log2:
            raw = np.log2(raw + 1e-9)
        cv = {c: raw.loc[data[label] == c].dropna() for c in classes}

        if n_pairs == 1:
            # Binary: use the precomputed (and already corrected) p-value
            ca, cb = pairs[0]
            if significance_dict is not None and mz in significance_dict:
                return [(ca, cb, float(significance_dict[mz]))]
            return [(ca, cb, _stat_test(cv[ca], cv[cb], test))]

        # Multi-class: compute raw pairwise p-values with the chosen test
        raw_pvals = [_stat_test(cv[ca], cv[cb], test) for ca, cb in pairs]

        # Apply user-chosen correction across the pairs of this feature
        try:
            from statsmodels.stats.multitest import multipletests as _mt
            _corr_map = {
                'Bonferroni': 'bonferroni',
                'FDR (Benjamini-Hochberg)': 'fdr_bh',
            }
            _method = _corr_map.get(pval_correction, None)
            if _method and not all(p >= 1.0 for p in raw_pvals):
                _, corrected, _, _ = _mt(raw_pvals, method=_method)
                corrected = list(corrected)
            else:
                corrected = raw_pvals  # 'None': raw pairwise p-values
        except Exception:
            corrected = raw_pvals

        return [(ca, cb, float(p)) for (ca, cb), p in zip(pairs, corrected)]

    # ── Create figure (NO subplot_titles) ─────────────────────────────────────
    fig = _subplots.make_subplots(
        rows=nrows, cols=ncols,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    # Shared axis style
    _ax = dict(
        showgrid=True, gridcolor="#e8e8e8", gridwidth=0.5,
        linecolor="black", linewidth=1.5, mirror=True,
        ticks="outside", tickcolor="black",
        tickfont=dict(size=11, color="black", family="Arial"),
        zeroline=False,
    )

    progress = st.progress(0)

    for idx, mz in enumerate(mz_values):
        r, c     = divmod(idx, ncols)
        row, col = r + 1, c + 1
        show_leg = (idx == 0)

        # ── Feature data ──────────────────────────────────────────────────────
        raw      = data[mz].replace([np.inf, -np.inf], np.nan)
        if use_log2:
            raw = np.log2(raw + 1e-9)
        col_data = raw
        y_label  = "log₂(Intensity)" if use_log2 else "Intensity"
        cls_vals = {cls: col_data.loc[data[label] == cls].dropna() for cls in classes}

        # ── Main shape traces — points natifs Plotly DANS la forme ──────────────
        # Stratégie : boxpoints/points natif Plotly → dots INSIDE box/violin/bar
        # jitter=0.4 → dispersion horizontale; pointpos=0 → centré dans la forme.
        for cls in classes:
            yv    = cls_vals[cls]
            color = color_map[cls]
            n_pts = len(yv)

            # Jitter adaptatif
            _jitter   = 0.5 if n_pts > 10 else (0.3 if n_pts > 4 else 0.1)
            _pt_style = dict(
                color="rgba(255,255,255,0.88)",
                size=6,
                line=dict(width=1.6, color=color),
                symbol="circle",
            )
            _pts_mode = "all" if show_scatter else False

            if plot_type == "violin":
                fig.add_trace(go.Violin(
                    y=yv, name=cls, legendgroup=cls, showlegend=show_leg,
                    line_color=color, fillcolor=color, opacity=0.70,
                    box_visible=True,
                    meanline_visible=True,
                    meanline=dict(color="white", width=2),
                    points=_pts_mode,
                    jitter=_jitter,
                    pointpos=0,
                    marker=_pt_style,
                    spanmode="soft",
                    x=[classes.index(cls)] * max(len(yv), 1),
                    hovertemplate=f"<b>{cls}</b><br>%{{y:.3f}}<extra></extra>",
                ), row=row, col=col)

            elif plot_type == "bar":
                mean_v = float(yv.mean()) if len(yv) else 0.0
                std_v  = float(yv.std())  if len(yv) > 1 else 0.0
                fig.add_trace(go.Bar(
                    x=[classes.index(cls)], y=[mean_v],
                    name=cls, legendgroup=cls, showlegend=show_leg,
                    marker=dict(color=color, opacity=0.50,
                                line=dict(color=color, width=1.3)),
                    error_y=dict(type="data", array=[std_v],
                                 visible=True, color="black",
                                 thickness=1.5, width=6),
                    hovertemplate=f"<b>{cls}</b><br>Mean±SD: %{{y:.3f}}<extra></extra>",
                ), row=row, col=col)
                if show_scatter and n_pts:
                    fig.add_trace(go.Box(
                        y=yv, x=[classes.index(cls)] * n_pts,
                        name=cls, legendgroup=cls, showlegend=False,
                        fillcolor="rgba(0,0,0,0)",
                        line=dict(color="rgba(0,0,0,0)", width=0),
                        whiskerwidth=0,
                        boxpoints="all",
                        jitter=_jitter,
                        pointpos=0,
                        marker=_pt_style,
                        hovertemplate=f"<b>{cls}</b><br>%{{y:.3f}}<extra></extra>",
                    ), row=row, col=col)

            else:  # box
                fig.add_trace(go.Box(
                    y=yv, name=cls, legendgroup=cls, showlegend=show_leg,
                    x=[classes.index(cls)] * max(len(yv), 1),
                    line=dict(color=color, width=2),
                    fillcolor=color, opacity=0.75,
                    boxmean="sd",
                    boxpoints=_pts_mode,
                    jitter=_jitter,
                    pointpos=0,
                    marker=_pt_style,
                    hovertemplate=f"<b>{cls}</b><br>%{{y:.3f}}<extra></extra>",
                ), row=row, col=col)

        # ── Y-range with bracket headroom ─────────────────────────────────────
        all_y = np.concatenate([v.values for v in cls_vals.values() if len(v)])
        if not len(all_y):
            progress.progress(min((idx + 1) / n, 1.0))
            continue

        y_min   = float(np.nanmin(all_y))
        y_max   = float(np.nanmax(all_y))
        y_range = max(abs(y_max - y_min), abs(y_max) * 0.02, 1e-6)

        all_sig_pairs = _sig_for(mz)

        # ── Filter brackets based on user's bracket mode selection ────────────
        if _show_ns_brackets:
            display_pairs = list(all_sig_pairs)
        else:
            display_pairs = [(ca, cb, p) for ca, cb, p in all_sig_pairs if p < 0.05]

        # Filtrer par les conditions sélectionnées (si restriction active)
        if _bracket_pairs is not None:
            display_pairs = [(ca, cb, p) for ca, cb, p in display_pairs
                             if (ca, cb) in _bracket_pairs or (cb, ca) in _bracket_pairs]

        # Sort by span (short first = lower bracket) to avoid overlap.
        display_pairs = sorted(display_pairs,
            key=lambda t: abs(classes.index(t[1]) - classes.index(t[0])))

        n_brk    = len(display_pairs)
        # Use a larger step for many brackets to avoid crowding
        step     = y_range * max(0.14, 0.10 + 0.02 * min(n_brk, 8))
        headroom = step * (0.5 + n_brk * 1.2) if n_brk else step * 0.2
        y_hi     = y_max + headroom
        y_lo     = y_min - y_range * 0.05

        fig.update_yaxes(range=[y_lo, y_hi], row=row, col=col)

        # ── Subplot axis refs ─────────────────────────────────────────────────
        ax_idx = "" if idx == 0 else str(idx + 1)
        xref   = f"x{ax_idx}"
        yref   = f"y{ax_idx}"

        # ── x position = numeric index (matches traces which now use x=[i]) ──
        def _data_x(cls_name):
            return float(classes.index(cls_name))

        # ── p-value formatter ─────────────────────────────────────────────────
        def _fmt_p(pv):
            if pv >= 1.0:   return "1"
            if pv < 0.0001: return f"{pv:.2e}"
            if pv < 0.001:  return f"{pv:.4f}"
            if pv < 0.01:   return f"{pv:.3f}"
            if pv < 0.1:    return f"{pv:.3f}"
            return f"{pv:.2f}"

        # ── Significance brackets — all in numeric data coords ────────────────
        for brk_k, (ca, cb, p) in enumerate(display_pairs):
            stars     = _pval_to_stars(p)
            is_sig    = p < 0.05
            tick_h    = step * 0.22
            br_y      = y_max + step * (0.5 + brk_k * 1.2)
            ann_color = "black" if is_sig else "#888888"
            brk_color = "black" if is_sig else "#aaaaaa"
            p_str     = _fmt_p(p)
            ann_text  = f"<b>{stars}</b> p={p_str}" if is_sig else f"<i>ns</i> p={p_str}"

            xa       = _data_x(ca)
            xb       = _data_x(cb)
            x_centre = (xa + xb) / 2

            fig.add_shape(type="line", x0=xa, x1=xb, y0=br_y, y1=br_y,
                xref=xref, yref=yref, line=dict(color=brk_color, width=1.8))
            fig.add_shape(type="line", x0=xa, x1=xa, y0=br_y - tick_h, y1=br_y,
                xref=xref, yref=yref, line=dict(color=brk_color, width=1.8))
            fig.add_shape(type="line", x0=xb, x1=xb, y0=br_y - tick_h, y1=br_y,
                xref=xref, yref=yref, line=dict(color=brk_color, width=1.8))

            fig.add_annotation(
                x=x_centre, y=br_y + step * 0.20,
                xref=xref, yref=yref,
                text=ann_text,
                showarrow=False,
                font=dict(size=10, color=ann_color, family="Arial"),
                bgcolor="rgba(255,255,255,0.88)", borderpad=2,
            )

        # ── Feature title above all brackets ──────────────────────────────────
        x_plot_centre = (_data_x(classes[0]) + _data_x(classes[-1])) / 2
        title_y = y_hi + step * 0.35

        if significance_dict is not None and mz in significance_dict:
            global_p = significance_dict[mz]
            g_stars = _pval_to_stars(global_p)
            g_str   = _fmt_p(global_p)
            if global_p < 0.05:
                title_text = f"<b>{mz}</b>  <span style='color:#c00'>{g_stars} p={g_str}</span>"
            else:
                title_text = f"<b>{mz}</b>  <span style='color:#888'><i>ns</i> p={g_str}</span>"
        else:
            title_text = f"<b>{mz}</b>"

        fig.add_annotation(
            x=x_plot_centre, y=title_y,
            xref=xref, yref=yref,
            text=title_text,
            showarrow=False,
            font=dict(size=13, color="black", family="Arial"),
            xanchor="center", yanchor="bottom",
        )

        # ── Axis styling — numeric x with class name labels ───────────────────
        fig.update_xaxes(
            **_ax,
            tickmode="array",
            tickvals=list(range(n_cls)),
            ticktext=classes,
            tickangle=-30 if n_cls > 3 else 0,
            row=row, col=col,
        )
        fig.update_yaxes(
            **_ax,
            title_text=y_label if col == 1 else "",
            title_font=dict(size=12, color="black", family="Arial"),
            range=[y_lo, title_y + step * 0.5],
            row=row, col=col,
        )
        progress.progress(min((idx + 1) / n, 1.0))

    # ── Global layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        width=fig_w,
        height=fig_h,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=12, family="Arial"),
        legend=dict(
            bgcolor="white", bordercolor="#888", borderwidth=1,
            font=dict(size=12, color="black", family="Arial"),
        ),
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="black"),
        violingap=0.3,
        boxgap=0.3,
        bargap=0.3,
    )

    # Render with fixed size — use_container_width=False keeps square aspect
    st.plotly_chart(fig, use_container_width=False, config={
        "displayModeBar": True,
        "modeBarButtonsToKeep": ["zoom2d","pan2d","zoomIn2d","zoomOut2d",
                                  "autoScale2d","resetScale2d","toImage"],
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png", "scale": 3,
            "filename": "feature_comparison",
            "width": fig_w, "height": fig_h,
        },
    })
    if capture_name:
        _capture_plotly(fig, capture_name)
    del fig; _gc.collect()


# ─── Public API ────────────────────────────────────────────────────────────────
def boxplot_significant_features(data, mz_values, class_colors=None, test="Kruskal",
                                  loc="inside", show_scatter=False, use_log2=False,
                                  capture_name=None, significance_dict=None,
                                  pval_correction='None'):
    _make_feature_subplots(data, mz_values, class_colors or {}, test,
                            show_scatter, use_log2, "box", capture_name,
                            significance_dict=significance_dict,
                            pval_correction=pval_correction)


def violinplot_significant_features(data, mz_values, class_colors=None, test="Kruskal",
                                     loc="inside", show_scatter=False, use_log2=False,
                                     capture_name=None, significance_dict=None,
                                     pval_correction='None'):
    _make_feature_subplots(data, mz_values, class_colors or {}, test,
                            show_scatter, use_log2, "violin", capture_name,
                            significance_dict=significance_dict,
                            pval_correction=pval_correction)


def barplot_significant_features(data, mz_values, class_colors=None, test="Kruskal",
                                  loc="inside", show_scatter=False, use_log2=False,
                                  capture_name=None, significance_dict=None,
                                  pval_correction='None'):
    _make_feature_subplots(data, mz_values, class_colors or {}, test,
                            show_scatter, use_log2, "bar", capture_name,
                            significance_dict=significance_dict,
                            pval_correction=pval_correction)


def eli5_format_to_dataframe(eli5_html):
    """
    Convert an ELI5/LIME HTML table into a Pandas DataFrame, handling multi-row headers and duplicate column names.
    """
    if hasattr(eli5_html, "data"):  
        eli5_html = eli5_html.data  

    soup = BeautifulSoup(eli5_html, "html.parser")  
    table = soup.find("table")

    if not table:
        return pd.DataFrame()  

    header_rows = table.find_all("tr")[:2]  # First two rows are headers
    headers = [[th.get_text(strip=True) for th in row.find_all("th")] for row in header_rows]

    # Fill missing headers (if any)
    max_columns = max(len(h) for h in headers)
    headers = [row + [""] * (max_columns - len(row)) for row in headers]  

    # Ensure uniqueness by adding an index if needed
    combined_headers = []
    seen_headers = {}
    for i, col_group in enumerate(zip(*headers)):  
        main_header, sub_header = col_group
        main_header = main_header or f"Category_{i}"  # Assign unique names if empty

        column_name = f"{main_header} - {sub_header}".strip()

        # Ensure uniqueness by adding a counter if needed
        if column_name in seen_headers:
            seen_headers[column_name] += 1
            column_name += f"_{seen_headers[column_name]}"
        else:
            seen_headers[column_name] = 1

        combined_headers.append(column_name)

    # Extract data rows
    rows = []
    for tr in table.find_all("tr")[2:]:  # Skip header rows
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        rows.append(cells)

    # Normalize row lengths
    rows = [row + [None] * (max_columns - len(row)) for row in rows]

    # Create DataFrame with unique column names
    df = pd.DataFrame(rows, columns=combined_headers)

    return df


def calculate_volcano_data(
    data, class_column, features,
    p_value_threshold=0.05, correction_method="fdr_bh",
    control_class=None
):
    # ── Résoudre les features (str vs float columns) ──────────────────────────
    data = data.copy()
    data.columns = [str(c) for c in data.columns]      # normalise TOUT en str
    features = [str(f) for f in features]               # idem pour les features
    features = _resolve_features(data, features)        # filtre les manquantes

    # Force class labels to str to avoid numpy.int64 in plotly properties
    data[class_column] = data[class_column].astype(str)
    classes = list(data[class_column].unique())
    results = []

    # ── If binary + control_class specified, reorder so control is always first ─
    if control_class is not None and control_class in classes and len(classes) == 2:
        other = [c for c in classes if c != control_class]
        classes = [control_class] + other

    grouped = {c: data.loc[data[class_column] == c, features] for c in classes}

    for i, control_class in enumerate(classes):
        for test_class in classes[i+1:]:
            control_data = grouped[control_class]
            test_data = grouped[test_class]

            t_stats, p_values = ttest_ind(
                control_data, test_data,
                axis=0, equal_var=False, nan_policy='omit'
            )

            # --- Multiple testing correction ---
            if correction_method and correction_method.lower() != "none":
                try:
                    reject, p_adj, _, _ = multipletests(
                        p_values, method=correction_method
                    )
                    p_values = p_adj
                except Exception as e:
                    print(f"Correction failed ({correction_method}): {e}")

            mean_control = np.nanmean(control_data, axis=0)
            mean_test = np.nanmean(test_data, axis=0)
            fold_change = np.divide(
                mean_control, mean_test,
                out=np.full_like(mean_control, np.nan), where=mean_test!=0
            )

            regulation = np.where(
                p_values < p_value_threshold,
                np.where(fold_change > 1, 'Upregulated',
                         np.where(fold_change < 1, 'Downregulated', 'Non-Significant')),
                'Non-Significant'
            )

            df = pd.DataFrame({
                'Feature': features,
                'Comparison': f"{control_class} vs {test_class}",
                'Fold Change': fold_change,
                'P-Value': p_values,
                'Regulation Type': regulation
            })
            results.append(df)

    df_all = pd.concat(results, ignore_index=True)
    df_all['Log2 Fold Change'] = np.log2(np.nan_to_num(df_all['Fold Change'], nan=np.nan, posinf=np.nan, neginf=np.nan))
    df_all['-Log10 P-Value'] = -np.log10(df_all['P-Value'].replace(0, np.nan))

    if len(classes) == 2:
        df_all['Color_Group'] = np.where(
            df_all['P-Value'] >= p_value_threshold, 'Non-Significant',
            np.where(df_all['Log2 Fold Change'] > 0, 'Upregulated', 'Downregulated')
        )
    else:
        df_all['Color_Group'] = np.where(df_all['P-Value'] >= p_value_threshold, 'Non-Significant', df_all['Comparison'])

    return df_all



# ── Thème graphique partagé (profiler_plot_theme) ───────────────────────────
# Import tolérant : si le module n'est pas déployé, on retombe sur un template
# Plotly standard au lieu de casser l'application.
try:
    import profiler_plot_theme as _theme          # enregistre le template
    _PROFILER_TEMPLATE = "profiler"
except Exception:                                  # pragma: no cover
    _theme = None
    _PROFILER_TEMPLATE = "plotly_white"


PALETTE_VOLCANO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                   "#E69F00", "#56B4E9", "#F0E442", "#8172B3"]


def plot_volcano(volcano_data, highlight_features=True, p_value_threshold=0.05,
                 fold_change_threshold=2.0, capture_name=None,
                 max_labels=15, label_features=None):
    """
    Volcano plot rapide et lisible.

    Changements par rapport à la version précédente :
      • Les étiquettes ne sont plus posées sur TOUS les points. Avec 5 000
        features, Plotly devait placer 5 000 textes → plusieurs secondes de
        rendu et un nuage illisible. On n'étiquette que les `max_labels`
        points les plus significatifs (ou la liste `label_features` fournie).
      • Rendu WebGL au-delà de ~1 200 points : le SVG sature vers 3 000
        marqueurs, la WebGL encaisse des centaines de milliers.
      • Les non-significatifs sont dessinés en premier, en gris translucide
        et sans contour : ils deviennent un fond, pas du bruit visuel.
      • Style unifié via le template `profiler`.

    `highlight_features=False` supprime toute étiquette.
    """
    import plotly.graph_objects as _go

    df = volcano_data.copy()
    df = df[np.isfinite(df["Log2 Fold Change"]) & np.isfinite(df["-Log10 P-Value"])]
    if df.empty:
        st.warning("No finite values to display in the volcano plot.")
        return None

    binary = "Upregulated" in df["Color_Group"].values
    if binary:
        color_map = {"Upregulated": "#D55E00",
                     "Downregulated": "#0072B2",
                     "Non-Significant": "#BDBDBD"}
    else:
        color_map = {"Non-Significant": "#BDBDBD"}
        for i, comp in enumerate(df["Comparison"].unique()):
            color_map[comp] = PALETTE_VOLCANO[i % len(PALETTE_VOLCANO)]

    n_pts = len(df)
    scatter_cls = _go.Scattergl if n_pts >= 1200 else _go.Scatter
    marker_size = 9 if n_pts < 500 else (6 if n_pts < 5000 else 4)

    fig = _go.Figure()

    # Non-significatifs d'abord → ils passent sous les points d'intérêt
    order = sorted(df["Color_Group"].unique(),
                   key=lambda g: 0 if g == "Non-Significant" else 1)
    for grp in order:
        sub = df[df["Color_Group"] == grp]
        is_ns = (grp == "Non-Significant")
        fig.add_trace(scatter_cls(
            x=sub["Log2 Fold Change"], y=sub["-Log10 P-Value"],
            mode="markers", name=str(grp),
            marker=dict(size=marker_size if not is_ns else marker_size * 0.8,
                        color=color_map.get(grp, "#888"),
                        opacity=0.35 if is_ns else 0.85,
                        line=dict(width=0)),
            # customdata = le nom de la feature seulement : les deux autres
            # champs étaient constants par trace et triplaient le JSON envoyé
            # au navigateur pour rien.
            customdata=sub["Feature"].astype(str).values,
            hovertemplate=("<b>%{customdata}</b><br>"
                           "log2FC = %{x:.3f}<br>"
                           "-log10 p = %{y:.3f}"
                           "<extra>%{fullData.name}</extra>"),
        ))

    # ── Étiquettes : uniquement les plus significatives ─────────────────────
    if highlight_features:
        if label_features:
            lab = df[df["Feature"].astype(str).isin([str(f) for f in label_features])]
        else:
            sig = df[(df["P-Value"] < p_value_threshold)
                     & (df["Log2 Fold Change"].abs() >= np.log2(fold_change_threshold))]
            pool = sig if not sig.empty else df
            score = pool["-Log10 P-Value"] * pool["Log2 Fold Change"].abs()
            lab = pool.loc[score.nlargest(min(max_labels, len(pool))).index]
        if not lab.empty:
            fig.add_trace(_go.Scatter(
                x=lab["Log2 Fold Change"], y=lab["-Log10 P-Value"],
                mode="text", text=lab["Feature"].astype(str),
                textposition="top center",
                textfont=dict(size=10, color="#111", family="Arial"),
                hoverinfo="skip", showlegend=False, cliponaxis=False,
            ))

    # ── Seuil de significativité ─────────────────────────────────────────────
    # Afficher uniquement le seuil de p-value.
    # Aucun seuil graphique de fold-change n'est tracé afin d'éviter toute
    # ligne oblique/indésirable sur le volcano plot.
    y_thr = -np.log10(p_value_threshold)
    fig.add_hline(y=y_thr, line_dash="dot", line_color="#777", line_width=1)
    fig.add_annotation(x=float(df["Log2 Fold Change"].min()), y=y_thr,
                       text=f"p = {p_value_threshold}", showarrow=False,
                       xanchor="left", yanchor="bottom",
                       font=dict(size=10, color="#777"))

    # ── Mise en forme ────────────────────────────────────────────────────────
    xmax = float(np.nanmax(np.abs(df["Log2 Fold Change"]))) * 1.08 or 1.0
    fig.update_layout(
        template=_PROFILER_TEMPLATE,
        title=dict(text="<b>Volcano plot</b>"),
        xaxis=dict(title_text="log<sub>2</sub> fold change",
                   range=[-xmax, xmax], zeroline=True, zerolinecolor="#DDD"),
        yaxis=dict(title_text="-log<sub>10</sub> p-value", rangemode="tozero"),
        legend=dict(title=dict(text="<b>Regulation</b>"),
                    orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"),
        hovermode="closest",
        margin=dict(l=70, r=160, t=55, b=60),
        height=620,
    )

    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig




def detect_peaks(data, intensity_threshold, show_stats=True):
    """
    Detects features (columns) in the dataset where peaks are present above a given intensity threshold.

    Parameters:
        data (pd.DataFrame): Input data with features as columns.
        intensity_threshold (float): Minimum intensity required to consider a peak.
        show_stats (bool): Whether to print min/max intensity info for each feature.

    Returns:
        list: Names of features with at least one detected peak above the threshold.
    """
    # ⚡ Joblib parallel peak detection across columns
    import joblib
    excluded_cols = {'Class', 'File', 'RT', 'Sum'}
    cols = [c for c in data.columns if c not in excluded_cols]

    def _check_peak(col):
        try:
            intensities = data[col].to_numpy(dtype='float32', na_value=0.0)
            peaks, _ = find_peaks(intensities, height=intensity_threshold)
            return col if len(peaks) > 0 else None
        except Exception:
            return None

    peak_features = [
        r for r in joblib.Parallel(n_jobs=_N_JOBS, prefer='threads')(
            joblib.delayed(_check_peak)(c) for c in cols
        ) if r is not None
    ]
    return peak_features
def plot_significant_features(data, mz_values, class_colors=None, test='Kruskal', 
                              plot_type='box', show_scatter=False, use_log2=False, 
                              pval_correction='None', significance_dict=None, capture_name=None):
    """
    Wrapper function to call the appropriate plotting function based on plot_type.
    
    Parameters:
    - data: DataFrame with the data
    - mz_values: list of feature names to plot
    - class_colors: dict mapping class labels to colors
    - test: statistical test to use
    - plot_type: 'box', 'violin', or 'bar'
    - show_scatter: whether to overlay individual points
    - use_log2: whether to apply log2 transformation
    - pval_correction: correction method applied to pairwise brackets in multi-class mode
                       ('None', 'Bonferroni', 'FDR (Benjamini-Hochberg)')
    - significance_dict: dict mapping features to adjusted p-values (used for binary comparison)
    """
    
    # Map plot type to the appropriate function
    if plot_type == 'box':
        boxplot_significant_features(data, mz_values, class_colors, test, 
                                     loc='inside', show_scatter=show_scatter, 
                                     use_log2=use_log2, capture_name=capture_name,
                                     significance_dict=significance_dict,
                                     pval_correction=pval_correction)
    elif plot_type == 'violin':
        violinplot_significant_features(data, mz_values, class_colors, test, 
                                        loc='inside', show_scatter=show_scatter, 
                                        use_log2=use_log2, capture_name=capture_name,
                                        significance_dict=significance_dict,
                                        pval_correction=pval_correction)
    elif plot_type == 'bar':
        barplot_significant_features(data, mz_values, class_colors, test, 
                                     loc='inside', show_scatter=show_scatter, 
                                     use_log2=use_log2, capture_name=capture_name,
                                     significance_dict=significance_dict,
                                     pval_correction=pval_correction)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    




import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, to_hex
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# WIDGET DE SÉLECTION DE BRANCHE — dendrogramme des FEATURES (vertical, gauche)
# et dendrogramme des ÉCHANTILLONS (horizontal, haut).
# Les deux panneaux partagent les mêmes helpers de rendu et d'export.
# ─────────────────────────────────────────────────────────────────────────────

def _branch_options(dend, link, labels):
    """
    Liste des nœuds internes du dendrogramme :
    [{idx, leaves, names, height, center}, …] + icoord/dcoord normalisés.
    """
    icoord = np.asarray(dend["icoord"], dtype=float)
    dcoord = np.asarray(dend["dcoord"], dtype=float)
    n = len(labels)
    opts = []
    for k in range(len(icoord)):
        leaves = _leaves_of_subtree(link, n + k, n)
        opts.append({
            "idx": k,
            "leaves": leaves,
            "names": [labels[l] for l in leaves],
            "height": float(dcoord[k][1]),
            "center": (float(icoord[k][1]) + float(icoord[k][2])) / 2.0,
        })
    return opts, icoord, dcoord


def _branch_dend_figure(opts, icoord, dcoord, labels, orientation,
                        title, line_color, unit_label="items"):
    """
    Dendrogramme cliquable/survolable — 2 traces seulement
    (1 pour toutes les branches, 1 pour tous les nœuds).
    orientation="left" → arbre vertical (features, racine à gauche)
    orientation="top"  → arbre horizontal (échantillons, racine en haut)
    """
    n_leaves = len(labels)
    leaf = (icoord - 5.0) / 10.0
    nan_col = np.full((leaf.shape[0], 1), np.nan)
    a = np.hstack([leaf, nan_col]).ravel()
    b = np.hstack([dcoord, nan_col]).ravel()

    fig = go.Figure()
    if orientation == "left":
        fig.add_trace(go.Scatter(x=-b, y=a, mode="lines",
                                 line=dict(color=line_color, width=1.6),
                                 hoverinfo="skip", showlegend=False,
                                 connectgaps=False))
    else:
        fig.add_trace(go.Scatter(x=a, y=b, mode="lines",
                                 line=dict(color=line_color, width=1.6),
                                 hoverinfo="skip", showlegend=False,
                                 connectgaps=False))

    node_pos, node_h, node_hover = [], [], []
    for o in opts:
        node_pos.append((o["center"] - 5.0) / 10.0)
        node_h.append(o["height"])
        preview = ", ".join(o["names"][:5])
        if len(o["names"]) > 5:
            preview += f" … (+{len(o['names']) - 5})"
        node_hover.append(f"<b>{len(o['names'])} {unit_label}</b><br>{preview}")

    if orientation == "left":
        nx, ny = [-h for h in node_h], node_pos
    else:
        nx, ny = node_pos, node_h

    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers",
        marker=dict(size=9, color="#e05f2e", symbol="circle",
                    line=dict(color="white", width=1.2)),
        hovertext=node_hover, hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    leaf_pos = [i for i in range(n_leaves)]
    max_h = float(dcoord.max()) if dcoord.size else 1.0
    show_labels = n_leaves <= 120

    axis_dist = dict(title="Ward distance", showgrid=True, gridcolor="#eee",
                     zeroline=False, tickfont=dict(size=9, family="Arial"))
    axis_leaf = dict(tickmode="array", tickvals=leaf_pos,
                     ticktext=labels if show_labels else [""] * n_leaves,
                     tickfont=dict(size=9, color="#111", family="Arial"),
                     showticklabels=show_labels, showgrid=False, zeroline=False)

    if orientation == "left":
        axis_dist.update(range=[-max_h * 1.05, 0],
                         tickvals=[-v for v in np.linspace(0, max_h, 6)],
                         ticktext=[f"{v:.1f}" for v in np.linspace(0, max_h, 6)])
        axis_leaf.update(side="right")
        height = max(300, min(900, n_leaves * 16 + 90))
        layout_axes = dict(xaxis=axis_dist, yaxis=axis_leaf)
        margin = dict(l=20, r=150, t=40, b=45)
    else:
        axis_dist.update(range=[0, max_h * 1.05])
        axis_leaf.update(tickangle=90)
        height = max(300, min(700, 260 + (90 if show_labels else 0)))
        layout_axes = dict(xaxis=axis_leaf, yaxis=axis_dist)
        margin = dict(l=55, r=20, t=40, b=150 if show_labels else 40)

    fig.update_layout(
        height=height, margin=margin,
        paper_bgcolor="white", plot_bgcolor="white",
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=13, color=line_color, family="Arial"), x=0.5),
        hovermode="closest", font=dict(family="Arial", color="#111"),
        **layout_axes,
    )
    return fig


def _branch_selectbox(opts, unit_label, key, max_options=400):
    """Selectbox des branches, triées par taille décroissante."""
    entries = sorted(opts, key=lambda o: -len(o["names"]))[:max_options]
    labels_opt = []
    for o in entries:
        preview = ", ".join(o["names"][:4])
        if len(o["names"]) > 4:
            preview += f" … +{len(o['names']) - 4}"
        labels_opt.append(f"{len(o['names'])} {unit_label} — {preview}")

    choice = st.selectbox(
        f"🌿 Select a branch ({unit_label})",
        options=["— Select a branch —"] + labels_opt,
        index=0, key=key,
        help="Each entry is an internal node of the dendrogram, "
             "sorted by branch size (largest first).",
    )
    if choice == "— Select a branch —":
        return None
    return entries[labels_opt.index(choice)]


# ─────────────────────────────────────────────────────────────────────────────
# Panneau 1 — branche de FEATURES (dendrogramme vertical)
# ─────────────────────────────────────────────────────────────────────────────

def _feature_branch_panel(dd, capture_name):
    row_dend = dd["row_dend"]
    row_link = dd["row_link"]
    feature_labels_ord = dd["feature_labels_ord"]
    matrix_ord = dd["matrix_ord"]
    sample_labels_ord = dd["sample_labels_ord"]
    class_labels_ord = dd["class_labels_ord"]
    data_original = dd["data_original"]

    st.caption("Select a branch to retrieve the up/down-regulated features it "
               "contains, and download the corresponding statistics.")

    opts, icoord, dcoord = _branch_options(row_dend, row_link, feature_labels_ord)
    fig = _branch_dend_figure(opts, icoord, dcoord, feature_labels_ord,
                              orientation="left",
                              title="Feature dendrogram (vertical)",
                              line_color="#2c5f8a", unit_label="features")
    st.plotly_chart(fig, use_container_width=True, key=f"{capture_name}_dend_viz_feat",
                    config={"displayModeBar": True, "displaylogo": False,
                            "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "scale": 3}})

    sel = _branch_selectbox(opts, "features", f"{capture_name}_branch_feat")
    if sel is None:
        return

    branch_features = sel["names"]
    n_branch = len(branch_features)
    st.success(f"✅ **{n_branch} feature(s)** selected in this branch")

    with st.expander(f"📋 {n_branch} selected features", expanded=False):
        st.dataframe(pd.DataFrame({"Feature": branch_features}),
                     use_container_width=True)

    # Over / under-expression per class (vectorised)
    unique_classes = list(dict.fromkeys(class_labels_ord))
    cls_arr = np.asarray(class_labels_ord)
    idx_feat = sel["leaves"]
    sub = matrix_ord[:, idx_feat]                       # (n_samples, n_branch)

    means = np.vstack([np.nanmean(sub[cls_arr == c], axis=0) for c in unique_classes])
    global_mean = np.nanmean(means, axis=0)
    delta = means - global_mean

    rows = []
    for ci, cls in enumerate(unique_classes):
        for fi, feat in enumerate(branch_features):
            rows.append({"Feature": feat, "Class": cls,
                         "Mean_Zscore": round(float(means[ci, fi]), 4),
                         "Delta_vs_global": round(float(delta[ci, fi]), 4)})
    df_all = pd.DataFrame(rows)
    df_over = df_all[df_all.Delta_vs_global > 0].sort_values(
        ["Class", "Delta_vs_global"], ascending=[True, False])
    df_under = df_all[df_all.Delta_vs_global <= 0].sort_values(
        ["Class", "Delta_vs_global"], ascending=[True, True])

    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            xls = _export_branch_excel(
                branch_features=branch_features, data_original=data_original,
                feature_labels_ord=feature_labels_ord, matrix_z=matrix_ord,
                sample_labels_ord=sample_labels_ord,
                class_labels_ord=class_labels_ord)
            st.download_button(f"📥 Full Excel ({n_branch} features)", data=xls,
                               file_name=f"branch_{n_branch}features.xlsx",
                               mime="application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet",
                               key=f"{capture_name}_feat_excel", use_container_width=True)
        except Exception as e:
            st.error(f"Excel export error: {e}")
    with c2:
        if not df_over.empty:
            st.download_button(f"📥 CSV Overexpressed ({len(df_over)})",
                               data=df_over.to_csv(index=False).encode("utf-8"),
                               file_name=f"branch_overexpressed_{n_branch}.csv",
                               mime="text/csv", key=f"{capture_name}_feat_over",
                               use_container_width=True)
            with st.expander("👆 Overexpressed — preview", expanded=False):
                st.dataframe(df_over, use_container_width=True)
        else:
            st.caption("No overexpressed features in this branch.")
    with c3:
        if not df_under.empty:
            st.download_button(f"📥 CSV Underexpressed ({len(df_under)})",
                               data=df_under.to_csv(index=False).encode("utf-8"),
                               file_name=f"branch_underexpressed_{n_branch}.csv",
                               mime="text/csv", key=f"{capture_name}_feat_under",
                               use_container_width=True)
            with st.expander("👇 Underexpressed — preview", expanded=False):
                st.dataframe(df_under, use_container_width=True)
        else:
            st.caption("No underexpressed features in this branch.")


# ─────────────────────────────────────────────────────────────────────────────
# Panneau 2 — branche d'ÉCHANTILLONS (dendrogramme horizontal)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_branch_panel(dd, capture_name):
    col_dend = dd.get("col_dend")
    col_link = dd.get("col_link")
    if col_dend is None or col_link is None:
        st.info("Re-generate the heatmap to enable sample-branch selection.")
        return

    sample_labels_ord = dd["sample_labels_ord"]
    class_labels_ord = dd["class_labels_ord"]
    feature_labels_ord = dd["feature_labels_ord"]
    matrix_ord = dd["matrix_ord"]
    data_original = dd["data_original"]

    st.caption("Select a branch of the sample dendrogram to inspect a sample "
               "cluster: class composition, features driving the cluster, "
               "and exports.")

    opts, icoord, dcoord = _branch_options(col_dend, col_link, sample_labels_ord)
    fig = _branch_dend_figure(opts, icoord, dcoord, sample_labels_ord,
                              orientation="top",
                              title="Sample dendrogram (horizontal)",
                              line_color="#7a4e9e", unit_label="samples")
    st.plotly_chart(fig, use_container_width=True, key=f"{capture_name}_dend_viz_samp",
                    config={"displayModeBar": True, "displaylogo": False,
                            "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "scale": 3}})

    sel = _branch_selectbox(opts, "samples", f"{capture_name}_branch_samp")
    if sel is None:
        return

    idx_in = np.asarray(sel["leaves"], dtype=int)
    n_branch = len(idx_in)
    mask = np.zeros(matrix_ord.shape[0], dtype=bool)
    mask[idx_in] = True
    st.success(f"✅ **{n_branch} sample(s)** selected in this branch")

    # Class composition
    cls_arr = np.asarray(class_labels_ord)
    comp = (pd.Series(cls_arr[mask]).value_counts()
            .rename_axis("Class").reset_index(name="n"))
    comp["% of branch"] = (comp["n"] / n_branch * 100).round(1)
    total_per_cls = pd.Series(cls_arr).value_counts()
    comp["% of class captured"] = comp.apply(
        lambda r: round(r["n"] / total_per_cls[r["Class"]] * 100, 1), axis=1)

    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("**Class composition**")
        st.dataframe(comp, use_container_width=True, hide_index=True)
    with c_right:
        st.markdown("**Samples in this branch**")
        st.dataframe(pd.DataFrame({"Sample": [sample_labels_ord[i] for i in idx_in],
                                   "Class": cls_arr[mask]}),
                     use_container_width=True, hide_index=True, height=220)

    # Features driving the cluster: mean z inside vs outside
    inside = np.nanmean(matrix_ord[mask], axis=0)
    outside = (np.nanmean(matrix_ord[~mask], axis=0)
               if (~mask).any() else np.zeros_like(inside))
    diff = inside - outside
    df_drv = pd.DataFrame({
        "Feature": feature_labels_ord,
        "Mean_Z_branch": np.round(inside, 4),
        "Mean_Z_rest": np.round(outside, 4),
        "Delta": np.round(diff, 4),
    }).sort_values("Delta", ascending=False)

    top_n = min(20, len(df_drv))
    top = pd.concat([df_drv.head(top_n // 2), df_drv.tail(top_n // 2)])
    fig_drv = go.Figure(go.Bar(
        x=top["Delta"][::-1], y=top["Feature"][::-1], orientation="h",
        marker=dict(color=["#b2182b" if v > 0 else "#2166ac"
                           for v in top["Delta"][::-1]]),
        hovertemplate="%{y}<br>Δz = %{x:.3f}<extra></extra>",
    ))
    fig_drv.update_layout(
        height=max(280, 22 * len(top) + 80),
        margin=dict(l=10, r=20, t=36, b=36),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial", size=10, color="#111"),
        title=dict(text="<b>Features driving this sample cluster (Δ z-score vs rest)</b>",
                   font=dict(size=12, family="Arial"), x=0.5),
        xaxis=dict(title="Δ z-score", zeroline=True, zerolinecolor="#999",
                   showgrid=True, gridcolor="#eee"),
        yaxis=dict(automargin=True, showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig_drv, use_container_width=True,
                    key=f"{capture_name}_samp_drivers",
                    config={"displaylogo": False,
                            "toImageButtonOptions": {"format": "png", "scale": 3}})

    # Exports
    c1, c2 = st.columns(2)
    with c1:
        try:
            xls = _export_sample_branch_excel(
                idx_in=idx_in, data_original=data_original,
                matrix_z=matrix_ord, feature_labels_ord=feature_labels_ord,
                sample_labels_ord=sample_labels_ord,
                class_labels_ord=class_labels_ord, df_drivers=df_drv)
            st.download_button(f"📥 Full Excel ({n_branch} samples)", data=xls,
                               file_name=f"sample_branch_{n_branch}samples.xlsx",
                               mime="application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet",
                               key=f"{capture_name}_samp_excel", use_container_width=True)
        except Exception as e:
            st.error(f"Excel export error: {e}")
    with c2:
        st.download_button(f"📥 CSV drivers ({len(df_drv)} features)",
                           data=df_drv.to_csv(index=False).encode("utf-8"),
                           file_name=f"sample_branch_{n_branch}samples_drivers.csv",
                           mime="text/csv", key=f"{capture_name}_samp_drv",
                           use_container_width=True)


def _export_sample_branch_excel(idx_in, data_original, matrix_z, feature_labels_ord,
                                sample_labels_ord, class_labels_ord, df_drivers):
    """Excel d'une branche d'échantillons : membres, drivers, z-scores, données brutes."""
    idx_in = np.asarray(idx_in, dtype=int)
    members = pd.DataFrame({"Sample": [sample_labels_ord[i] for i in idx_in],
                            "Class": [class_labels_ord[i] for i in idx_in]})

    z_df = pd.DataFrame(matrix_z[idx_in], columns=feature_labels_ord)
    z_df.insert(0, "Class", members["Class"].values)
    z_df.insert(0, "Sample", members["Sample"].values)

    if isinstance(data_original, np.ndarray):
        raw_df = pd.DataFrame(data_original[idx_in], columns=feature_labels_ord)
        raw_df.insert(0, "Class", members["Class"].values)
        raw_df.insert(0, "Sample", members["Sample"].values)
    else:
        raw_df = pd.DataFrame()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        members.to_excel(writer, sheet_name="Branch_Samples", index=False)
        df_drivers.to_excel(writer, sheet_name="Branch_Drivers", index=False)
        z_df.to_excel(writer, sheet_name="Branch_Zscores", index=False)
        if not raw_df.empty:
            raw_df.to_excel(writer, sheet_name="Branch_RawData", index=False)
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                vals = [len(str(c.value)) for c in col[:8] if c.value is not None]
                sheet.column_dimensions[col[0].column_letter].width = \
                    min(max(vals or [10]) + 4, 40)
    buf.seek(0)
    return buf.read()


def render_heatmap_dendrogram_widget(capture_name: str = "heatmap_fig"):
    """
    Widget persistant de sélection de branche, sur les DEUX dendrogrammes :
      • onglet 1 — features (dendrogramme vertical, axe gauche du heatmap)
      • onglet 2 — échantillons (dendrogramme horizontal, axe haut du heatmap)
    À appeler depuis la GUI dans le bloc show_heatmap, en dehors de
    plot_heatmap_samples. Lit les données stockées en session_state.
    """
    dd = st.session_state.get(f"{capture_name}_dend_data")
    if dd is None:
        return

    st.markdown("---")
    st.markdown("🌿 Dendrogram branch explorer")

    tab_feat, tab_samp = st.tabs(["Features — vertical dendrogram",
                                  "Samples — horizontal dendrogram"])
    with tab_feat:
        _feature_branch_panel(dd, capture_name)
    with tab_samp:
        _sample_branch_panel(dd, capture_name)



# ─────────────────────────────────────────────────────────────────────────────
# Exact replica of the original seaborn static output
# ─────────────────────────────────────────────────────────────────────────────


def _build_static_png_fast(
    matrix_ord, raw_ord,
    feature_labels_ord, sample_labels_ord, class_labels_ord,
    class_colors, custom_colors,
    data_meta_orig=None, meta_cols=None, col_order=None,
):
    """
    Fast matplotlib PNG from already-ordered numpy arrays.
    Works directly on pre-computed matrix_ord (Z-scores) and raw_ord.
    Avoids re-running clustering or copying large DataFrames.
    """
    import io
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    meta_cols = meta_cols or []
    n_samples, n_features = matrix_ord.shape

    cmap = LinearSegmentedColormap.from_list("custom", custom_colors, N=256)
    vmin, vmax = float(matrix_ord.min()), float(matrix_ord.max())

    # Cap cell size to keep figure manageable
    _cell = max(0.12, min(0.28, 800 / max(n_samples, n_features) / 72))
    fig_w = max(10, min(28, n_samples * _cell + 4))
    fig_h = max(8,  min(24, n_features * _cell + 2 + len(meta_cols) * 0.3))

    _n_ann = 1 + len(meta_cols)   # Class + meta strips
    _h_ratios = [0.08] * _n_ann + [1.0]
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs  = gridspec.GridSpec(
        _n_ann + 1, 1,
        height_ratios=_h_ratios,
        hspace=0.01,
        figure=fig,
    )

    _AUTO_PAL = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
                 "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

    legend_handles = []

    # ── Annotation strips ─────────────────────────────────────────────────────
    for ann_idx, ann_label in enumerate(["Class"] + meta_cols):
        ax_ann = fig.add_subplot(gs[ann_idx])

        if ann_label == "Class":
            uniq_cls = list(dict.fromkeys(class_labels_ord))
            cls_idx  = {c: i for i, c in enumerate(uniq_cls)}
            cls_cs   = LinearSegmentedColormap.from_list(
                "cls", [class_colors.get(c, "#aaa") for c in uniq_cls],
                N=len(uniq_cls)
            )
            z_cls = np.array([[cls_idx[c] for c in class_labels_ord]])
            ax_ann.imshow(z_cls, aspect="auto", interpolation="none",
                          cmap=cls_cs, vmin=0, vmax=len(uniq_cls))
            ax_ann.set_yticks([0]); ax_ann.set_yticklabels(["Class"], fontsize=7)
            legend_handles += [
                mpatches.Patch(color=class_colors.get(c, "#aaa"), label=f"Class: {c}")
                for c in uniq_cls
            ]
        else:
            if data_meta_orig is not None and ann_label in data_meta_orig.columns:
                raw_vals = data_meta_orig[ann_label].values
                if col_order is not None:
                    raw_vals = raw_vals[col_order]
            else:
                raw_vals = np.zeros(n_samples)

            if np.issubdtype(raw_vals.dtype, np.number):
                _vmin_m = float(np.nanmin(raw_vals))
                _vmax_m = float(np.nanmax(raw_vals))
                import matplotlib.colors as _mc
                _norm = _mc.Normalize(vmin=_vmin_m, vmax=_vmax_m)
                _cmap_m = plt.cm.get_cmap("viridis")
                z_m = np.array([[float(v) if not np.isnan(v) else _vmin_m for v in raw_vals]])
                ax_ann.imshow(z_m, aspect="auto", interpolation="none",
                              cmap=_cmap_m, vmin=_vmin_m, vmax=_vmax_m)
                legend_handles += [
                    mpatches.Patch(color=_mc.to_hex(_cmap_m(t)),
                                   label=f"{ann_label}: {_vmin_m + t*(_vmax_m-_vmin_m):.2g}")
                    for t in [0.0, 0.5, 1.0]
                ]
            else:
                uniq_v = list(dict.fromkeys(str(v) for v in raw_vals))
                v_map  = {v: i for i, v in enumerate(uniq_v)}
                palette_cs = LinearSegmentedColormap.from_list(
                    "meta", [_AUTO_PAL[i % len(_AUTO_PAL)] for i in range(len(uniq_v))],
                    N=len(uniq_v)
                )
                z_v = np.array([[v_map.get(str(v), 0) for v in raw_vals]])
                ax_ann.imshow(z_v, aspect="auto", interpolation="none",
                              cmap=palette_cs, vmin=0, vmax=len(uniq_v))
                legend_handles += [
                    mpatches.Patch(color=_AUTO_PAL[i % len(_AUTO_PAL)], label=f"{ann_label}: {v}")
                    for i, v in enumerate(uniq_v)
                ]
                ax_ann.set_yticks([0]); ax_ann.set_yticklabels([ann_label], fontsize=7)

        ax_ann.set_xticks([]); ax_ann.set_yticks([0])
        if ann_label == "Class":
            ax_ann.set_yticklabels(["Class"], fontsize=7)
        else:
            ax_ann.set_yticklabels([ann_label], fontsize=7)

    # ── Main heatmap ──────────────────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[_n_ann])
    im = ax_hm.imshow(
        matrix_ord.T,    # (n_features, n_samples)
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        origin="lower",
    )
    plt.colorbar(im, ax=ax_hm, fraction=0.015, pad=0.01, label="Z-score")

    # Feature labels (Y-axis)
    _show_feat_ticks = n_features <= 200
    if _show_feat_ticks:
        ax_hm.set_yticks(range(n_features))
        ax_hm.set_yticklabels(feature_labels_ord, fontsize=max(4, min(9, 180 // n_features)))
    else:
        ax_hm.set_yticks([])

    # Sample labels (X-axis)
    _show_samp_ticks = n_samples <= 200
    if _show_samp_ticks:
        ax_hm.set_xticks(range(n_samples))
        ax_hm.set_xticklabels(sample_labels_ord, rotation=90,
                               fontsize=max(4, min(8, 150 // n_samples)))
    else:
        ax_hm.set_xticks([])
        ax_hm.set_xlabel(f"{n_samples} samples", fontsize=9)

    # Legend
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right",
                   bbox_to_anchor=(1.0, 1.0), fontsize=7,
                   framealpha=0.9, ncol=1)

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()



# ─────────────────────────────────────────────────────────────────────────────
# HELPER : extraire les feuilles d'un sous-arbre à partir d'un nœud cliqué
# ─────────────────────────────────────────────────────────────────────────────

def _leaves_of_subtree(Z, node_id, n_leaves):
    """
    Retourne la liste des indices feuilles du sous-arbre raciné à `node_id`.
    Z   : linkage matrix (N-1, 4)
    node_id : entier interne scipy (feuille 0..n-1, nœud n..2n-2)
    n_leaves : nombre total de feuilles
    """
    if node_id < n_leaves:
        return [node_id]
    stack = [int(node_id)]
    leaves = []
    while stack:
        nid = stack.pop()
        if nid < n_leaves:
            leaves.append(nid)
        else:
            row = Z[nid - n_leaves]
            stack.append(int(row[0]))
            stack.append(int(row[1]))
    return sorted(leaves)




# ─────────────────────────────────────────────────────────────────────────────
# HELPER : export Excel d'une branche (style Perseus)
# ─────────────────────────────────────────────────────────────────────────────

def _export_branch_excel(
    branch_features: list,
    data_original,          # np.ndarray (n_samples, n_features) in dend order, OR pd.DataFrame (legacy)
    feature_labels_ord: list,
    matrix_z: np.ndarray,
    sample_labels_ord: list,
    class_labels_ord: list,
):
    """
    Generate Excel with:
      - Sheet 1 "Branch_Summary" : mean ± SD Z-score per class per feature
      - Sheet 2 "Branch_Zscores": individual Z-scores per sample
      - Sheet 3 "Branch_RawData": original (unscaled) values
    Accepts data_original as numpy array (ordered) or legacy DataFrame.
    """
    branch_idx    = [i for i, f in enumerate(feature_labels_ord) if f in set(branch_features)]
    feats_ordered = [feature_labels_ord[i] for i in branch_idx]
    unique_classes = list(dict.fromkeys(class_labels_ord))

    # ── Sheet 1: Summary per class ────────────────────────────────────────────
    summary_rows = []
    for feat in feats_ordered:
        row = {"Feature": feat}
        feat_col_idx = feature_labels_ord.index(feat)
        for cls in unique_classes:
            cls_mask = [i for i, c in enumerate(class_labels_ord) if c == cls]
            if cls_mask:
                vals = matrix_z[np.ix_(cls_mask, [feat_col_idx])][:, 0]
                row[f"{cls}_mean_zscore"] = round(float(np.nanmean(vals)), 4)
                row[f"{cls}_sd_zscore"]   = round(float(np.nanstd(vals)),  4)
            else:
                row[f"{cls}_mean_zscore"] = np.nan
                row[f"{cls}_sd_zscore"]   = np.nan
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)

    # ── Sheet 2: Individual Z-scores ──────────────────────────────────────────
    feat_indices = [feature_labels_ord.index(f) for f in feats_ordered]
    z_branch = matrix_z[:, feat_indices]   # (n_samples, n_branch)
    df_raw = pd.DataFrame(z_branch, columns=feats_ordered)
    df_raw.insert(0, "Class",  class_labels_ord)
    df_raw.insert(0, "Sample", sample_labels_ord)

    # ── Sheet 3: Raw (unscaled) values ────────────────────────────────────────
    if isinstance(data_original, np.ndarray):
        # data_original is already ordered (n_samples, n_features) in dend order
        raw_branch = data_original[:, feat_indices]
        df_orig = pd.DataFrame(raw_branch, columns=feats_ordered)
        df_orig.insert(0, "Class",  class_labels_ord)
        df_orig.insert(0, "Sample", sample_labels_ord)
    else:
        # Legacy: full DataFrame
        orig_cols = [f for f in branch_features if f in data_original.columns]
        meta_cols = [c for c in ["ID", "Class", "File"] if c in data_original.columns]
        df_orig = data_original[meta_cols + orig_cols].copy()

    # ── Write Excel ───────────────────────────────────────────────────────────
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Branch_Summary",  index=False)
        df_raw.to_excel(writer,     sheet_name="Branch_Zscores",  index=False)
        df_orig.to_excel(writer,    sheet_name="Branch_RawData",  index=False)

        # Mise en forme basique (largeur colonnes)
        for sheet_name in ["Branch_Summary", "Branch_Zscores", "Branch_RawData"]:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(
                    len(str(col[0].value)) if col[0].value else 0,
                    *[len(str(c.value)) if c.value else 0 for c in col[1:6]]
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    buf.seek(0)
    return buf.read()





# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILER — HEATMAP "PUBLI-READY"  (drop-in replacement for plot_heatmap_samples)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  INSTALLATION
#  ------------
#  Coller ce bloc *à la fin* de profiler_features_importance.py.
#  Python garde la dernière définition d'un nom : cette version remplace donc
#  automatiquement les 3 anciennes `plot_heatmap_samples` (lignes ~1291, ~1824,
#  ~3024), pour Profiler.py ET Profiler_Desktop_Gui.py, sans toucher aux appels.
#  (Nettoyage recommandé ensuite : supprimer les 3 anciennes définitions.)
#
#  CE QUI CHANGE — POURQUOI C'EST RAPIDE
#  -------------------------------------
#   • Dendrogrammes : 1 SEULE trace chacun (segments séparés par None) au lieu
#     d'une go.Scatter par branche  →  ~2 traces au lieu de 2×(n-1).
#   • Bandes d'annotation : 1 SEUL go.Heatmap (n_ann × n_samples) avec une
#     colorscale discrète globale, au lieu d'un subplot + 1 trace par annotation.
#   • Grille : 3×2 fixe au lieu de (2+n_ann)×3  →  moins d'axes à monter.
#   • Métadonnées numériques : binnées en 9 paliers au lieu de n_samples paliers
#     (l'ancienne version construisait une colorscale de 2×n_samples points !).
#   • Hover : tableau de chaînes construit seulement si n_cells ≤ HOVER_MAX_CELLS,
#     sinon template léger — plus de payload JSON de plusieurs Mo.
#   • Légende plafonnée (MAX_LEGEND_PER_GROUP) : plus de 60 entrées fantômes.
#
#  CE QUI CHANGE — POURQUOI C'EST BEAU
#  -----------------------------------
#   • Colorscale divergente RdBu_r centrée sur 0 (les z-scores sont signés) et
#     bornes robustes (percentile 2/98 symétrisé) : les outliers n'écrasent plus
#     le contraste.  `diverging=False` restaure l'ancien comportement.
#   • Dendrogrammes gris fins, non intrusifs.
#   • Bande d'annotation compacte collée au heatmap, légende groupée à droite,
#     colorbar en bas à droite — jamais de recouvrement.
#   • Typo Arial homogène, pas de gras partout, fond blanc, cadre fin.
#   • Export PNG vectorisé-quality : scale 3 (~300 dpi) + SVG dans la toolbar.
# ═══════════════════════════════════════════════════════════════════════════════

import gc as _gc
import numpy as _np
import pandas as _pd
import plotly.graph_objects as _go
from plotly.subplots import make_subplots as _make_subplots
import matplotlib.colors as _mcolors

try:
    import streamlit as _st
except Exception:                                    # pragma: no cover
    _st = None


# ── Réglages ────────────────────────────────────────────────────────────────
HOVER_MAX_CELLS      = 60_000   # au-delà : hover léger (z seul)
MAX_LEGEND_PER_GROUP = 12       # entrées de légende max par annotation
N_META_BINS          = 9        # paliers pour une annotation numérique
_EPS                 = 1e-9

_AUTO_PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
             "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
_NUM_CMAPS = ["viridis", "cividis", "magma", "YlGnBu", "PuRd"]
_NA_COLOR = "#d9d9d9"


def _get_cmap(name):
    """matplotlib >= 3.9 compatible."""
    try:
        import matplotlib as _mpl
        return _mpl.colormaps[name]
    except Exception:                                # pragma: no cover
        import matplotlib.pyplot as _plt
        return _plt.cm.get_cmap(name)


# ── 1. Dendrogramme → UNE seule trace ───────────────────────────────────────
def _dend_single_trace(dend, orientation="top", color="#6b6b6b", width=1.0):
    """
    Tous les segments du dendrogramme dans une seule go.Scatter, séparés par
    None (rupture de ligne). Visuellement identique, ~N fois moins lourd.
    """
    ic = _np.asarray(dend["icoord"], dtype=float)
    dc = _np.asarray(dend["dcoord"], dtype=float)
    if ic.size == 0:
        return _go.Scatter(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False)

    leaf = (ic - 5.0) / 10.0                      # scipy place les feuilles à 5,15,25…
    n = leaf.shape[0]
    nan_col = _np.full((n, 1), _np.nan)

    a = _np.hstack([leaf, nan_col]).ravel()       # positions "feuille"
    b = _np.hstack([dc,   nan_col]).ravel()       # hauteurs

    if orientation == "top":
        x, y = a, b
    else:                                          # "left" : arbre ouvert vers le heatmap
        x, y = -b, a

    return _go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=width, shape="linear"),
        hoverinfo="skip", showlegend=False, connectgaps=False,
    )


# ── 2. Colorscales ──────────────────────────────────────────────────────────
def _discrete_colorscale(colors):
    """Colorscale en paliers nets : z = idx + 0.5, zmin=0, zmax=len(colors)."""
    k = len(colors)
    if k == 0:
        return [[0.0, _NA_COLOR], [1.0, _NA_COLOR]]
    cs = []
    for i, c in enumerate(colors):
        t0, t1 = i / k, (i + 1) / k
        cs += [[t0, c], [max(t1 - _EPS, t0), c]]
    cs[-1][0] = 1.0
    return cs


def _diverging_colorscale(name="RdBu_r", n=64):
    cm = _get_cmap(name)
    return [[i / (n - 1), _mcolors.to_hex(cm(i / (n - 1)))] for i in range(n)]


def _continuous_colorscale(custom_colors, n=64):
    cm = _mcolors.LinearSegmentedColormap.from_list("cc", custom_colors, N=256)
    return [[i / (n - 1), _mcolors.to_hex(cm(i / (n - 1)))] for i in range(n)]


def _robust_sym_limits(mat, lo=2.0, hi=98.0):
    """Bornes symétriques robustes → 0 reste au centre de la RdBu."""
    v = _np.nanpercentile(mat, [lo, hi])
    m = float(max(abs(v[0]), abs(v[1])))
    if not _np.isfinite(m) or m == 0:
        m = float(_np.nanmax(_np.abs(mat))) or 1.0
    return -m, m


# ── 3. Bande d'annotation : UN seul heatmap pour toutes les lignes ──────────
def _build_annotation_band(ann_labels, ann_values, class_colors, n_samples):
    """
    Retourne (trace, legend_traces, y_labels).
    z : (n_ann, n_samples) d'indices globaux → une colorscale discrète unique.
    """
    palette, z_rows, hover_rows, legend_traces = [], [], [], []
    num_cmap_i = 0

    for label, vals in zip(ann_labels, ann_values):
        vals = list(vals)
        is_num = False
        if label != "Class":
            try:
                is_num = _pd.api.types.is_numeric_dtype(_pd.Series(vals))
            except Exception:
                is_num = False

        if label == "Class":
            uniq = list(dict.fromkeys([str(v) for v in vals]))
            base = len(palette)
            palette += [class_colors.get(u, "#aaaaaa") for u in uniq]
            idx = {u: base + i for i, u in enumerate(uniq)}
            z_rows.append([idx[str(v)] + 0.5 for v in vals])
            hover_rows.append([f"Class: {v}" for v in vals])
            for i, u in enumerate(uniq[:MAX_LEGEND_PER_GROUP]):
                legend_traces.append(_legend_item(
                    u, class_colors.get(u, "#aaaaaa"), "cls_group", "Class", i == 0))

        elif is_num:
            arr = _np.array([float(v) if (v is not None and v == v) else _np.nan
                             for v in vals], dtype=float)
            cmap = _get_cmap(_NUM_CMAPS[num_cmap_i % len(_NUM_CMAPS)])
            num_cmap_i += 1
            finite = arr[_np.isfinite(arr)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0
            rng = (vmax - vmin) or 1.0

            base = len(palette)
            bin_colors = [_mcolors.to_hex(cmap(i / (N_META_BINS - 1)))
                          for i in range(N_META_BINS)]
            palette += bin_colors
            b = _np.clip(((arr - vmin) / rng * (N_META_BINS - 1)).round(), 0, N_META_BINS - 1)
            z_rows.append([(base + int(bi) + 0.5) if _np.isfinite(a) else None
                           for bi, a in zip(_np.nan_to_num(b), arr)])
            hover_rows.append([f"{label}: {a:.3g}" if _np.isfinite(a) else f"{label}: N/A"
                               for a in arr])
            for i, (txt, t) in enumerate([(f"{vmin:.3g}", 0.0),
                                          (f"{(vmin+vmax)/2:.3g}", 0.5),
                                          (f"{vmax:.3g}", 1.0)]):
                legend_traces.append(_legend_item(
                    txt, _mcolors.to_hex(cmap(t)), f"meta_{label}", label, i == 0))

        else:
            svals = [None if (v is None or v != v) else str(v) for v in vals]
            uniq = sorted({s for s in svals if s is not None})
            base = len(palette)
            palette += [_AUTO_PAL[i % len(_AUTO_PAL)] for i in range(len(uniq))]
            idx = {u: base + i for i, u in enumerate(uniq)}
            z_rows.append([(idx[s] + 0.5) if s is not None else None for s in svals])
            hover_rows.append([f"{label}: {s if s is not None else 'N/A'}" for s in svals])
            for i, u in enumerate(uniq[:MAX_LEGEND_PER_GROUP]):
                legend_traces.append(_legend_item(
                    u, _AUTO_PAL[i % len(_AUTO_PAL)], f"meta_{label}", label, i == 0))
            if len(uniq) > MAX_LEGEND_PER_GROUP:
                legend_traces.append(_legend_item(
                    f"… +{len(uniq) - MAX_LEGEND_PER_GROUP}", "#ffffff",
                    f"meta_{label}", label, False))

    k = max(len(palette), 1)
    # Ligne du bas = première annotation (Class) → on inverse pour l'affichage
    trace = _go.Heatmap(
        z=z_rows[::-1],
        x=list(range(n_samples)),
        y=ann_labels[::-1],
        colorscale=_discrete_colorscale(palette),
        zmin=0, zmax=k, showscale=False,
        customdata=hover_rows[::-1],
        hovertemplate="%{customdata}<extra></extra>",
        hoverongaps=False, xgap=0, ygap=1,
    )
    return trace, legend_traces, ann_labels[::-1]


def _legend_item(name, color, group, group_title, first):
    return _go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=9, color=color, symbol="square",
                    line=dict(width=0.5, color="#444")),
        name=str(name), legendgroup=group,
        legendgrouptitle=dict(text=f"<b>{group_title}</b>",
                              font=dict(size=11, family="Arial")) if first else {},
        showlegend=True, hoverinfo="skip",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════
def plot_heatmap_samples(
    data,
    class_colors: dict,
    selected_features: list,
    custom_colors: list,
    show_sample_names: bool = True,
    sample_label_col: str = "ID",
    show_feature_names: bool = True,
    caption: str = None,
    capture_name: str = None,
    meta_annotation_cols: list = None,
    *,
    diverging: bool = True,      # RdBu_r centrée sur 0 (recommandé pour z-scores)
    robust: bool = True,         # bornes percentile 2/98 symétrisées
    cell_border: bool = None,    # None = auto (bordures si < 80 lignes/colonnes)
    colorbar_pos: str = "top-left",   # "top-left" (style clustermap) ou "right"
    feature_label_side: str = "right",  # côté des noms de features
    fit_width: bool = False,     # True = s'étire au conteneur (px non garantis)
):
    """
    Heatmap hiérarchique publication-ready, rendu léger.
    Signature et contrats session_state identiques à l'ancienne version
    (capture_name, {capture_name}_dend_data, bouton PNG).
    """
    # ── 1. Préparation des données ──────────────────────────────────────────
    data.columns = data.columns.astype(str)
    selected_features = _resolve_features(data, [str(f) for f in selected_features])

    missing = [f for f in selected_features if f not in data.columns]
    if missing:
        _st.error(f"Invalid features: {', '.join(missing)}")
        return

    features = list(selected_features)
    meta_cols_present = [m for m in (meta_annotation_cols or []) if m in data.columns]
    keep = list(dict.fromkeys(["Class"] + features + meta_cols_present +
                              [c for c in ["ID", "File"] if c in data.columns]))
    df = data[keep].copy()

    data_meta_orig = df[meta_cols_present].copy() if meta_cols_present else None
    raw_features = df[features].values.copy()

    df[features] = df[features].replace([_np.inf, -_np.inf], _np.nan)
    df[features] = df[features].fillna(df[features].mean())

    matrix = StandardScaler().fit_transform(df[features].values).astype(_np.float32)
    if not _np.isfinite(matrix).all():
        _st.error("Missing/infinite values after preprocessing.")
        return

    label_col = sample_label_col if sample_label_col in df.columns else next(
        (c for c in ["ID", "File", "Class"] if c in df.columns), None)
    sample_labels = (df[label_col].astype(str).tolist() if label_col
                     else [f"sample_{i+1}" for i in range(len(df))])
    class_labels = df["Class"].astype(str).tolist()
    n_samples, n_features = matrix.shape
    large = (n_features > 300) or (n_samples > 500)

    # ── 2. Clustering ───────────────────────────────────────────────────────
    try:
        import fastcluster as _fc
        _ward = lambda X: _fc.linkage(X, method="ward", metric="euclidean")
    except ImportError:
        _ward = lambda X: linkage(X, method="ward", metric="euclidean")

    if n_features > 800:
        from sklearn.decomposition import TruncatedSVD
        mat_row = TruncatedSVD(n_components=min(50, max(n_samples - 1, 2)),
                               random_state=0).fit_transform(matrix.T)
    else:
        mat_row = matrix.T

    col_link = _ward(matrix)
    col_dend = dendrogram(col_link, no_plot=True)
    col_order = col_dend["leaves"]

    row_link = _ward(mat_row)
    row_dend = dendrogram(row_link, no_plot=True)
    row_order = row_dend["leaves"]

    matrix_ord = matrix[_np.ix_(col_order, row_order)]
    sample_labels_ord = [sample_labels[i] for i in col_order]
    class_labels_ord = [class_labels[i] for i in col_order]
    feature_labels_ord = [features[i] for i in row_order]
    export_raw_ord = raw_features[_np.ix_(col_order, row_order)]

    # ── 3. Figure : grille fixe 3×2 ─────────────────────────────────────────
    valid_meta = [m for m in (meta_annotation_cols or [])
                  if data_meta_orig is not None and m in data_meta_orig.columns]
    ann_labels = ["Class"] + valid_meta
    n_ann = len(ann_labels)

    dend_top_frac = 0.13 if n_samples > 2 else 0.04
    ann_frac = min(0.16, 0.030 * n_ann + 0.008)
    heat_frac = max(0.55, 1.0 - dend_top_frac - ann_frac)

    # ── Géométrie : on réserve explicitement la place des labels de features
    #    pour que colorbar et légende ne les chevauchent JAMAIS ────────────────
    _CHAR_PX = 5.4                       # largeur moyenne Arial 9 px
    feat_lab_px = 0
    if show_feature_names and feature_labels_ord:
        feat_lab_px = min(240, 10 + _CHAR_PX * max(len(str(f))
                                                   for f in feature_labels_ord))
    samp_lab_px = 0
    if show_sample_names and sample_labels_ord:
        samp_lab_px = min(180, 10 + _CHAR_PX * max(len(str(s))
                                                   for s in sample_labels_ord))

    legend_px = 0
    if n_ann:
        _leg_chars = max([len(str(a)) for a in ann_labels] + [10])
        legend_px = int(min(260, 46 + _CHAR_PX * _leg_chars * 1.6))
    cbar_px = 96 if colorbar_pos == "right" else 0

    px_f = max(9, min(18, 720 // max(n_features, 1)))
    px_s = max(9, min(18, 720 // max(n_samples, 1)))

    left_px = 8 + (feat_lab_px if feature_label_side == "left" else 0)
    right_px = 14 + (feat_lab_px if feature_label_side == "right" else 0) \
               + legend_px + cbar_px
    bottom_px = 12 + samp_lab_px
    top_px = 18

    height = int(max(520, min(1900, top_px + bottom_px + n_features * px_f
                              + n_ann * 22 + 120)))
    width = int(max(620, min(2400, left_px + right_px + n_samples * px_s + 120)))
    plot_w = max(width - left_px - right_px, 200)

    # décalage en coordonnées "paper" correspondant à la largeur des labels
    _lab_off = (feat_lab_px / plot_w) if feature_label_side == "right" else 0.0
    _pad = 0.012

    if colorbar_pos == "right":
        _cbar = dict(
            title=dict(text=f"<b>{{}}</b>", side="right",
                       font=dict(size=12, family="Arial")),
            thickness=12, len=0.28, orientation="v",
            x=1.0 + _lab_off + _pad, xanchor="left", y=0.0, yanchor="bottom",
            tickfont=dict(size=10, family="Arial"),
            outlinecolor="#333", outlinewidth=0.8, ticks="outside", ticklen=3,
        )
        _legend_x = 1.0 + _lab_off + _pad + (cbar_px / plot_w)
    else:
        # colorbar horizontale en haut à gauche (style seaborn clustermap) :
        # aucune collision possible avec les labels de features
        _cbar = dict(
            title=dict(text=f"<b>{{}}</b>", side="top",
                       font=dict(size=11, family="Arial")),
            thickness=10, len=max(0.10, min(0.18, 140 / plot_w)),
            orientation="h",
            x=0.0, xanchor="left", y=1.0, yanchor="top",
            tickfont=dict(size=9, family="Arial"), nticks=5,
            outlinecolor="#333", outlinewidth=0.8, ticks="outside", ticklen=3,
        )
        _legend_x = 1.0 + _lab_off + _pad

    fig = _make_subplots(
        rows=3, cols=2,
        row_heights=[dend_top_frac, ann_frac, heat_frac],
        column_widths=[0.11, 0.89],
        horizontal_spacing=0.006, vertical_spacing=0.008,
    )

    # 3a — dendrogrammes (1 trace chacun)
    fig.add_trace(_dend_single_trace(col_dend, "top"), row=1, col=2)
    fig.add_trace(_dend_single_trace(row_dend, "left"), row=3, col=1)

    # 3b — bande d'annotation (1 trace) + légende groupée
    ann_values = [class_labels_ord]
    for m in valid_meta:
        col_vals = data_meta_orig[m].values
        ann_values.append([col_vals[i] for i in col_order])

    band_trace, legend_traces, band_y = _build_annotation_band(
        ann_labels, ann_values, class_colors, n_samples)
    fig.add_trace(band_trace, row=2, col=2)
    for t in legend_traces:
        fig.add_trace(t)

    # 3c — heatmap principal
    if diverging:
        colorscale = _diverging_colorscale("RdBu_r")
        cb_title = "Z-score"
    else:
        colorscale = _continuous_colorscale(custom_colors)
        cb_title = "Z-score"

    if robust:
        zmin, zmax = _robust_sym_limits(matrix_ord)
    else:
        m = float(_np.nanmax(_np.abs(matrix_ord))) or 1.0
        zmin, zmax = (-m, m) if diverging else (float(matrix_ord.min()),
                                                float(matrix_ord.max()))

    n_cells = n_samples * n_features
    if n_cells <= HOVER_MAX_CELLS:
        s = _np.asarray(sample_labels_ord)
        f = _np.asarray(feature_labels_ord)
        zs = _np.round(matrix_ord.T.astype(_np.float64), 2).astype(str)
        hover = [[f"{f[i]}<br>{s[j]}<br>z = {zs[i, j]}" for j in range(n_samples)]
                 for i in range(n_features)]
        hover_kw = dict(text=hover, hovertemplate="%{text}<extra></extra>")
    else:
        hover_kw = dict(hovertemplate="z = %{z:.2f}<extra></extra>")

    if cell_border is None:
        cell_border = (n_samples <= 80 and n_features <= 80)
    gap = 0.6 if cell_border else 0

    fig.add_trace(_go.Heatmap(
        z=matrix_ord.T, x=list(range(n_samples)), y=list(range(n_features)),
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=_cbar,
        xgap=gap, ygap=gap, **hover_kw,
    ), row=3, col=2)

    # ── 4. Axes ─────────────────────────────────────────────────────────────
    blank = dict(showticklabels=False, showgrid=False, zeroline=False,
                 showline=False, ticks="")
    for r, c in [(1, 1), (2, 1), (1, 2), (3, 1)]:
        fig.update_xaxes(**blank, row=r, col=c)
        fig.update_yaxes(**blank, row=r, col=c)

    fig.update_xaxes(range=[-0.5, n_samples - 0.5], **blank, row=1, col=2)
    fig.update_yaxes(autorange=True, **blank, row=1, col=2)
    fig.update_yaxes(range=[-0.5, n_features - 0.5], autorange=False, **blank,
                     row=3, col=1)
    fig.update_xaxes(autorange=True, **blank, row=3, col=1)

    fig.update_xaxes(range=[-0.5, n_samples - 0.5], **blank, row=2, col=2)
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=9, family="Arial"),
                     showgrid=False, zeroline=False, showline=False, ticks="",
                     row=2, col=2)

    fig.update_xaxes(
        tickmode="array", tickvals=list(range(n_samples)),
        ticktext=sample_labels_ord if show_sample_names else [""] * n_samples,
        tickangle=90, tickfont=dict(size=9, family="Arial", color="#111"),
        showticklabels=show_sample_names, showgrid=False, zeroline=False,
        showline=True, linecolor="#333", linewidth=0.8, ticks="outside", ticklen=2,
        range=[-0.5, n_samples - 0.5], row=3, col=2,
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(n_features)),
        ticktext=feature_labels_ord if show_feature_names else [""] * n_features,
        tickfont=dict(size=9, family="Arial", color="#111"),
        showticklabels=show_feature_names, showgrid=False, zeroline=False,
        showline=True, linecolor="#333", linewidth=0.8, ticks="outside", ticklen=2,
        range=[-0.5, n_features - 0.5], side=feature_label_side,
        automargin=False, row=3, col=2,
    )

    # ── 5. Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=height, width=None if fit_width else width,
        margin=dict(l=left_px, r=right_px, t=top_px, b=bottom_px),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#111", family="Arial", size=11),
        hovermode="closest", dragmode="zoom",
        showlegend=True,
        legend=dict(
            x=_legend_x, y=1.0 if colorbar_pos == "right" else 0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=10, family="Arial"),
            tracegroupgap=10, itemsizing="constant", itemclick=False,
            itemdoubleclick=False,
        ),
        uirevision=capture_name or "heatmap",
    )

    if caption:
        fig.add_annotation(text=caption, xref="paper", yref="paper",
                           x=0.5, y=-0.05, showarrow=False,
                           font=dict(size=10, color="#666", family="Arial"))

    # ── 6. Rendu ────────────────────────────────────────────────────────────
    _st.plotly_chart(fig, use_container_width=fit_width, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png", "scale": 3,
            "width": width, "height": height,
            "filename": capture_name or "heatmap",
        },
        "modeBarButtonsToAdd": ["toggleSpikelines"],
    })

    # ── 7. PNG statique + contrats session_state (inchangés) ────────────────
    png_key = f"dl_heatmap_png_{capture_name or 'heatmap'}"
    bytes_key = f"{capture_name}_png_bytes" if capture_name else None

    if capture_name:
        _st.session_state[f"_report_{capture_name}"] = ("plotly", fig)
        _st.session_state[capture_name] = fig

    def _png():
        return _build_static_png_fast(
            matrix_ord=matrix_ord, raw_ord=export_raw_ord,
            feature_labels_ord=feature_labels_ord,
            sample_labels_ord=sample_labels_ord,
            class_labels_ord=class_labels_ord,
            class_colors=class_colors, custom_colors=custom_colors,
            data_meta_orig=data_meta_orig, meta_cols=valid_meta, col_order=col_order,
        )

    if large:
        if _st.button("📥 Generate & Download Heatmap PNG",
                      key=f"gen_png_{capture_name or 'hm'}",
                      help="Large dataset — PNG generation may take a few seconds."):
            with _st.spinner("Building high-resolution PNG…"):
                img = _png()
                if bytes_key:
                    _st.session_state[bytes_key] = img
                _st.download_button("📥 Download PNG", data=img,
                                    file_name="heatmap.png", mime="image/png",
                                    key=f"{png_key}_dl")
    else:
        img = _png()
        if bytes_key:
            _st.session_state[bytes_key] = img
        _st.download_button("📥 Download Heatmap as PNG", data=img,
                            file_name="heatmap.png", mime="image/png", key=png_key)

    _gc.collect()

    if capture_name:
        _st.session_state[f"{capture_name}_dend_data"] = {
            "row_dend": row_dend, "row_link": row_link, "n_features": n_features,
            "feature_labels_ord": feature_labels_ord, "matrix_ord": matrix_ord,
            "sample_labels_ord": sample_labels_ord,
            "class_labels_ord": class_labels_ord, "data_original": export_raw_ord,
            # nécessaires au panneau "Samples — horizontal dendrogram"
            "col_dend": col_dend, "col_link": col_link, "n_samples": n_samples,
        }
