"""
Software Name: Profiler
Module Name: Features importance
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 11/03/2026
Version: 1.3.0

Context:
This module is part of the "Profiler" project, originally developed for a web version (https://prism-profiler.univ-lille.fr) and now adapted for a desktop version (profiler_desktop_GUI).
It is designed for archiving on Zenodo and integration into GitHub releases.

License: l’Agence pour la Protection des Programmes IDDN (InterDeposit Digital Number) : FR2 .0013 .0300044 .0005 .S6 .C7 .20258 .0009 .312301
Citation:
If Profiler or this module (a part of Profiler) is used in a publication, please cite:
Zirem, Y. (2025). Profiler: an open web platform for multi-omics analysis. Journal of Bioinformatics. [DOI or Zenodo/GitHub link available in the article].

Links:
- GitHub temporary Repository: https://github.com/yanisZirem/Profiler_v1_requests_datatests

"""

# --- Standard library ---
import io
import gc
from itertools import combinations


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
            bg = shap.kmeans(X_transformed, min(50, X_transformed.shape[0]))
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

    st.markdown("#### 🐝 SHAP Beeswarm")
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

    st.markdown("#### 📊 SHAP Feature Importance (Bar)")
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
    classes   = sorted(data[label].dropna().unique())
    pairs     = list(combinations(classes, 2))
    n_cls     = len(classes)
    n_pairs   = len(pairs)
    color_map = {c: class_colors.get(c, "#636EFA") for c in classes}

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



def plot_volcano(volcano_data, highlight_features=True, p_value_threshold=0.05, fold_change_threshold=2.0, capture_name=None):
    color_map = {
        'Upregulated': 'red',
        'Downregulated': 'blue',
        'Non-Significant': 'gray'
    }

    # Create a unique color for each comparison if there are multiple classes
    if 'Upregulated' not in volcano_data['Color_Group'].values:
        unique_comparisons = volcano_data['Comparison'].unique()
        color_map.update({
            comp: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, comp in enumerate(unique_comparisons)
        })

    fig = px.scatter(
        volcano_data, 
        x='Log2 Fold Change', 
        y='-Log10 P-Value',
        text='Feature' if highlight_features else None,
        color='Color_Group',
        title='Volcano Plot',
        hover_data=['Feature', 'Comparison', 'Regulation Type'],
        labels={
            'Log2 Fold Change': 'Log2 Fold Change',
            '-Log10 P-Value': '-Log10 P-Value'
        },
        color_discrete_map=color_map
    )

    # Customize points
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='black')))
    
    # Add significance cutoffs
    fig.add_hline(
        y=-np.log10(p_value_threshold),
        line_dash="dash", 
        annotation_text=f"p={p_value_threshold}",
        annotation_position="top left",
        annotation_font=dict(color="black", size=14)
    )
    fig.add_vline(
        x=fold_change_threshold, 
        line_dash="dash", 
        annotation_text=f"{fold_change_threshold}-fold",
        annotation_position="top right",
        annotation_font=dict(color="black", size=14)
    )
    fig.add_vline(
        x=-fold_change_threshold, 
        line_dash="dash", 
        annotation_text=f"-{fold_change_threshold}-fold",
        annotation_position="top left",
        annotation_font=dict(color="black", size=14)
    )

    # Adjust text labels for features
    if highlight_features:
        fig.update_traces(textposition='top center', textfont=dict(color='black', size=12))



    fig.update_layout(
        title=dict(font=dict(size=24, color="black")),
        font=dict(size=20, color="black"),
        legend=dict(font=dict(size=18, color="black")),
        xaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=18, color="black")),
        yaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=18, color="black")),
        plot_bgcolor="white",
        autosize=True
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
        r for r in joblib.Parallel(n_jobs=-1, prefer='threads')(
            joblib.delayed(_check_peak)(c) for c in cols
        ) if r is not None
    ]
    return peak_features







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import streamlit as st
import base64

def plot_heatmap_samples(data, class_colors, selected_features, custom_colors,
                         show_sample_names=True, caption=None, capture_name=None):
    """
    Fully Plotly-native interactive heatmap with hierarchical clustering.
    • Dendrograms on both axes (scipy fastcluster)
    • Class colour bar at top
    • Stored in st.session_state as ("plotly", fig) for HTML report embedding
    • Download as interactive HTML or static PNG
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from sklearn.preprocessing import StandardScaler
    import streamlit as st

    # ── Validate ───────────────────────────────────────────────────────────
    data = data.copy()
    data.columns = data.columns.astype(str)
    selected_features = [str(f) for f in selected_features]
    selected_features = _resolve_features(data, selected_features)
    missing = [f for f in selected_features if f not in data.columns]
    if missing:
        st.error(f"Invalid features: {', '.join(missing[:8])}")
        return

    features = list(selected_features)
    mat = data[features].replace([np.inf, -np.inf], np.nan)
    mat = mat.fillna(mat.mean())
    scaler = StandardScaler()
    mat_z = scaler.fit_transform(mat.values.astype("float32"))   # (n_samples, n_feat)

    n_samp, n_feat = mat_z.shape
    class_labels = data["Class"].astype(str).tolist()

    # ── Hierarchical clustering ────────────────────────────────────────────
    def _linkage(m, axis=0):
        if m.shape[axis] < 2:
            return None, list(range(m.shape[axis]))
        X_ = m if axis == 0 else m.T
        try:
            import fastcluster as fc
            Z = fc.linkage(X_, method="ward", metric="euclidean")
        except Exception:
            Z = linkage(X_, method="ward", metric="euclidean")
        order = leaves_list(Z)
        return Z, order

    Z_col, col_ord = _linkage(mat_z, axis=0)   # samples
    Z_row, row_ord = _linkage(mat_z, axis=1)   # features

    mat_r = mat_z[np.ix_(col_ord, row_ord)]
    feat_r  = [features[i]     for i in row_ord]
    samp_r  = [class_labels[i] for i in col_ord]
    file_r  = data["File"].astype(str).iloc[col_ord].tolist() if "File" in data.columns else samp_r

    # ── Colour scale from custom_colors list ──────────────────────────────
    n_stops = len(custom_colors)
    colorscale = [[i / (n_stops - 1), c] for i, c in enumerate(custom_colors)]

    # ── Main heatmap ───────────────────────────────────────────────────────
    hover = np.round(mat_r, 3)
    hovertext = [[
        f"<b>Sample:</b> {file_r[ci]}<br>"
        f"<b>Class:</b>  {samp_r[ci]}<br>"
        f"<b>Feature:</b> {feat_r[ri]}<br>"
        f"<b>Z-score:</b> {hover[ci, ri]}"
        for ri in range(n_feat)] for ci in range(n_samp)]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=mat_r,
        x=feat_r,
        y=file_r if show_sample_names else [str(i) for i in range(n_samp)],
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(
            title=dict(text="Z-score", font=dict(size=12, family="Arial")),
            tickfont=dict(size=11, family="Arial"),
            len=0.7, x=1.01,
        ),
        hovertext=hovertext,
        hoverinfo="text",
        xgap=0.5, ygap=0.5,
    ))

    # ── Class colour annotation bar (top) ─────────────────────────────────
    if class_colors:
        unique_cls = list(dict.fromkeys(samp_r))
        cls_y      = [0] * n_samp
        cls_colors = [class_colors.get(c, "#aaaaaa") for c in samp_r]

        fig.add_trace(go.Bar(
            x=feat_r,
            y=[1] * n_feat,
            marker_color=cls_colors[0],   # placeholder — overridden below
            showlegend=False,
            visible=False,
        ))

        # One invisible scatter per class for the legend
        seen = set()
        for cls in samp_r:
            if cls not in seen:
                seen.add(cls)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=class_colors.get(cls, "#aaa"),
                                symbol="square"),
                    name=cls,
                    showlegend=True,
                ))

    # ── Layout ────────────────────────────────────────────────────────────
    show_x_ticks = n_feat  <= 80
    show_y_ticks = n_samp  <= 60

    fig.update_layout(
        title=dict(
            text="<b>Sample × Feature Heatmap</b>",
            font=dict(size=20, color="black", family="Arial Black"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="<b>Features</b>" if show_x_ticks else "",
            titlefont=dict(size=14, color="black", family="Arial Black"),
            tickfont=dict(size=max(8, 13 - n_feat // 20), color="black", family="Arial"),
            showticklabels=show_x_ticks,
            tickangle=-45 if n_feat > 20 else 0,
            showgrid=False, showline=True, linecolor="black", mirror=True,
        ),
        yaxis=dict(
            title="<b>Samples</b>" if show_y_ticks else "",
            titlefont=dict(size=14, color="black", family="Arial Black"),
            tickfont=dict(size=max(8, 13 - n_samp // 15), color="black", family="Arial"),
            showticklabels=show_y_ticks,
            showgrid=False, showline=True, linecolor="black", mirror=True,
            autorange="reversed",
        ),
        legend=dict(
            title=dict(text="<b>Class</b>", font=dict(size=12, family="Arial Black")),
            font=dict(size=12, family="Arial"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black", borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(500, min(2000, 160 + n_samp * 22 + 50)),
        width=max(600, min(2200, 200 + n_feat * 18)),
        margin=dict(l=130, r=80, t=80, b=120),
    )

    if caption:
        fig.add_annotation(
            text=caption,
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=11, color="#555", family="Arial"),
            xanchor="center",
        )

    # ── Display & store ───────────────────────────────────────────────────
    st.plotly_chart(fig, use_container_width=True)

    # Store for report
    key = capture_name or "heatmap_samples"
    st.session_state[f"_report_{key}"] = ("plotly", fig)

    # ── Download buttons ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "📥 Download Interactive HTML",
            html_bytes,
            file_name="heatmap.html",
            mime="text/html",
        )
    with col2:
        try:
            img_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                "📥 Download PNG (2×)",
                img_bytes,
                file_name="heatmap.png",
                mime="image/png",
            )
        except Exception:
            st.info("Install `kaleido` for PNG export: `pip install kaleido`")

    gc.collect()


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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_plotly_colorscale(custom_colors: list, n=256):
    cmap = LinearSegmentedColormap.from_list("cc", custom_colors, N=n)
    return [[i / (n - 1), to_hex(cmap(i / (n - 1)))] for i in range(n)]


def _dend_to_traces(dend, n_leaves, orientation="top", color="#555"):
    """
    Convert scipy dendrogram icoord/dcoord to Plotly Scatter traces.
    orientation="top"  → sample dendrogram: x=leaf, y=height
    orientation="left" → feature dendrogram: x=-height (mirrored), y=leaf
    """
    traces = []
    icoord = np.array(dend["icoord"])
    dcoord = np.array(dend["dcoord"])

    def norm(x):
        # scipy places leaves at 5, 15, 25 … → map to 0, 1, 2 …
        return (x - 5) / 10

    for xs, ys in zip(icoord, dcoord):
        xs_n = [norm(v) for v in xs]
        if orientation == "top":
            traces.append(go.Scatter(
                x=xs_n, y=list(ys),
                mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="skip", showlegend=False,
            ))
        else:
            # Mirror height on x so the tree opens toward the heatmap (root at left)
            traces.append(go.Scatter(
                x=[-v for v in ys], y=xs_n,
                mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="skip", showlegend=False,
            ))
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_samples(
    data: pd.DataFrame,
    class_colors: dict,
    selected_features: list,
    custom_colors: list,
    show_sample_names: bool = True,
    caption: str = None,
    capture_name: str = None,
    meta_annotation_cols: list = None,
):
    # ── 1. Data prep ──────────────────────────────────────────────────────────
    data = data.copy()
    data.columns = data.columns.astype(str)
    selected_features = [str(f) for f in selected_features]
    selected_features = _resolve_features(data, selected_features)

    missing = [f for f in selected_features if f not in data.columns]
    if missing:
        st.error(f"Invalid features: {', '.join(missing)}")
        return

    features = list(selected_features)
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data[features] = data[features].fillna(data[features].mean())

    scaler = StandardScaler()
    Z = scaler.fit_transform(data[features])
    data[features] = Z

    if np.isnan(Z).any() or np.isinf(Z).any():
        st.error("Missing/infinite values after preprocessing.")
        return

    matrix = data[features].values   # (n_samples, n_features)
    if "ID" in data.columns:
        sample_labels = data["ID"].astype(str).tolist()
    elif data.index.name and data.index.name != "index":
        sample_labels = data.index.astype(str).tolist()
    else:
        # Fallback: Class + position index for readability
        sample_labels = [
            f"{cls} #{i}"
            for i, cls in enumerate(data["Class"].astype(str).tolist())
        ]
    class_labels = data["Class"].astype(str).tolist()
    n_samples, n_features = matrix.shape

    # ── 2. Hierarchical clustering ────────────────────────────────────────────
    col_link  = linkage(pdist(matrix,   metric="euclidean"), method="ward")
    col_dend  = dendrogram(col_link,  no_plot=True)
    col_order = col_dend["leaves"]

    row_link  = linkage(pdist(matrix.T, metric="euclidean"), method="ward")
    row_dend  = dendrogram(row_link,  no_plot=True)
    row_order = row_dend["leaves"]

    matrix_ord         = matrix[np.ix_(col_order, row_order)]
    sample_labels_ord  = [sample_labels[i] for i in col_order]
    class_labels_ord   = [class_labels[i]  for i in col_order]
    feature_labels_ord = [features[i]      for i in row_order]

    # ── 3. Build figure — rows adapt to meta annotations ─────────────────────
    #  Structure:  row1=top-dend | row2=annotation strips | row3=heatmap
    #  col1=left-dend | col2=gap | col3=content

    colorscale = _make_plotly_colorscale(custom_colors)
    vmin = matrix_ord.min()
    vmax = matrix_ord.max()

    # ── Resolve valid meta columns & build their colour mappings ─────────────
    _AUTO_PAL = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
                 "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
    meta_annotation_cols = meta_annotation_cols or []
    valid_meta = [m for m in meta_annotation_cols if m in data.columns]

    # Annotation strip labels: Class first, then meta columns
    ann_labels  = ["Class"] + valid_meta
    n_ann_rows  = len(ann_labels)   # ≥1

    # Height allocation: top-dend=12%, each ann strip proportional, heatmap=rest
    ann_strip_frac = 0.03           # 3% per annotation strip
    total_ann = ann_strip_frac * n_ann_rows
    top_frac   = 0.12
    heat_frac  = max(0.40, 1.0 - top_frac - total_ann)

    row_heights = [top_frac] + [ann_strip_frac] * n_ann_rows + [heat_frac]
    n_rows = 2 + n_ann_rows         # top-dend + N strip rows + heatmap

    fig = make_subplots(
        rows=n_rows, cols=3,
        column_widths=[0.10, 0.005, 0.895],
        row_heights=row_heights,
        horizontal_spacing=0.002,
        vertical_spacing=0.002,
    )

    heatmap_row = n_rows            # last row = main heatmap

    # ── 3a. Sample dendrogram (top) ───────────────────────────────────────────
    for tr in _dend_to_traces(col_dend, n_samples, orientation="top", color="#444"):
        fig.add_trace(tr, row=1, col=3)

    # ── 3b. Feature dendrogram (left) ────────────────────────────────────────
    for tr in _dend_to_traces(row_dend, n_features, orientation="left", color="#444"):
        fig.add_trace(tr, row=heatmap_row, col=1)

    # ── 3c. Annotation strips (row 2 … row 1+n_ann_rows) ────────────────────
    # We collect legend traces for meta (categorical only) to show in layout
    legend_annotations = []

    for ann_idx, ann_label in enumerate(ann_labels):
        strip_row = 2 + ann_idx     # row 2, 3, … for each annotation bar

        if ann_label == "Class":
            # ── Class strip (same as original) ──────────────────────────────
            unique_cls = list(dict.fromkeys(class_labels_ord))
            n_cls = len(unique_cls)
            cls_idx = {c: i for i, c in enumerate(unique_cls)}
            z_cls = [[cls_idx[c] for c in class_labels_ord]]

            if n_cls == 1:
                cls_cs = [[0.0, class_colors.get(unique_cls[0], "#aaa")],
                           [1.0, class_colors.get(unique_cls[0], "#aaa")]]
            else:
                cls_cs = []
                for i, c in enumerate(unique_cls):
                    t0, t1 = i / n_cls, (i + 1) / n_cls
                    cls_cs += [[t0, class_colors.get(c, "#aaa")],
                                [t1, class_colors.get(c, "#aaa")]]

            fig.add_trace(go.Heatmap(
                z=z_cls, x=list(range(n_samples)), y=["Class"],
                colorscale=cls_cs, zmin=0, zmax=n_cls, showscale=False,
                hovertemplate="<b>Class:</b> %{customdata}<extra></extra>",
                customdata=[class_labels_ord], xgap=0, ygap=0,
            ), row=strip_row, col=3)

            # Add class legend entries via invisible scatter
            for cls_name in unique_cls:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=class_colors.get(cls_name, "#aaa"),
                                symbol="square"),
                    name=f"Class: {cls_name}", legendgroup=f"cls_{cls_name}",
                    showlegend=True,
                ))

        else:
            # ── Meta strip ───────────────────────────────────────────────────
            col_vals = data[ann_label].values
            reordered = [col_vals[i] for i in col_order]

            import pandas as _pd_local
            if _pd_local.api.types.is_numeric_dtype(data[ann_label]):
                # Continuous → viridis gradient
                z_meta = [[float(v) if v == v else 0.0 for v in reordered]]
                fig.add_trace(go.Heatmap(
                    z=z_meta, x=list(range(n_samples)), y=[ann_label],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(
                        title=dict(text=ann_label, side="right"),
                        thickness=8, len=ann_strip_frac * 4,
                        y=1.0 - top_frac - ann_strip_frac * (ann_idx + 0.5),
                        x=1.22, xanchor="left", tickfont=dict(size=8),
                    ),
                    hovertemplate=f"<b>{ann_label}:</b> %{{z:.2f}}<extra></extra>",
                    xgap=0, ygap=0,
                ), row=strip_row, col=3)

            else:
                # Categorical → discrete colours
                uniq_vals = sorted(set(v for v in reordered if v == v))  # drop NaN
                val_map = {v: _AUTO_PAL[i % len(_AUTO_PAL)]
                           for i, v in enumerate(uniq_vals)}
                n_v = len(uniq_vals)
                val_idx = {v: i for i, v in enumerate(uniq_vals)}
                z_meta = [[val_idx.get(v, 0) for v in reordered]]

                if n_v == 1:
                    meta_cs = [[0.0, val_map[uniq_vals[0]]],
                                [1.0, val_map[uniq_vals[0]]]]
                else:
                    meta_cs = []
                    for i, v in enumerate(uniq_vals):
                        t0, t1 = i / n_v, (i + 1) / n_v
                        meta_cs += [[t0, val_map[v]], [t1, val_map[v]]]

                hover_meta = [f"<b>{ann_label}:</b> {v}" for v in reordered]
                fig.add_trace(go.Heatmap(
                    z=z_meta, x=list(range(n_samples)), y=[ann_label],
                    colorscale=meta_cs, zmin=0, zmax=n_v, showscale=False,
                    text=[hover_meta], hovertemplate="%{text}<extra></extra>",
                    xgap=0, ygap=0,
                ), row=strip_row, col=3)

                # Legend via invisible scatter
                for v in uniq_vals:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode="markers",
                        marker=dict(size=10, color=val_map[v], symbol="square"),
                        name=f"{ann_label}: {v}",
                        legendgroup=f"meta_{ann_label}_{v}",
                        showlegend=True,
                    ))

    # ── 3d. Main heatmap ──────────────────────────────────────────────────────
    hover_text = [
        [
            f"<b>Sample:</b> {sample_labels_ord[j]}<br>"
            f"<b>Class:</b>  {class_labels_ord[j]}<br>"
            f"<b>Feature:</b> {feature_labels_ord[i]}<br>"
            f"<b>Z-score:</b> {matrix_ord[j, i]:.3f}"
            for j in range(n_samples)
        ]
        for i in range(n_features)
    ]

    fig.add_trace(
        go.Heatmap(
            z=matrix_ord.T.tolist(),
            x=list(range(n_samples)),
            y=list(range(n_features)),
            colorscale=colorscale,
            zmin=vmin, zmax=vmax,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(
                title=dict(text="Z-score", side="right"),
                thickness=12,
                len=0.45,
                x=1.25,
                xanchor="left",
                tickfont=dict(size=9),
            ),
            xgap=0.4, ygap=0.4,
        ),
        row=heatmap_row, col=3,
    )

    # ── 4. Axes ───────────────────────────────────────────────────────────────
    # Hide all tick labels on dendrogram & gap panels
    for r in range(1, n_rows + 1):
        for c in [1, 2]:
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)

    # Top dendrogram
    fig.update_xaxes(range=[-0.5, n_samples - 0.5], showticklabels=False, row=1, col=3)
    fig.update_yaxes(autorange=True, showticklabels=False, row=1, col=3)

    # Left dendrogram
    fig.update_yaxes(range=[-0.5, n_features - 0.5], autorange=False, row=heatmap_row, col=1)
    fig.update_xaxes(autorange=True, row=heatmap_row, col=1)

    # Annotation strips — link x axis to heatmap x
    for strip_row in range(2, 2 + n_ann_rows):
        fig.update_xaxes(range=[-0.5, n_samples - 0.5],
                         showticklabels=False, row=strip_row, col=3)
        # Show the annotation label on y-axis (strip label)
        fig.update_yaxes(showticklabels=True, tickfont=dict(size=8, color="black"),
                         showgrid=False, row=strip_row, col=3)

    # Main heatmap x: sample names
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_samples)),
        ticktext=sample_labels_ord,
        tickangle=90,
        tickfont=dict(size=8),
        showgrid=False, zeroline=False,
        range=[-0.5, n_samples - 0.5],
        row=heatmap_row, col=3,
    )

    # Heatmap y: feature names on right side, bottom→top (index 0 = bottom)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_features)),
        ticktext=feature_labels_ord,
        tickfont=dict(size=8),
        showgrid=False, zeroline=False,
        range=[-0.5, n_features - 0.5],
        side="right",
        row=heatmap_row, col=3,
    )

    # ── 5. Layout ─────────────────────────────────────────────────────────────
    plot_height = max(550, min(2000, 220 + n_features * 20 + n_ann_rows * 30))

    fig.update_layout(
        height=plot_height,
        margin=dict(l=10, r=220, t=20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#222", family="Arial"),
        dragmode="zoom",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=1.28, y=1.0, xanchor="left", yanchor="top",
            bgcolor="white", bordercolor="#ccc", borderwidth=1,
            font=dict(size=9, color="black"),
            tracegroupgap=4,
        ),
    )

    if caption:
        fig.add_annotation(
            text=caption,
            xref="paper", yref="paper",
            x=0.5, y=-0.04,
            showarrow=False,
            font=dict(size=11, color="gray"),
        )

    # ── 6. Render ─────────────────────────────────────────────────────────────
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "toImageButtonOptions": {"format": "png", "scale": 3},
    })

    # ── 7. Static PNG (exact original seaborn version) ────────────────────────
    img_bytes = _build_static_png_original(
        data_orig=data,
        features=features,
        class_colors=class_colors,
        custom_colors=custom_colors,
        show_sample_names=show_sample_names,
        n_samples=n_samples,
        n_features=n_features,
        meta_annotation_cols=valid_meta,
    )

    # Store plotly fig — two purposes:
    #   1. Keyed by capture_name for the HTML report pipeline (_report_*)
    #   2. Keyed by capture_name directly so the GUI can re-display it across
    #      reruns (e.g. when a download_button is clicked) without re-running
    #      the full computation.
    if capture_name:
        st.session_state[f"_report_{capture_name}"] = ("plotly", fig)
        st.session_state[capture_name] = fig          # ← direct re-display key
        # Store PNG bytes so GUI persistent blocks can re-render the download
        # button across reruns without needing kaleido or re-running the computation.
        st.session_state[f"{capture_name}_png_bytes"] = img_bytes

    st.download_button(
        label="📥 Download Heatmap as PNG",
        data=img_bytes,
        file_name="heatmap.png",
        mime="image/png",
        key=f"dl_heatmap_png_{capture_name or 'heatmap'}",
    )

    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Exact replica of the original seaborn static output
# ─────────────────────────────────────────────────────────────────────────────

def _build_static_png_original(
    data_orig, features, class_colors, custom_colors,
    show_sample_names, n_samples, n_features,
    meta_annotation_cols=None,
):
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    _AUTO_PAL = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
                 "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

    cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
    vmin = data_orig[features].min().min()
    vmax = data_orig[features].max().max()
    fontsize = max(9, 18 - max(n_samples, n_features) // 10)

    meta_annotation_cols = meta_annotation_cols or []
    valid_meta = [m for m in meta_annotation_cols if m in data_orig.columns]

    col_colors_list = [data_orig["Class"].map(class_colors).rename("Class")]
    legend_handles = [
        mpatches.Patch(color=c, label=f"Class: {cls}")
        for cls, c in class_colors.items()
        if cls in data_orig["Class"].unique()
    ]
    for m in valid_meta:
        vals = data_orig[m]
        if vals.dtype.kind in "iuf":
            norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
            col_colors_list.append(
                vals.map(lambda v: mcolors.to_hex(plt.cm.viridis(norm(v)))).rename(m)
            )
        else:
            uniq = sorted(vals.dropna().unique())
            m_map = {v: _AUTO_PAL[i % len(_AUTO_PAL)] for i, v in enumerate(uniq)}
            col_colors_list.append(vals.map(m_map).rename(m))
            legend_handles += [mpatches.Patch(color=c, label=f"{m}: {v}") for v, c in m_map.items()]

    import pandas as _pd_local
    col_colors_df = _pd_local.concat(col_colors_list, axis=1) if len(col_colors_list) > 1 else col_colors_list[0]

    g = sns.clustermap(
        data_orig[features].T,
        cmap=cmap, square=False,
        col_cluster=True, row_cluster=True, center=0, z_score=None,
        col_colors=col_colors_df,
        cbar_kws={"shrink": 0.6, "label": "Z-score"},
        yticklabels=True if n_features <= 60 else False,
        xticklabels=False, vmin=vmin, vmax=vmax,
        figsize=(max(12, n_samples * 0.35), max(8, n_features * 0.25)),
    )

    if show_sample_names and n_samples <= 50:
        col_order = g.dendrogram_col.reordered_ind
        ordered_labels = data_orig["File"].iloc[col_order].values if "File" in data_orig.columns             else data_orig.index[col_order]
        g.ax_heatmap.set_xticks(np.arange(len(ordered_labels)) + 0.5)
        g.ax_heatmap.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=max(7, fontsize-2))

    if n_features <= 60:
        g.ax_heatmap.set_yticklabels(
            [lbl.get_text() for lbl in g.ax_heatmap.get_yticklabels()],
            fontsize=fontsize, color="black",
        )

    try:
        ann_axes = g.ax_col_colors if isinstance(g.ax_col_colors, list) else [g.ax_col_colors]
        for ax_ann, lbl in zip(ann_axes, ["Class"] + valid_meta):
            ax_ann.set_ylabel(lbl, fontsize=fontsize, rotation=0,
                              labelpad=50, va="center", ha="right", color="black")
            ax_ann.yaxis.set_label_position("left")
    except Exception:
        pass

    if legend_handles:
        g.ax_heatmap.legend(handles=legend_handles, loc="upper left",
                            bbox_to_anchor=(1.18, 1.0), frameon=True,
                            fontsize=max(8, fontsize-1), title="Legend",
                            title_fontsize=fontsize)

    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close(g.fig)
    return buf.getvalue()
